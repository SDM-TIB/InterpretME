import hashlib
import json
import os
import os.path
import time
from pathlib import Path

import pandas as pd
import validating_models.stats as stats
from validating_models.checker import Checker
from validating_models.constraint import ShaclSchemaConstraint
from validating_models.dataset import BaseDataset, ProcessedDataset
from validating_models.models.decision_tree import get_shadow_tree_from_checker
from validating_models.shacl_validation_engine import ReducedTravshaclCommunicator

import InterpretME.utils as utils
from InterpretME import preprocessing_data, sampling_strategy, classification
from InterpretME.semantification import rdf_semantification
from InterpretME.upload import upload_to_virtuoso


def constraint_md5_sum(constraint):
    """Generates checksum for SHACL constraints.

    Parameters
    ----------
    constraint : ShaclSchemaConstraint
        List of constraints needs to be checked.
    Returns
    -------
    hash value

    """
    paths = Path(constraint.shape_schema_dir).glob('**/*')
    sorted_shape_files = sorted([path for path in paths if path.is_file()])
    hash_md5 = hashlib.md5()
    for file in sorted_shape_files:
        with open(file, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    hash_md5.update(constraint.target_shape.encode(encoding='UTF-8', errors='ignore'))
    return hash_md5.hexdigest()

def read_dataset(input_data,st):
    path = input_data['path_to_data']
    if os.path.splitext(path)[1].lower() == '.json':
        with stats.measure_time('PIPE_DATASET_EXTRACTION'):
            annotated_dataset = pd.read_json(path)
    else:  # assuming CSV
        with stats.measure_time('PIPE_DATASET_EXTRACTION'):
            annotated_dataset = pd.read_csv(path)
    seed_var = input_data['Index_var']
    sampling = input_data['sampling_strategy']
    cv = input_data['cross_validation_folds']
    test_split = input_data['test_split']
    num_imp_features = input_data['number_important_features']
    train_model = input_data['model']
    min_max_depth = input_data.get('min_max_depth', 4)
    max_max_depth = input_data.get('max_max_depth', 6)
    independent_var = []
    dependent_var = []
    classes = []
    class_names = []
    definition = []

    # Create the dataset
    annotated_dataset = annotated_dataset.set_index(seed_var)
    for k, v in enumerate(input_data['Independent_variable']):
        independent_var.append(v)
        definition.append(v)
    for k, v in enumerate(input_data['Dependent_variable']):
        dependent_var.append(v)
        definition.append(v)

    features = independent_var + dependent_var

    for k, v in input_data['classes'].items():
        classes.append(v)
        class_names.append(k)

    with stats.measure_time('PIPE_OUTPUT'):
        df1 = pd.DataFrame({'features': pd.Series(features), 'definition': pd.Series(definition)})
        df1.loc[:, 'run_id'] = st
        df1 = df1.set_index('run_id')
        df1.to_csv('interpretme/files/feature_definition.csv')

        df2 = pd.DataFrame({'classes': pd.Series(classes)})
        df2.loc[:, 'run_id'] = st
        df2 = df2.set_index('run_id')
        df2.to_csv('interpretme/files/classes.csv')
    return seed_var, independent_var, dependent_var, classes, class_names, st, sampling, test_split, num_imp_features, train_model, cv, annotated_dataset, min_max_depth, max_max_depth


def read_KG(input_data, st):
    """Reads from the input knowledge graphs and extracts data.

    Extracting features and its definition from the input knowledge graphs using sparqlQuery
    and applying SHACL constraints on the input knowledge graphs to identify valid and invalid
    entities in the input knowledge graphs.

    Parameters
    ----------
    input_data : dict
        Input configuration (JSON).
    st : int
        Unique identifier as run id.

    Returns
    -------
    (str, list, list, list, list, Dataframe, list, Dataframe, int, input_data['3_valued_logic'] )
        Annotated dataset with SHACL results and class information given by the user.

    """
    endpoint = input_data['Endpoint']
    independent_var = []
    dependent_var = []
    classes = []
    class_names = []
    definition = []

    seed_var = input_data['Index_var']
    sampling = input_data['sampling_strategy']
    cv = input_data['cross_validation_folds']
    test_split = input_data['test_split']
    num_imp_features = input_data['number_important_features']
    train_model = input_data['model']
    min_max_depth = input_data.get('min_max_depth', 4)
    max_max_depth = input_data.get('max_max_depth', 6)

    # Create the dataset generating query
    query_select_clause = "SELECT "
    query_where_clause = """WHERE { """
    for k, v in input_data['Independent_variable'].items():
        independent_var.append(k)
        query_select_clause = query_select_clause + "?" + k + " "
        query_where_clause = query_where_clause + v
        definition.append(v)

    for k, v in input_data['Dependent_variable'].items():
        dependent_var.append(k)
        query_select_clause = query_select_clause + "?" + k + " "
        query_where_clause = query_where_clause + v
        target_name = k
        definition.append(v)

        query_where_clause = query_where_clause + "}"
        sparqlQuery = query_select_clause + " " + query_where_clause

        features = independent_var + dependent_var

        shacl_engine_communicator = ReducedTravshaclCommunicator(
            '', endpoint,
            {
                "backend": "travshacl",
                "start_with_target_shape": True,
                "replace_target_query": True,
                "prune_shape_network": True,
                "output_format": "simple",
                "outputs": False
            }
        )

    def hook(results):
        bindings = [{key: value['value'] for key, value in binding.items()}
                    for binding in results['results']['bindings']]
        df = pd.DataFrame.from_dict(bindings)
        for column in df.columns:
            df[column] = df[column].str.rsplit('/', n=1).str[-1]
        return df

    with stats.measure_time('PIPE_DATASET_EXTRACTION'):
        base_dataset = BaseDataset.from_knowledge_graph(endpoint, shacl_engine_communicator, sparqlQuery,
                                                    target_name, seed_var=seed_var,
                                                    raw_data_query_results_to_df_hook=hook)

    constraints = [ShaclSchemaConstraint.from_dict(constraint) for constraint in input_data['Constraints']]
    constraint_identifiers = [constraint_md5_sum(constraint) for constraint in constraints]

    utils.pbar.total += len(constraints)
    utils.pbar.set_description('SHACL Validation', refresh=True)
    with stats.measure_time('PIPE_SHACL_VALIDATION'):
        shacl_validation_results = base_dataset.get_shacl_schema_validation_results(
                constraints, rename_columns=True, replace_non_applicable_nans=True
        )
    utils.pbar.update(len(constraints))

    sample_to_node_mapping = base_dataset.get_sample_to_node_mapping().rename('node')

    annotated_dataset = pd.concat(
            (base_dataset.df, shacl_validation_results, sample_to_node_mapping), axis='columns'
    )

    annotated_dataset = annotated_dataset.drop_duplicates()
    annotated_dataset = annotated_dataset.set_index(seed_var)

    for k, v in input_data['classes'].items():
        classes.append(v)
        class_names.append(k)

    with stats.measure_time('PIPE_OUTPUT'):
        df1 = pd.DataFrame({'features': pd.Series(features), 'definition': pd.Series(definition)})
        df1.loc[:, 'run_id'] = st
        df1 = df1.set_index('run_id')
        df1.to_csv('interpretme/files/feature_definition.csv')

        df2 = pd.DataFrame({'classes': pd.Series(classes)})
        df2.loc[:, 'run_id'] = st
        df2 = df2.set_index('run_id')
        df2.to_csv('interpretme/files/classes.csv')

        dfs_shacl_results = []

        for constraint, identifier in zip(constraints, constraint_identifiers):
            df6 = pd.DataFrame(annotated_dataset.loc[:, [constraint.name]]).rename(
                columns={constraint.name: 'SHACL result'})
            df6['run_id'] = st
            df6['SHACL schema'] = constraint.shape_schema_dir
            df6['SHACL shape'] = constraint.target_shape.rsplit('/', 1)[1][:-1]  # remove the prefix from the shape name
            df6['SHACL constraint name'] = constraint.name
            df6['constraint identifier'] = identifier

            df6 = df6.reset_index()
            df6 = df6.rename(columns={df6.columns[0]: 'index'})
            dfs_shacl_results.append(df6)
            pd.concat(dfs_shacl_results, axis='rows').to_csv(
                'interpretme/files/shacl_validation_results.csv', index=False
            )

        df7 = pd.DataFrame(annotated_dataset.loc[:, ['node']])
        df7['run_id'] = st
        df7 = df7.drop_duplicates()
        df7 = df7.reset_index()
        df7 = df7.rename(columns={df7.columns[0]: 'index'})
        df7.to_csv('interpretme/files/entityAlignment.csv', index=False)

        df8 = pd.DataFrame({'endpoint': pd.Series(endpoint)})
        df8.loc[:, 'run_id'] = st
        df8 = df8.set_index('run_id')
        df8.to_csv('interpretme/files/endpoint.csv')

    annotated_dataset = annotated_dataset.drop(columns=['node'])
    num = len(input_data['Constraints'])
    annotated_dataset = annotated_dataset.iloc[:, :-num]

    return seed_var, independent_var, dependent_var, classes, class_names, annotated_dataset, constraints, base_dataset, st, input_data['3_valued_logic'], sampling, test_split, num_imp_features, train_model, cv, min_max_depth, max_max_depth


# get the start time and use it as run_id
def current_milli_time():
    return round(time.time() * 1000)


def pipeline(path_config, lime_results, server_url=None, username=None, password=None,
             sampling=None, cv=None, imp_features=None, test_split=None, model=None):
    """Executing InterpretME pipeline.

    InterpretME pipeline is the important function which helps executes all the main components of
    InterpretME including extraction of data from input knowledge graphs, preprocessing data,
    applying sampling strategy if specified by user, and generating machine learning model outputs.

    If a server URL, username, and password are given, the generated InterpretME KG is uploaded there.

    Parameters
    ----------
    path_config : str
        Path to the configuration file from input knowledge graphs.
    lime_results : str
        Path where to store LIME results in HTML format.
    server_url : str, OPTIONAL
        Server URL to upload the InterpretME knowledge graph.(e.g- Virtuoso)
    username : str, OPTIONAL
        Username to upload data to InterpretME KG.
    password : str, OPTIONAL
        Password to upload data to InterpretME KG.
    sampling : str, OPTIONAL
        Sampling strategy desired by user. Default is None (read from input .json file)
    cv : int, OPTIONAL
        Cross validation folds desired by user. Default is None (read from input .json file)
    imp_features : int, OPTIONAL
        Number of important features to train predictive model desired by user. Default is None (read from input .json file)
    test_split : float, OPTIONAL
        Training testing split of data desired by user. Default is None (read from input .json file)
    model : str, OPTIONAL
        Model to perform Stratified shuffle split desired by user. Default is None (read from input .json file)

    Returns
    -------
    dict
        Pipeline function will return a dictionary with all the results stored as
        objects which can be used for further analysis.

    """
    st = current_milli_time()
    results = {'run_id': st}

    utils.pbar = utils.tqdm(total=5, miniters=1, desc='InterpretME Pipeline', unit='task')

    if not os.path.exists('interpretme/files'):
        os.makedirs('interpretme/files')

    stats.STATS_COLLECTOR.activate(hyperparameters=[])
    stats.STATS_COLLECTOR.new_run(hyperparameters=[])

    with open(path_config, "r") as input_file_descriptor:
        input_data = json.load(input_file_descriptor)

    utils.pbar.set_description('Read input data', refresh=True)
    input_is_kg = None
    if "Endpoint" in input_data.keys() and "path_to_data" not in input_data.keys():
        # input from SPARQL endpoint
        input_is_kg = True
        seed_var, independent_var, dependent_var, classes, class_names, annotated_dataset, constraints, base_dataset, st, non_applicable_counts, sampling, train_test_split, num_imp_features, train_model, cross_validation, min_max_depth, max_max_depth = read_KG(input_data, st)
    elif "Endpoint" not in input_data.keys() and "path_to_data" in input_data.keys():
        # input from dataset
        input_is_kg = False
        seed_var, independent_var, dependent_var, classes, class_names, st, sampling, test_split, num_imp_features, train_model, cv, annotated_dataset, min_max_depth, max_max_depth = read_dataset(input_data, st)
    else:
        # error
        raise Exception("Please provide either SPARQL endpoint or Dataset")

    if sampling is None:
        sampling = "None"
    else:
        sampling = sampling

    df3 = pd.DataFrame({'sampling': pd.Series(sampling)})
    df3.loc[:, 'run_id'] = st
    df3 = df3.set_index('run_id')
    df3.to_csv('interpretme/files/sampling_strategy.csv')

    if imp_features is None:
        imp_features = num_imp_features
    else:
        imp_features = imp_features

    df4 = pd.DataFrame({'num_imp_features': pd.Series(imp_features)})
    df4.loc[:, 'run_id'] = st
    df4 = df4.set_index('run_id')
    df4.to_csv('interpretme/files/imp_features.csv')

    if cv is None:
        cv = cross_validation
    else:
        cv = cv

    df5 = pd.DataFrame({'cross_validation': pd.Series(cv)})
    df5.loc[:, 'run_id'] = st
    df5 = df5.set_index('run_id')
    df5.to_csv('interpretme/files/cross_validation.csv')

    if test_split is None:
        test_split = train_test_split
    else:
        test_split = test_split

    if model is None:
        model = train_model
    else:
        model = model
    utils.pbar.update(1)  # end of reading the input dataset

    utils.pbar.set_description('Preprocessing', refresh=True)
    with stats.measure_time('PIPE_PREPROCESSING'):
        encoded_data, encode_target = preprocessing_data.load_data(seed_var, dependent_var, classes, annotated_dataset)
    utils.pbar.update(1)

    utils.pbar.set_description('Sampling', refresh=True)
    with stats.measure_time('PIPE_SAMPLING'):
        if sampling == "None":
            sampled_data, sampled_target = encoded_data, encode_target
        else:
            sampled_data, sampled_target, results = sampling_strategy.sampling_strategy(encoded_data, encode_target,sampling, results)
    utils.pbar.update(1)

    new_sampled_data, clf, results = classification.classify(sampled_data, sampled_target, imp_features, cv, classes, st, lime_results, test_split, model, results, min_max_depth, max_max_depth)
    processed_df = pd.concat((new_sampled_data, sampled_target), axis='columns')
    processed_df.reset_index(inplace=True)

    if "Endpoint" in input_data.keys() and "path_to_data" not in input_data.keys():
        utils.pbar.total += 1
        utils.pbar.set_description('Preparing Plots Data', refresh=True)
        with stats.measure_time('PIPE_CONSTRAINT_VIZ'):
            processedDataset = ProcessedDataset.from_node_unique_columns(
                processed_df,
                base_dataset,
                base_columns=[seed_var],
                target_name='class',
                categorical_mapping={'class': {i: classes[i] for i in range(len(classes))}}
            )
            checker = Checker(clf.predict, processedDataset)

            shadow_tree = get_shadow_tree_from_checker(clf, checker)

            results['checker'] = checker
            results['shadow_tree'] = shadow_tree
            results['constraints'] = constraints
            results['non_applicable_counts'] = non_applicable_counts
        utils.pbar.update(1)

    categories_stats = ['PIPE_DATASET_EXTRACTION', 'PIPE_SHACL_VALIDATION', 'PIPE_PREPROCESSING',
                        'PIPE_SAMPLING', 'PIPE_IMPORTANT_FEATURES', 'PIPE_LIME', 'PIPE_TRAIN_MODEL',
                        'PIPE_DTREEVIZ', 'PIPE_CONSTRAINT_VIZ', 'PIPE_OUTPUT', 'join',
                        'PIPE_InterpretMEKG_SEMANTIFICATION']

    utils.pbar.set_description('Semantifying Results', refresh=True)
    with stats.measure_time('PIPE_InterpretMEKG_SEMANTIFICATION'):
        rdf_semantification(input_is_kg)
    utils.pbar.update(1)

    if server_url is not None and username is not None and password is not None:
        categories_stats.append('PIPE_InterpretMEKG_UPLOAD_VIRTUOSO')
        utils.pbar.total += 1
        utils.pbar.set_description('Uploading to Virtuoso', refresh=True)
        with stats.measure_time('PIPE_InterpretMEKG_UPLOAD_VIRTUOSO'):
            upload_to_virtuoso(run_id=st, rdf_file='./rdf-dump/interpretme.nt',
                               server_url=server_url, username=username, password=password)
        utils.pbar.update(1)

    stats.STATS_COLLECTOR.to_file('times.csv', categories=categories_stats)

    utils.pbar.set_description('InterpretME Pipeline')
    utils.pbar.close()
    return results
