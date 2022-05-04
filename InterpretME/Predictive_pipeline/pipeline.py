import hashlib
import json
import os.path
import time
import preprocessing_data
import sampling_strategy
import classification
import pandas as pd
from pathlib import Path
import sys
PACKAGE_VALIDATING_MODELS = str(Path(__file__).parent.parent.joinpath('validating_models').resolve())
sys.path.append(PACKAGE_VALIDATING_MODELS)
from validating_models.dataset import BaseDataset, ProcessedDataset
from validating_models.shacl_validation_engine import ReducedTravshaclCommunicator
from validating_models.constraint import ShaclSchemaConstraint
from validating_models.checker import Checker
import validating_models.visualizations.decision_trees as constraint_viz
from validating_models.visualizations.classification import confusion_matrix_decomposition
from validating_models.models.decision_tree import get_shadow_tree_from_checker
import validating_models.stats as stats
sys.path.remove(PACKAGE_VALIDATING_MODELS)
from argparse import ArgumentParser


def constraint_md5_sum(constraint):
    paths = Path(constraint.shape_schema_dir).glob('**/*')
    sorted_shape_files = sorted([path for path in paths if path.is_file()])
    hash_md5 = hashlib.md5()
    for file in sorted_shape_files:
        with open(file, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    hash_md5.update(constraint.target_shape.encode(encoding='UTF-8', errors='ignore'))
    return hash_md5.hexdigest()


def read_process(input_file, st):
    with open(input_file, "r") as input_file_descriptor:
        input_data = json.load(input_file_descriptor)
        # print(input_data)

        if not os.path.exists('files'):
            os.makedirs('files')
            print("The directory for files is created")

        endpoint = input_data['Endpoint']
        seed_var = input_data['Index_var']

        independent_var = []
        dependent_var = []
        classes = []
        class_names = []
        definition = []

        # Create the dataset generating query
        query_select_clause = "SELECT "
        query_where_clause = """WHERE { """
        for k, v in input_data['Independent_variable'].items():
            independent_var.append(k)
            # print("data_list:",independent_var)
            query_select_clause = query_select_clause + "?" + k + " "
            query_where_clause = query_where_clause + v
            definition.append(v)

        for k, v in input_data['Dependent_variable'].items():
            dependent_var.append(k)
            # print("target_list:", dependent_var)
            query_select_clause = query_select_clause + "?" + k + " "
            query_where_clause = query_where_clause + v
            target_name = k
            definition.append(v)

        query_where_clause = query_where_clause + "}"
        sparqlQuery = query_select_clause + " " + query_where_clause

        features = independent_var + dependent_var

        shacl_engine_communicator = ReducedTravshaclCommunicator('', endpoint, 'example/shacl_api_config.json')

        def hook(results):
            bindings = [{key: value['value'] for key, value in binding.items()}
                        for binding in results['results']['bindings']]
            df = pd.DataFrame.from_dict(bindings)
            for column in df.columns:
                df[column] = df[column].str.rsplit('/', 1).str[-1]
            return df

        with stats.measure_time('PIPE_DATASET_EXTRACTION'):
            base_dataset = BaseDataset.from_knowledge_graph(endpoint, shacl_engine_communicator, sparqlQuery,
                                                            target_name, seed_var=seed_var,
                                                            raw_data_query_results_to_df_hook=hook)

        constraints = [ShaclSchemaConstraint.from_dict(constraint) for constraint in input_data['Constraints']]
        constraint_identifiers = [constraint_md5_sum(constraint) for constraint in constraints]

        with stats.measure_time('PIPE_SHACL_VALIDATION'):
            shacl_validation_results = base_dataset.get_shacl_schema_validation_results(constraints,
                                                                                        rename_columns=True,
                                                                                        replace_non_applicable_nans=True)

        sample_to_node_mapping = base_dataset.get_sample_to_node_mapping().rename('node')

        annotated_dataset = pd.concat((base_dataset.df, shacl_validation_results, sample_to_node_mapping),
                                      axis='columns')

        annotated_dataset = annotated_dataset.drop_duplicates()
        annotated_dataset = annotated_dataset.set_index(seed_var)

        for k, v in input_data['classes'].items():
            classes.append(v)
            class_names.append(k)
        strategy = input_data['sampling_strategy']
        imp_features = input_data['number_important_features']
        cv = input_data['cross_validation_folds']

        with stats.measure_time('PIPE_OUTPUT'):
            df1 = pd.DataFrame({'features': pd.Series(features), 'definition': pd.Series(definition)})
            df1.loc[:, 'run_id'] = st
            df1 = df1.set_index('run_id')
            df1.to_csv('files/feature_definition.csv')

            df2 = pd.DataFrame({'classes': pd.Series(classes)})
            df2.loc[:, 'run_id'] = st
            df2 = df2.set_index('run_id')
            df2.to_csv('files/classes.csv')

            df3 = pd.DataFrame({'sampling': pd.Series(strategy)})
            df3.loc[:, 'run_id'] = st
            df3 = df3.set_index('run_id')
            df3.to_csv('files/sampling_strategy.csv')

            df4 = pd.DataFrame({'num_imp_features': pd.Series(imp_features)})
            df4.loc[:, 'run_id'] = st
            df4 = df4.set_index('run_id')
            df4.to_csv('files/imp_features.csv')

            df5 = pd.DataFrame({'cross_validation': pd.Series(cv)})
            df5.loc[:, 'run_id'] = st
            df5 = df5.set_index('run_id')
            df5.to_csv('files/cross_validation.csv')

            dfs_shacl_results = []

            for constraint, identifier in zip(constraints, constraint_identifiers):
                df6 = pd.DataFrame(annotated_dataset.loc[:, [constraint.name]]).rename(
                    columns={constraint.name: 'SHACL result'})
                df6['run_id'] = st
                df6['SHACL schema'] = constraint.shape_schema_dir
                df6['SHACL shape'] = constraint.target_shape
                df6['SHACL constraint name'] = constraint.name
                df6['constraint identifier'] = identifier

                df6 = df6.reset_index()
                df6 = df6.rename(columns={df6.columns[0]: 'index'})
                dfs_shacl_results.append(df6)
                pd.concat(dfs_shacl_results, axis='rows').to_csv('files/shacl_validation_results.csv', index=False)

            df7 = pd.DataFrame(annotated_dataset.loc[:, ['node']])
            df7['run_id'] = st
            df7 = df7.drop_duplicates()
            df7 = df7.reset_index()
            df7 = df7.rename(columns={df7.columns[0]: 'index'})
            df7.to_csv('files/entityAlignment.csv', index=False)

            df8 = pd.DataFrame({'endpoint': pd.Series(endpoint)})
            df8.loc[:, 'run_id'] = st
            df8 = df8.set_index('run_id')
            df8.to_csv('files/endpoint.csv')

    annotated_dataset = annotated_dataset.drop(columns=['node'])

    return seed_var, independent_var, dependent_var, classes, class_names, strategy, imp_features, cv, annotated_dataset, constraints, base_dataset, st, \
           input_data['3_valued_logic']


# get the start time and use it as run_id
def current_milli_time():
    return round(time.time() * 1000)


def main():
    st = current_milli_time()
    print(st)

    parser = ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    path_config = args.path

    stats.STATS_COLLECTOR.activate(hyperparameters=[])
    stats.STATS_COLLECTOR.new_run(hyperparameters=[])

    seed_var, independent_var, dependent_var, classes, class_names, strategy, imp_features, cv, annotated_dataset, constraints, base_dataset, st, non_applicable_counts = read_process(
        path_config, st)

    with stats.measure_time('PIPE_PREPROCESSING'):
        encoded_data, encode_target = preprocessing_data.load_data(seed_var, independent_var, dependent_var, classes,
                                                                   annotated_dataset)

    with stats.measure_time('PIPE_SAMPLING'):
        sampled_data, sampled_target = sampling_strategy.sampling_strategy(encoded_data, encode_target, strategy)

    new_sampled_data, clf = classification.classify(sampled_data, sampled_target, imp_features, cv, classes, st)
    processed_df = pd.concat((new_sampled_data, sampled_target), axis='columns')
    processed_df.reset_index(inplace=True)

    with stats.measure_time('PIPE_CONSTRAINT_VIZ'):
        processedDataset = ProcessedDataset.from_node_unique_columns(processed_df, base_dataset,
                                                                     base_columns=[seed_var],
                                                                     target_name='class', categorical_mapping={
                'class': {i: classes[i] for i in range(len(classes))}})
        checker = Checker(clf.predict, processedDataset)

        shadow_tree = get_shadow_tree_from_checker(clf, checker)

        for i, constraint in enumerate(constraints):
            plot = constraint_viz.dtreeviz(shadow_tree, checker, [constraint], coverage=False,
                                           non_applicable_counts=non_applicable_counts)
            plot.save(f'output/plots/constraint{i}_validation_dtree.svg')
            plot = confusion_matrix_decomposition(shadow_tree, checker, constraint,
                                                  non_applicable_counts=non_applicable_counts)
            plot.save(f'output/plots/constraint{i}_validation_matrix.svg')

        plot = constraint_viz.dtreeviz(shadow_tree, checker, constraints, coverage=True,
                                       non_applicable_counts=non_applicable_counts)
        plot.save('output/plots/constraints_validation_dtree.svg')

    stats.STATS_COLLECTOR.to_file('files/times.csv',
                                  categories=['PIPE_DATASET_EXTRACTION', 'PIPE_SHACL_VALIDATION', 'PIPE_PREPROCESSING',
                                              'PIPE_SAMPLING', 'PIPE_IMPORTANT_FEATURES', 'PIPE_LIME',
                                              'PIPE_TRAIN_MODEL',
                                              'PIPE_DTREEVIZ', 'PIPE_CONSTRAINT_VIZ', 'PIPE_OUTPUT', 'join'])


if __name__ == '__main__':
    main()
