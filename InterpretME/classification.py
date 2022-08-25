import os

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import validating_models.stats as stats
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from slugify import slugify

from . import dtreeviz_lib


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

time_lime = stats.get_decorator('PIPE_LIME')
time_output = stats.get_decorator('PIPE_OUTPUT')


def classify(sampled_data, sampled_target, imp_features, cv, classes, st, lime_results, train_test_split, model,
             results):
    """Selecting classification strategy based on the number of classes provided by the user.

    Parameters
    ----------
    sampled_data : dataframe
        Preprocessed and sampled data.
    sampled_target : dataframe
        Preprocessed and sampled target.
    imp_features : int
        Number of important features.
    cv : int
        Number of cross validation splits required while performing stratified shuffle split.
    classes : list
        List of classes used for classification.
    st : int
        Unique identifier.
    lime_results : str
        Path to save LIME results in HTML format.
    train_test_split : int
        Splits of dataset into training set and testing set.
    model : str
        Model used to perform stratified shuffle split specified by user.
    results : dict
        Dictionary to save plot results.

    Returns
    -------
    (dataframe, model, dict)

    """
    if (len(classes) == 2):
        new_sampled_data, clf, results = binary_classification(sampled_data, sampled_target, imp_features, cv, classes,
                                                               st, lime_results, train_test_split, model, results)
    else:
        new_sampled_data, clf, results = multiclass(sampled_data, sampled_target, imp_features, cv, classes, st,
                                                    lime_results, train_test_split, model, results)

    return new_sampled_data, clf, results


@time_output
def plot_feature_importance(importance, names):
    """Plots list of important features

    Parameters
    ----------
    importance : list
        Value of important features.
    names : list
        Label of important features.

    Returns
    -------
    dataframe

    """
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    return fi_df


@time_lime
def lime_interpretation(X_train, new_sampled_data, best_clf, ind_test, X_test, classes, st, lime_results):
    """Generates LIME interpretation results.

    Parameters
    ----------
    X_train : array
        Training dataset used to generate LIME interpretation.
    new_sampled_data : dataframe
        Preporcessed dataset.
    best_clf : model
        Best model saved after applying Decision tree.
    ind_test : index
        Testing index.
    X_test : array
        Testing dataset used to generate LIME interpretation.
    classes : list
        List of classes for classification.
    st : int
        Unique identifier.
    lime_results : str
        Path to save LIME interpretation results.

    Returns
    -------
    dataframe

    """
    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),
                                                       feature_names=new_sampled_data.columns.values,
                                                       class_names=classes, discretize_continuous=True,
                                                       random_state=123)

    lst = []
    lst_prob = []

    if lime_results is None:
        print("##############################################################################")
        print("********* LIME results will not be saved as the path is not specified*********")
        print("##############################################################################")
    else:
        if not os.path.exists(lime_results):
            os.makedirs(lime_results)
        print("#############################################################")
        print("***************** Saving LIME results ***********************")
        print("#############################################################")
        with tqdm(total=min(len(ind_test), len(X_test))) as pbar:
            for i, j in zip(ind_test, X_test):
                explainer.explain_instance(j, best_clf.predict_proba, num_features=10).save_to_file(
                    lime_results + '/Lime_' + slugify(str(i)) + '.html')
                pbar.update(1)

    # with tqdm(total=min(len(ind_test), len(X_test))) as pbar:
    for i, j in zip(ind_test, X_test):
        exp = explainer.explain_instance(j, best_clf.predict_proba, num_features=10)
        df = pd.DataFrame(exp.as_list())
        df['index'] = i
        lst.append(df)
        df2 = pd.DataFrame(exp.predict_proba.tolist())
        df2['index'] = i
        lst_prob.append(df2)
        # pbar.update(1)

    df1 = pd.concat(lst)
    df1.loc[:, 'run_id'] = st
    df1 = df1.set_index('index')
    df1.rename(columns={df1.columns[0]: "features", df1.columns[1]: "weights"}, inplace=True)
    df1['tool'] = 'LIME'
    df1.to_csv("interpretme/files/lime_interpretation_features.csv")

    df2 = pd.concat(lst_prob)
    df2 = df2.reset_index()
    df2.rename(columns={df2.columns[0]: "class", df2.columns[1]: "PredictionProbabilities"}, inplace=True)
    df2['tool'] = 'LIME'
    df2.loc[:, 'run_id'] = st
    df2 = df2.set_index('index')
    df2.to_csv("interpretme/files/predicition_probabilities.csv")

    # print("***************************** Lime Interpretability results saved in output folder ****************************")
    return df1


def binary_classification(sampled_data, sampled_target, imp_features, cross_validation, classes, st, lime_results,
                          test_split, model, results):
    """Binary classification technique.

    Parameters
    ----------
    sampled_data : dataframe
        Dataframe given as input to the classifier.
    sampled_target : dataframe
        Target dataframe given as input to classifier.
    imp_features : int
        Number of important features desired by user.
    cross_validation : int
        Number of cross validation splits required by user.
    classes : list
        List of classes used for classification.
    st : int
        Unique indentifier.
    lime_results : str
        Path to store HTML format LIME results.
    test_split : float
        Percentile splits of training and testing dataset.
    model : model
        Saved best model after applying GridSearch.
    results : dict
        Dictionary to store plots results.

    Returns
    -------
    (dataframe, model, dict)

    """
    sampled_target['class'] = sampled_target['class'].astype(int)
    X = sampled_data
    y = sampled_target['class']

    X_imput, y_imput = X.values, y.values
    with stats.measure_time('PIPE_IMPORTANT_FEATURES'):
        # print("---------------- Random Forest Classification with Stratified shuffle split -----------------------")
        # print(model)
        if (model == 'Random Forest'):
            print('Random Forest Classifier')
            estimator = RandomForestClassifier(max_depth=4, random_state=0)
        elif (model == 'AdaBoost'):
            print('AdaBoost Classifier')
            estimator = AdaBoostClassifier(random_state=0)
        elif (model == 'Gradient Boosting'):
            print('Gradient Boosting Classifier')
            estimator = GradientBoostingClassifier(random_state=0)

        cv = StratifiedShuffleSplit(n_splits=cross_validation, test_size=test_split, random_state=123)
        important_features = set()
        important_features_size = imp_features
        print("###########################################################################")
        print("************** Classification report for every iteration ******************")
        print("###########################################################################")
        for i, (train, test) in enumerate(cv.split(X_imput, y_imput)):
            estimator.fit(X_imput[train], y_imput[train])
            y_predicted = estimator.predict(X_imput[test])

            print(classification_report(y_imput[test], y_predicted))

            fea_importance = estimator.feature_importances_
            indices = np.argsort(fea_importance)[::-1]
            for f in range(important_features_size):
                important_features.add(X.columns.values[indices[f]])

    data = plot_feature_importance(estimator.feature_importances_, X.columns, model, st)
    results['feature_importance'] = data

    # Taking important features
    new_sampled_data = sampled_data[list(important_features)]
    indices = new_sampled_data.index.values
    # print(indices)
    X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(new_sampled_data.values,
                                                                             sampled_target['class'].values, indices,
                                                                             random_state=123)

    feature_names = new_sampled_data.columns
    parameters = {"max_depth": range(4, 6)}
    with stats.measure_time('PIPE_TRAIN_MODEL'):
        # Defining Decision tree Classifier
        clf = tree.DecisionTreeClassifier()

        # GrdiSearchCV to select best hyperparameters
        grid = GridSearchCV(estimator=clf, param_grid=parameters)
        grid_res = grid.fit(X_train, y_train)
        best_clf = grid_res.best_estimator_

    # predictions = (clf.fit(X_train, y_train)).predict(X_test)
    with stats.measure_time('PIPE_OUTPUT'):
        acc = best_clf.score(X_test, y_test)
        y_pred = best_clf.predict(X_test)
        model_name = type(best_clf).__name__

        hyp = best_clf.get_params()
        hyp_keys = hyp.keys()
        hyp_val = hyp.values()

        res = pd.DataFrame({'hyperparameters_name': pd.Series(hyp_keys), 'hyperparameters_value': pd.Series(hyp_val)})
        res.loc[:, 'run_id'] = st
        res.loc[:, 'model'] = model_name
        res.loc[:, 'accuracy'] = acc
        res = res.set_index('run_id')
        res.to_csv('interpretme/files/model_accuracy_hyperparameters.csv')

    lime_interpretation(X_train, new_sampled_data, best_clf, ind_test, X_test, classes, st, lime_results)

    with stats.measure_time('PIPE_OUTPUT'):
        print("#########################################################################################")
        print("****************** Classification report saved in output folder *************************")
        print("#########################################################################################")

        report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
        classificationreport = pd.DataFrame(report).transpose()
        classificationreport.loc[:, 'run_id'] = st
        classificationreport = classificationreport.reset_index()
        classificationreport = classificationreport.rename(columns={classificationreport.columns[0]: 'classes'})
        print(classificationreport)
        report = classificationreport.iloc[:-3, :]
        # print(report)
        report.to_csv("interpretme/files/precision_recall.csv", index=False)

    with stats.measure_time('PIPE_DTREEVIZ'):
        bool_feature = []
        for feature in new_sampled_data.columns:
            values = new_sampled_data[feature].unique()
            if len(values) == 2:
                values = sorted(values)
                if values[0] == 0 and values[1] == 1:
                    bool_feature.append(feature)

        viz = dtreeviz_lib.dtreeviz(best_clf, new_sampled_data, sampled_target['class'], target_name='class',
                                    feature_names=feature_names, class_names=classes, fancy=True,
                                    show_root_edge_labels=True, bool_feature=bool_feature)

        results['dtree'] = viz

    return new_sampled_data, best_clf, results


def multiclass(sampled_data, sampled_target, imp_features, cv, classes, st, lime_results, test_split, model,
               results):
    """Multiclass classification technique

    Parameters
    ----------
    sampled_data : dataframe
        Dataframe given as input to the classifier.
    sampled_target : dataframe
        Target dataframe given as input to the classifier.
    imp_features : int
        Number of important features desired by user.
    cv : int
        Number of cross validation splits required by user.
    classes : list
        List of classes used for classification.
    st : int
        Unique identifier.
    lime_results : str
        Path to save LIME results in HTML format.
    test_split : float
        Percentile splits of training and testing dataset.
    model : model
        Saved best model after applying GridSearch.
    results : dict
        Dictionary to store plots results.

    Returns
    -------
    (dataframe, model, dict)

    """
    sampled_target['class'] = sampled_target['class'].astype(int)

    X = sampled_data
    y = sampled_target['class']

    X_imput, y_imput = X.values, y.values
    with stats.measure_time('PIPE_IMPORTANT_FEATURES'):
        # print("---------------- Random Forest Classification with Stratified shuffle split -----------------------")
        if (model == 'Random Forest'):
            print('Random Forest Classifier')
            estimator = RandomForestClassifier(max_depth=4, random_state=0)
        elif (model == 'AdaBoost'):
            print('AdaBoost Classifier')
            estimator = AdaBoostClassifier(random_state=0)
        elif (model == 'Gradient Boosting'):
            print('Gradient Boosting Classifier')
            estimator = GradientBoostingClassifier(random_state=0)

        cv = StratifiedShuffleSplit(n_splits=cv, test_size=train_test_split, random_state=123)
        important_features = set()
        important_features_size = imp_features
        print("#################################################################################")
        print("************** Classification report for every iteration ************************")
        print("#################################################################################")
        for i, (train, test) in enumerate(cv.split(X_imput, y_imput)):
            estimator.fit(X_imput[train], y_imput[train])
            y_predicted = estimator.predict(X_imput[test])

            print(classification_report(y_imput[test], y_predicted))

            fea_importance = estimator.feature_importances_
            indices = np.argsort(fea_importance)[::-1]
            for f in range(important_features_size):
                important_features.add(X.columns.values[indices[f]])

    data = plot_feature_importance(estimator.feature_importances_, X.columns, model, st)
    results['feature_importance'] = data

    # Taking important features
    new_sampled_data = sampled_data[list(important_features)]
    indices = new_sampled_data.index.values
    # print(indices)
    X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(new_sampled_data.values,
                                                                             sampled_target['class'].values,
                                                                             indices, random_state=123)

    feature_names = new_sampled_data.columns
    parameters = {"max_depth": 3}
    # range(1,3)
    # Defining Decision tree Classifier
    with stats.measure_time('PIPE_TRAIN_MODEL'):
        clf = tree.DecisionTreeClassifier()

        # GrdiSearchCV to select best hyperparameters
        grid = GridSearchCV(estimator=clf, param_grid=parameters)
        grid_res = grid.fit(X_train, y_train)
        best_clf = grid_res.best_estimator_

    # predictions = (clf.fit(X_train, y_train)).predict(X_test)
    with stats.measure_time('PIPE_OUTPUT'):
        acc = best_clf.score(X_test, y_test)
        y_pred = best_clf.predict(X_test)

        model_name = type(best_clf).__name__

        hyp = best_clf.get_params()
        hyp_keys = hyp.keys()
        hyp_val = hyp.values()

        res = pd.DataFrame({'hyperparameters_name': pd.Series(hyp_keys), 'hyperparameters_value': pd.Series(hyp_val)})
        res.loc[:, 'run_id'] = st
        res.loc[:, 'model'] = model_name
        res.loc[:, 'accuracy'] = acc
        res = res.set_index('run_id')

        if not os.path.isfile('interpretme/files/model_accuracy_hyperparameters.csv'):
            res.to_csv('interpretme/files/model_accuracy_hyperparameters.csv')
        else:
            res.to_csv('interpretme/files/model_accuracy_hyperparameters.csv', mode='a', header=False)

    lime_interpretation(X_train, new_sampled_data, best_clf, ind_test, X_test, classes, st, lime_results)

    with stats.measure_time('PIPE_OUTPUT'):
        print("#########################################################################################")
        print("****************** Classification report saved in output folder *************************")
        print("#########################################################################################")

        report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
        classificationreport = pd.DataFrame(report).transpose()
        classificationreport.loc[:, 'run_id'] = st
        classificationreport = classificationreport.reset_index()
        classificationreport = classificationreport.rename(columns={classificationreport.columns[0]: 'classes'})
        print(classificationreport)
        report = classificationreport.iloc[:-3, :]
        # print(report)
        report.to_csv("interpretme/files/precision_recall.csv", index=False)

    with stats.measure_time('PIPE_DTREEVIZ'):
        bool_feature = []
        for feature in new_sampled_data.columns:
            values = new_sampled_data[feature].unique()
            if len(values) == 2:
                values = sorted(values)
                if values[0] == 0 and values[1] == 1:
                    bool_feature.append(feature)

        viz = dtreeviz_lib.dtreeviz(best_clf, new_sampled_data, sampled_target['class'], target_name='class',
                                    feature_names=feature_names, class_names=classes, fancy=True,
                                    show_root_edge_labels=True, bool_feature=bool_feature)

        results['dtree'] = viz

    return new_sampled_data, best_clf, results
