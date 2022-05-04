import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, make_scorer
import lime
import lime.lime_tabular
from sklearn import tree
import seaborn as sns
import dtreeviz_lib
from tqdm import tqdm
import os,sys
from pathlib import Path
from slugify import slugify

PACKAGE_VALIDATING_MODELS = str(Path(__file__).parent.parent.joinpath('validating_models').resolve())
sys.path.append(PACKAGE_VALIDATING_MODELS)
import validating_models.stats as stats
sys.path.remove(PACKAGE_VALIDATING_MODELS)
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


time_lime = stats.get_decorator('PIPE_LIME')
time_output = stats.get_decorator('PIPE_OUTPUT')

def classify(sampled_data,sampled_target,imp_features, cv, classes,st):

    if (len(classes) == 2):
        new_sampled_data, clf = binary_classification(sampled_data,sampled_target,imp_features, cv, classes,st)
    else:
        new_sampled_data, clf = multiclass(sampled_data,sampled_target,imp_features, cv, classes,st)

    return new_sampled_data, clf

@time_output
def plot_feature_importance(importance, names, model_type,st):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(20, 15))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + '_FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    if not os.path.exists('output/plots'):
        os.makedirs('output/plots')
        print("The directory for output/plots is created")
    print("****************** Plot for important features is saved in output folder ******************")
    plt.savefig('output/plots/Random_Forest_Feature_importance_'+str(st)+'.png')

@time_lime
def lime_interpretation(X_train,new_sampled_data,best_clf,ind_test,X_test,classes,st):
    # lime interpretability

    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),
                                                       feature_names=new_sampled_data.columns.values,
                                                       class_names=classes, discretize_continuous=True,
                                                       random_state=123)

    if not os.path.exists('output/Lime_results'):
        os.makedirs('output/Lime_results')
        print("The directory for output/Lime_results is created")

    lst = []
    lst_prob = []
    print("***************** Saving LIME results ***********************")
    with tqdm(total=min(len(ind_test), len(X_test))) as pbar:
        for i, j in zip(ind_test, X_test):
            explainer.explain_instance(j, best_clf.predict_proba, num_features=10).save_to_file('output/Lime_results/Lime_' + slugify(str(i)) + '.html')
            exp = explainer.explain_instance(j, best_clf.predict_proba, num_features=10)
            df = pd.DataFrame(exp.as_list())
            df['index'] = i
            lst.append(df)
            df2 = pd.DataFrame(exp.predict_proba.tolist())
            df2['index'] = i
            lst_prob.append(df2)
            pbar.update(1)

    df1 = pd.concat(lst)
    df1.loc[:,'run_id'] = st
    df1 = df1.set_index('index')
    df1.rename(columns={df1.columns[0]: "features", df1.columns[1]: "weights"}, inplace=True)
    df1['tool'] = 'LIME'
    df1.to_csv("files/lime_interpretation_features.csv")

    df2 = pd.concat(lst_prob)
    df2 = df2.reset_index()
    df2.rename(columns={df2.columns[0]: "class", df2.columns[1]: "PredictionProbablities"}, inplace=True)
    df2['tool'] = 'LIME'
    df2.loc[:, 'run_id'] = st
    df2 = df2.set_index('index')
    df2.to_csv("files/predicition_probablities.csv")

    print("***************************** Lime Interpretability results saved in output folder ****************************")
    return df1



def binary_classification(sampled_data, sampled_target, imp_features, cross_validation, classes,st):
    sampled_target['class'] = sampled_target['class'].astype(int)
    X = sampled_data
    y = sampled_target['class']

    X_imput, y_imput = X.values, y.values
    with stats.measure_time('PIPE_IMPORTANT_FEATURES'):
        print("---------------- Random Forest Classification with Stratified shuffle split -----------------------")
        rf_estimator = RandomForestClassifier(max_depth=4, random_state=0)
        cv = StratifiedShuffleSplit(n_splits=cross_validation, test_size=0.3, random_state=123)
        important_features = set()
        important_features_size = imp_features
        print("************** Classification report for every iteration ************************************")
        for i, (train, test) in enumerate(cv.split(X_imput, y_imput)):
            rf_estimator.fit(X_imput[train], y_imput[train])
            y_predicted = rf_estimator.predict(X_imput[test])

            print(classification_report(y_imput[test], y_predicted))

            fea_importance = rf_estimator.feature_importances_
            indices = np.argsort(fea_importance)[::-1]
            for f in range(important_features_size):
                important_features.add(X.columns.values[indices[f]])

    plot_feature_importance(rf_estimator.feature_importances_, X.columns, 'RANDOM FOREST',st)

    # Taking important features
    new_sampled_data = sampled_data[list(important_features)]
    indices = new_sampled_data.index.values
    #print(indices)
    X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(new_sampled_data.values, sampled_target['class'].values,indices,random_state=123)

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
        res.loc[:,'run_id'] = st
        res.loc[:,'model'] = model_name
        res.loc[:,'accuracy'] = acc
        res = res.set_index('run_id')
        res.to_csv('files/model_accuracy_hyperparameters.csv')


    lime_interpretation(X_train,new_sampled_data,best_clf,ind_test,X_test,classes,st)

    with stats.measure_time('PIPE_OUTPUT'):
        print("****************** Classification report saved in output folder *************************")

        report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
        classificationreport = pd.DataFrame(report).transpose()
        classificationreport.loc[:, 'run_id'] = st
        classificationreport = classificationreport.reset_index()
        classificationreport = classificationreport.rename(columns={classificationreport.columns[0]:'classes'})
        print(classificationreport)
        report = classificationreport.iloc[:-3,:]
        # print(report)
        report.to_csv("files/precision_recall.csv", index=False)

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
        viz.save('output/plots/RF_binary_final_results'+str(st)+'.svg')
        print("****** Decision tree plot saved in output/plot folder *********")

    return new_sampled_data, best_clf



def multiclass(sampled_data,sampled_target,imp_features, cv, classes,st):
    classes.append('Others')
    sampled_target['class'] = sampled_target['class'].astype(int)

    X = sampled_data
    y = sampled_target['class']

    X_imput, y_imput = X.values, y.values
    with stats.measure_time('PIPE_IMPORTANT_FEATURES'):
        print("---------------- Random Forest Classification with Stratified shuffle split -----------------------")
        rf_estimator = RandomForestClassifier(max_depth=4, random_state=0)
        cv = StratifiedShuffleSplit(n_splits=cv, test_size=0.3, random_state=123)
        important_features = set()
        important_features_size = imp_features
        print("************** Classification report for every iteration ************************************")
        for i, (train, test) in enumerate(cv.split(X_imput, y_imput)):
            rf_estimator.fit(X_imput[train], y_imput[train])
            y_predicted = rf_estimator.predict(X_imput[test])

            print(classification_report(y_imput[test], y_predicted))

            fea_importance = rf_estimator.feature_importances_
            indices = np.argsort(fea_importance)[::-1]
            for f in range(important_features_size):
                important_features.add(X.columns.values[indices[f]])

    plot_feature_importance(rf_estimator.feature_importances_, X.columns, 'RANDOM FOREST',st)

    # Taking important features
    new_sampled_data = sampled_data[list(important_features)]
    indices = new_sampled_data.index.values
    # print(indices)
    X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(new_sampled_data.values,
                                                                             sampled_target['class'].values,
                                                                             indices, random_state=123)

    feature_names = new_sampled_data.columns
    parameters = {"max_depth": range(4, 6)}

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
        if not os.path.isfile('files/model_accuracy_hyperparameters.csv'):
            res.to_csv('files/model_accuracy_hyperparameters.csv')
        else:
            res.to_csv('files/model_accuracy_hyperparameters.csv', mode='a', header=False)

    lime_interpretation(X_train,new_sampled_data,best_clf,ind_test,X_test,classes,st)

    with stats.measure_time('PIPE_OUTPUT'):
        print("****************** Classification report saved in output folder *************************")
        report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
        classificationreport = pd.DataFrame(report).transpose()
        classificationreport.loc[:, 'run_id'] = st
        report = classificationreport.iloc[:-3, :]
        if not os.path.isfile('files/precision_recall.csv'):
            report.to_csv("files/precision_recall.csv", index=False)
        else:
            report.to_csv("files/precision_recall.csv", index=False, mode='a', header=False)

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
        viz.save('output/plots/RF_multiclass_classification_final_results'+str(st)+'.svg')
        print("****** Decision tree plot saved in output/plot folder *********")

    return new_sampled_data, best_clf


