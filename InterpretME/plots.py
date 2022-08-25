import matplotlib.pyplot as plt
import seaborn as sns
import validating_models.visualizations.decision_trees as constraint_viz
from validating_models.visualizations.classification import confusion_matrix_decomposition


def sampling(results,path):
    """Sampling strategy plots.

    Parameters
    ----------
    results : dict
        Dictionary to save results.
    path : str
        Path to save plot results.

    Returns
    -------

    """

    print("########################################################################")
    print("************************* Sampling strategy ****************************")
    print("########################################################################")
    autopct = "%.2f"
    val = results['sampling']
    run = results['run_id']
    val.plot.pie(autopct=autopct)
    plt.title("Sampling Strategy")
    plt.savefig(path+f'/sampling_{run}.png')


def feature_importance(results,path):
    """

    Parameters
    ----------
    results : dict
         Dictionary to save results.
    path : str
        Path to save plot results.

    Returns
    -------

    """
    print("#####################################################################")
    print("******************* Feature Importance plot *************************")
    print("#####################################################################")
    fi_df = results['feature_importance']
    run = results['run_id']
    # Define size of bar plot
    plt.figure(figsize=(20, 15))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title('FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.savefig(path +f'/Feature Importance_{run}.png')


def decision_trees(results, path):
    """

    Parameters
    ----------
    results : dict
        Dictionary to save results.
    path : str
        Path to save plot results.

    Returns
    -------

    """
    print("#####################################################################")
    print("*********************** Decision Trees ******************************")
    print("#####################################################################")
    vis = results['dtree']
    run = results['run_id']
    vis.save(path+f'/Decision_tree_{run}.svg')

def constraints_decision_trees(results, path, constraint_num):
    """

    Parameters
    ----------
    results : dict
        Dictionary to save results.
    path : str
        Path to save plot results.
    constraint_num : list
        Number of constraints for saving plots.

    Returns
    -------

    """
    print("#########################################################################")
    print("*************************** Constraints Decision Trees ******************")
    print("##########################################################################")
    run = results['run_id']
    checker = results['checker']
    shadow_tree = results['shadow_tree']
    constraints = results['constraints']
    non_applicable_counts = results['non_applicable_counts']
    num = constraint_num

    for i, constraint in enumerate(constraints, start=1):
        for x in num:
            if(x == i):
                plot = constraint_viz.dtreeviz(shadow_tree, checker, [constraint], coverage=False,
                                               non_applicable_counts=non_applicable_counts)
                plot.save(path + f'/constraint_{i}_validation_dtree_{run}.svg')

                plot = confusion_matrix_decomposition(shadow_tree, checker, constraint,
                                                      non_applicable_counts=non_applicable_counts)
                plot.save(path + f'/constraint_{i}_validation_matrix_{run}.svg')
            else:
                plot = constraint_viz.dtreeviz(shadow_tree, checker, constraints, coverage=True,
                                               non_applicable_counts=non_applicable_counts)
                plot.save(path + f'/constraints_validation_dtree_{run}.svg')








