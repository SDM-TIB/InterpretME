import matplotlib.pyplot as plt
import seaborn as sns
import validating_models.visualizations.decision_trees as constraint_viz
from validating_models.visualizations.classification import confusion_matrix_decomposition


def sampling(results, path):
    """Sampling strategy plots.

    Parameters
    ----------
    results : dict
        Dictionary to save results.
    path : str
        Path to save plot results.

    """
    autopct = "%.2f"
    val = results['sampling']
    file = path + f"/sampling_{results['run_id']}.png"
    print("Saving sampling strategy plot to", file)
    val.plot.pie(autopct=autopct)
    plt.title("Sampling Strategy")
    plt.savefig(file)


def feature_importance(results, path):
    """

    Parameters
    ----------
    results : dict
         Dictionary to save results.
    path : str
        Path to save plot results.

    """
    fi_df = results['feature_importance']
    file = path + f"/Feature Importance_{results['run_id']}.png"
    print("Saving feature importance plot to", file)
    # Define size of bar plot
    plt.figure(figsize=(20, 15))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title('FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.savefig(file)


def decision_trees(results, path):
    """

    Parameters
    ----------
    results : dict
        Dictionary to save results.
    path : str
        Path to save plot results.

    """
    file = path + f"/Decision_trees_{results['run_id']}.svg"
    print("Saving decision trees to", file)
    vis = results['dtree']
    vis.save(file)


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

    """
    print("Saving constraints decision trees to", path)
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
