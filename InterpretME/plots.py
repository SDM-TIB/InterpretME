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

    Returns
    -------
    str
        Path of the saved plot.

    """
    autopct = "%.2f"
    val = results['sampling']
    file = path + f"/sampling_{results['run_id']}.png"
    print("Saving sampling strategy plot to", file)
    val.plot.pie(autopct=autopct)
    plt.title("Sampling Strategy")
    plt.savefig(file)
    return file


def feature_importance(results, path):
    """

    Parameters
    ----------
    results : dict
         Dictionary to save results.
    path : str
        Path to save plot results.

    Returns
    -------
    str
        Path of the saved plot.

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
    return file


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
    str
        Path of the saved plot.

    """
    file = path + f"/Decision_trees_{results['run_id']}.svg"
    print("Saving decision trees to", file)
    vis = results['dtree']
    vis.save(file)
    return file


def constraints_decision_trees(results, path, constraint_num=None):
    """

    Parameters
    ----------
    results : dict
        Dictionary to save results.
    path : str
        Path to save plot results.
    constraint_num : list, OPTIONAL
        Number of constraints for saving plots.

    Returns
    -------
    List[str]
        List containing the paths of the saved plots.

    """
    print("Saving constraints decision trees to", path)
    run = results['run_id']
    checker = results['checker']
    shadow_tree = results['shadow_tree']
    constraints = results['constraints']
    non_applicable_counts = results['non_applicable_counts']
    files = []
    if constraint_num is None:
        constraint_num = []

    # Overall
    plot = constraint_viz.dtreeviz(
        shadow_tree, checker, constraints, coverage=True, non_applicable_counts=non_applicable_counts
    )
    plot.save(path + f'/constraints_validation_dtree_{run}.svg')
    files.append(path + f'/constraints_validation_dtree_{run}.svg')

    for i in constraint_num:
        constraint = constraints[i-1]
        c_name = constraint.name
        plot = constraint_viz.dtreeviz(
            shadow_tree, checker, [constraint], coverage=False, non_applicable_counts=non_applicable_counts,
            title='Decision Tree with Validation Results for ' + c_name
        )
        plot.save(path + f'/constraint_{c_name}_validation_dtree_{run}.svg')
        files.append(path + f'/constraint_{c_name}_validation_dtree_{run}.svg')
        plot = confusion_matrix_decomposition(
            shadow_tree, checker, constraint, non_applicable_counts=non_applicable_counts
        )
        plot.save(path + f'/constraint_{c_name}_validation_matrix_{run}.svg')
        files.append(path + f'/constraint_{c_name}_validation_matrix_{run}.svg')

    return files
