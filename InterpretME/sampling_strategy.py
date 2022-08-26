import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


def sampling_strategy(encode_data, encode_target, strategy, results):
    """Sampling strategy.

    Parameters
    ----------
    encode_data : dataframe
        Preprocessed unbalanced dataframe needed to apply sampling.
    encode_target : dataframe
        Preprocessed unbalanced target dataframe needed to apply sampling.
    strategy : str
        Sampling strategy specified by user (undersampling or oversampling).
    results : dict
        To store results of sampling plots.

    Returns
    -------
    (dataframe, dataframe, dict)
        Sampled dataa and results dictionary.

    """
    global X, y  # TODO: is there any reason why they are marked as global but are not defined at module level?
    if strategy == 'undersampling':
        X, y, results = undersampling(encode_data, encode_target, results)
    if strategy == 'oversampling':
        X, y, results = oversampling(encode_data, encode_target, results)
    return X, y, results


def undersampling(encode_data, encode_target, results):
    """Under-Sampling strategy.

    Parameters
    ----------
    encode_data : dataframe
        Preprocessed unbalanced dataframe needed to apply sampling.
    encode_target : dataframe
        Preprocessed unbalanced target dataframe needed to apply sampling.
    results : dict
        To store results of sampling plots.

    Returns
    -------
    (dataframe, dataframe, dict)
        Sampled data and results dictionary.

    """
    sampling_strategy = "not minority"
    X = encode_data
    Y = encode_target['class']
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=123)
    X_res, y_res = rus.fit_resample(X, Y)
    X_res.index = X.index[rus.sample_indices_]
    y_res.index = Y.index[rus.sample_indices_]
    samp = y_res.value_counts()
    results['sampling'] = samp
    y_res = pd.DataFrame(data=y_res)
    return X_res, y_res, results


def oversampling(encode_data, encode_target, results):
    """Over-Sampling strategy.

    Parameters
    ----------
    encode_data : dataframe
        Preprocessed unbalanced dataframe needed to apply sampling.
    encode_target : dataframe
        Preprocessed unbalanced target dataframe needed to apply sampling.
    results : dict
        To store results of sampling plots.

    Returns
    -------
    (dataframe, dataframe, dict)
        Sampled dataa and results dictionary.

    """
    sampling_strategy = "not majority"
    autopct = "%.2f"  # TODO: unused variable; should it be deleted?
    X = encode_data
    Y = encode_target['class']
    ros = RandomOverSampler(sampling_strategy=sampling_strategy)
    X_res, y_res = ros.fit_resample(X, Y)
    X_res.index = X.index[ros.sample_indices_]
    y_res.index = Y.index[ros.sample_indices_]
    samp = y_res.value_counts()
    results['sampling'] = samp
    y_res = pd.DataFrame(data=y_res)
    return X_res, y_res, results
