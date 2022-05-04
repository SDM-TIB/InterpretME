import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import os

def sampling_strategy(encode_data,encode_target,strategy):
    global X,y
    if (strategy=='undersampling'):
        X, y = undersampling(encode_data,encode_target)
    if (strategy=='oversampling'):
        X, y = oversampling(encode_data,encode_target)
    return X, y


def undersampling(encode_data, encode_target):
    print("-------- Undersampling strategy -----------")
    if not os.path.exists('output/plots'):
        os.makedirs('output/plots')
        print("The directory for output/plots is created")

    sampling_strategy = "not minority"
    autopct = "%.2f"
    X = encode_data
    Y = encode_target['class']
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=123)
    X_res, y_res = rus.fit_resample(X, Y)
    X_res.index = X.index[rus.sample_indices_]
    y_res.index = Y.index[rus.sample_indices_]
    # print(y_res)
    y_res.value_counts().plot.pie(autopct=autopct)
    plt.title("Under-sampling")
    plt.savefig('output/plots/Under-sampling.png')
    print("******** Undersampling Plot saved in output folder ***********")
    y_res = pd.DataFrame(data=y_res)
    # print(y_res)
    y_res.to_csv('dataset/target_under_sampling.csv')
    X_res.to_csv('dataset/data_under_sampling.csv', index=None)
    return X_res, y_res

def oversampling(encode_data, encode_target):
    print("-------- Oversampling strategy -----------")
    if not os.path.exists('output/plots'):
        os.makedirs('output/plots')
        print("The directory for output/plots is created")

    sampling_strategy = "not majority"
    autopct = "%.2f"
    # print(encode_data)
    X = encode_data
    Y = encode_target['class']
    ros = RandomOverSampler(sampling_strategy=sampling_strategy)
    X_res, y_res = ros.fit_resample(X, Y)
    X_res.index = X.index[ros.sample_indices_]
    y_res.index = Y.index[ros.sample_indices_]
    y_res.value_counts().plot.pie(autopct=autopct)
    plt.title("Over-sampling")
    plt.savefig('output/plots/Over-sampling.png')
    print("******** Oversampling Plot saved in output folder ***********")
    y_res = pd.DataFrame(data=y_res)
    # print(y_res)
    y_res.to_csv('dataset/target_Over_sampling.csv')
    X_res.to_csv('dataset/data_Over_sampling.csv', index=None)
    return X_res, y_res

