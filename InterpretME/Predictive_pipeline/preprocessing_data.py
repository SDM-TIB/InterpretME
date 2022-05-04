import numpy as np
import pandas as pd
from pathlib import Path
import sys
pd.options.mode.chained_assignment = None  # default='warn'
PACKAGE_VALIDATING_MODELS = str(Path(__file__).parent.parent.joinpath('validating_models').resolve())
sys.path.append(PACKAGE_VALIDATING_MODELS)
import validating_models.stats as stats
sys.path.remove(PACKAGE_VALIDATING_MODELS)

time_preprocessing = stats.get_decorator('PIPE_PREPROCESSING')

def define_class(classes,dependent_var, annotated_dataset):
    #print(annotated_dataset)
    cls = {}
    target = annotated_dataset[dependent_var]
    #print(target)
    length = len(classes)
    if length >=2:
        if length == 2:
              class0, class1 = map(str,(classes))
              id_0 = target.index[target.iloc[:,0].str.contains(class0)]
              target.loc[(target.index.isin(id_0)), 'class'] = 0
              target.loc[~(target.index.isin(id_0)), 'class'] = 1
              target = target[['class']]
              #print(target)
        else:
            for i,j in enumerate(classes):
                cls[j]=i
            target['class']= target.iloc[:,0].map(cls)

            if (target['class'].isnull().values.any()):
                n = len(classes)
                target['class'] = target['class'].replace(np.nan,n)
            target = target[['class']]
            # print(target)
    else:
        print("Error - less than 2 classes given for classification")

    return target


def transform_to_binary(data, attribute, val_a, val_b):
    data.loc[data[attribute]==val_a, attribute]= 0
    data.loc[data[attribute]==val_b, attribute]= 1
    return data.rename(columns={attribute: attribute+'_'+val_b})



def hot_encode(data, seed_var):
    # del independent_var[0]
    col_list = []
    count = data.T.apply(lambda x: x.nunique(dropna=False), axis=1)
    #print(count)
    for col_name,v in count.items():
        if v == 2:
            col_val = data[col_name].values.ravel()
            unique = pd.unique(col_val)
            val_a, val_b = ["".join(item) for item in unique.astype(str)]
            data = transform_to_binary(data,col_name,val_a,val_b)
        else:
            if col_name != seed_var:
                col_list.append(col_name)
    new_data = pd.get_dummies(data=data, columns=col_list)
    return new_data


@time_preprocessing
def load_data(seed_var,independent_var, dependent_var,classes,annotated_dataset):
    print("--------- Preprocessing Data --------------")
    encode_target = define_class(classes, dependent_var, annotated_dataset)
    ann_data = annotated_dataset.drop(dependent_var,1)
    encode_data = hot_encode(ann_data, seed_var)
    #print(encode_target)
    #encode_data.to_csv('encoded_data.csv')
    return encode_data,encode_target






