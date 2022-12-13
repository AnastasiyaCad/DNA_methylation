import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

import loadCreateData.lodingData as loadData
import globalConstants


# возвращвет созданные тренировочную, валидационную и тестовую выборку
# с соотношением 70/15/15
def CreateTrainValTest():
    X = loadData.LoadingDataBeta()
    y = loadData.LoadingDataLabels()

    X_train, X_trainval, y_train, y_trainval = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_trainval, y_trainval, test_size=0.5, stratify=y_trainval,
                                                    random_state=42)

    y_train, y_val, y_test = np.asarray(y_train).reshape(-1), np.asarray(y_val).reshape(-1), np.asarray(y_test).reshape(
        -1)

    return X_train, y_train, X_test, y_test, X_val, y_val


def getDataTrainValFromCrossVal():
    X = loadData.LoadingDataBeta()
    y = loadData.LoadingDataLabels()
    rkf = RepeatedKFold(n_splits=4, n_repeats=1, random_state=2652124)
    X_train, y_train, X_val, y_val = [], [], [], []
    for train_index, val_index in rkf.split(X):
        X_train.append(X[train_index])
        X_val.append(X[val_index])
        y_train.append(y[train_index])
        y_val.append(y[val_index])
    return X_train, y_train, X_val, y_val


def getDataTrainValFromCrossValTest():
    X, DataBetaNamePerson = loadData.LoadingDataHome()
    y = loadData.LoadingDataLabelsHome()
    rkf = RepeatedKFold(n_splits=4, n_repeats=1, random_state=2652124)
    XTrainSet, yTrainSet, XValSet, yValSet, DataBetaNamePersonTrainSet, DataBetaNamePersonTestSet = [], [], [], [], [], []
    for train_index, val_index in rkf.split(X):
        # XTrainSet, XValSet = X[train_index], X[val_index]
        # yTrainSet, yValSet = y[train_index], y[val_index]
        XTrainSet.append(X[train_index])
        XValSet.append(X[val_index])
        yTrainSet.append(y[train_index])
        yValSet.append(y[val_index])
        DataBetaNamePersonTrainSet.append(DataBetaNamePerson[train_index])
        DataBetaNamePersonTestSet.append(DataBetaNamePerson[val_index])
    return XTrainSet, yTrainSet, XValSet, yValSet, DataBetaNamePersonTrainSet, DataBetaNamePersonTestSet

