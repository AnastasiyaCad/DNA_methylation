import pickle5 as pickle
import numpy as np
import pandas as pd

import globalConstants


# Загрузка матрицы бета-значений
def LoadingDataBeta():
    with open(globalConstants.fnameDataBeta, 'rb') as handle:
        dfDataBeta = pd.DataFrame(pickle.load(handle))

    print("Data Beta loaded")
    print("Shape: ", dfDataBeta.shape)
    print(dfDataBeta.head(), '\n')

    return dfDataBeta.values


# загрузка данных меток класса
def LoadingDataLabels():
    dfLabels = pd.read_csv(globalConstants.fnameDataLabels, header=None)
    dfLabels.rename(columns={0: 'labels'}, inplace=True)

    print("Data Labels:\n")
    print("Shape: ", dfLabels.shape, '\n')
    print(dfLabels.head(), '\n')

    return dfLabels.values


# Загрузка матрицы бета-значений для теста
def LoadingDataHome():
    with open(globalConstants.fnameDataBeta, 'rb') as fileData:
        fileDataNew = pickle.load(fileData)
    dfDataBeta = pd.DataFrame(fileDataNew)
    dfDataBeta = dfDataBeta.iloc[:, :10]
    dfPersonInfo = LoadingNamePersonHome()
    PersonName = dfPersonInfo["id_person"]
    DataBetaNamePerson = np.array(PersonName[:1393])
    # categorical_features_indices = np.where(dfDataBeta.dtypes != float)[0]
    return dfDataBeta.values, DataBetaNamePerson #, list(dfDataBeta.columns.values), categorical_features_indices


def LoadingNamePersonHome():
    with open(globalConstants.fnameDataPersonInfo, 'rb') as fileData:
        fileDataPersonInfo = pickle.load(fileData)
    dfPersonInfo = pd.DataFrame(fileDataPersonInfo)
    return dfPersonInfo


def LoadingDataLabelsHome():
    with open(globalConstants.fnameDataLabels, 'rb') as f:
        dfLabels = pickle.load(f)
    return dfLabels


# возвращает размер, но хз, стоит ли
def returnSizeDataX():
    X = LoadingDataBeta(globalConstants.fnameDataBeta)
    return X.shape[1]


def returnSizeDataY():
    dfLabels = pd.read_csv(globalConstants.fnameDataLabels, header=None)
    dfLabels.rename(columns={0: 'labels'}, inplace=True)
    return dfLabels.shape

