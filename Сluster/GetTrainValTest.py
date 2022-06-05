import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score


fnameDataBeta = '/common/home/nerobova_a/DataMethy/MatrixBetaNoNanNoBadCpGandPerson.pkl'
fnameDataLabels = '/common/home/nerobova_a/DataMethy/markerList.txt'
fnamesavegraph = '/common/home/nerobova_a/DataMethy/output'


def LoadingDataBeta(fnameDataBeta):
    with open(fnameDataBeta, 'rb') as handle:
        dfDataBeta = pd.DataFrame(pickle.load(handle))

    print("Data Beta:")
    print("Shape: ", dfDataBeta.shape)
    print(dfDataBeta.head(), '\n')

    return dfDataBeta.values


def returnSizeDataX(fnameDataBeta):
    X = LoadingDataBeta(fnameDataBeta)
    return X.shape[1]


def LoadingDataLabels(fnameDataLabels):
    dfLabels = pd.read_csv(fnameDataLabels, header=None)
    dfLabels.rename(columns={0: 'labels'}, inplace=True)

    print("Data Labels:\n")
    print("Shape: ", dfLabels.shape, '\n')
    print(dfLabels.head(), '\n')

    return dfLabels.values


def CreateTrainValTest(fnameDataBeta, fnameDataLabels):
    X = LoadingDataBeta(fnameDataBeta)
    y = LoadingDataLabels(fnameDataLabels)

    X_train, X_trainval, y_train, y_trainval = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_trainval, y_trainval, test_size=0.5, stratify=y_trainval,
                                                    random_state=42)

    y_train, y_val, y_test = np.asarray(y_train).reshape(-1), np.asarray(y_val).reshape(-1), np.asarray(y_test).reshape(
        -1)

    return X_train, y_train, X_test, y_test, X_val, y_val


def main():
    X_train, y_train, X_test, y_test, X_val, y_val = CreateTrainValTest(fnameDataBeta, fnameDataLabels)
    pickle.dump(X_train, open(fnamesavegraph + "/X_train.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(y_train, open(fnamesavegraph + "/y_train.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(X_test, open(fnamesavegraph + "/X_test.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(y_test, open(fnamesavegraph + "/y_test.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(X_val, open(fnamesavegraph + "/X_val.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(y_val, open(fnamesavegraph + "/y_val.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()