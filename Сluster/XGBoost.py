import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score


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


def CreateTrainTest(fnameDataBeta, fnameDataLabels):
    X = LoadingDataBeta(fnameDataBeta)
    y = LoadingDataLabels(fnameDataLabels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

    y_train, y_test = np.asarray(y_train).reshape(-1), np.asarray(y_test).reshape(-1)

    return X_train, y_train, X_test, y_test


def main():
    X_train, y_train, X_test, y_test = CreateTrainTest(fnameDataBeta, fnameDataLabels)
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test)

    dmat_train = xgb.DMatrix(X_train, y_train)
    dmat_test = xgb.DMatrix(X_test, y_test)
    
    num_boost_round = 16
    #param train model
    params = {
        # максимальная глубина дерева по умолч 6
        'max_depth': 3,
        'objective': 'multi:softmax',  # error evaluation for multiclass training
        'num_class': 25,
        'n_gpus': 0
    }
    # train model's
    bst = xgb.train(params, num_boost_round=num_boost_round, dtrain=dtrain, evals=[(dtrain, "train"), (dmat_test, "test")])
    print('booster', bst)

    y_pred = bst.predict(dtest)

    print('pred.sum, y_test.sum, len_y_test = ')
    print(y_pred.sum(), y_test.sum(), len(y_test))

    with plt.style.context("ggplot"):
        fig, ax = plt.subplots()
        xgb.plotting.plot_importance(bst, ax=ax, height=0.6, importance_type="weight")
    fig.savefig(fnamesavegraph + '/graph_' + 'plot_importance.png')
    plt.close()

    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred))

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix_df, annot=True)
    fig.savefig(fnamesavegraph + '/graph_' + 'matrix.png')
    plt.close()

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()