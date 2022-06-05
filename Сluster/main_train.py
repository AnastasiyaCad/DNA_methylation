import xgboost_train
import catboost_train
import lightgbm_train

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score

# pip install catboost
import catboost
from catboost import CatBoostClassifier
from catboost.utils import eval_metric


# pip install pytorch-tabnet
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier


fnameDataBeta = '/common/home/nerobova_a/DataMethy/MatrixBetaNoNanNoBadCpGandPerson.pkl'
fnameDataLabels = '/common/home/nerobova_a/DataMethy/markerList.txt'
fnamesavegraph = '/common/home/nerobova_a/DataMethy/output'


def LoadingDataBeta(fnameDataBeta):
    with open(fnameDataBeta, 'rb') as handle:
        dfDataBeta = pd.DataFrame(pickle.load(handle))

    print("Data Beta:")
    print("Shape: ", dfDataBeta.shape)
    print(dfDataBeta.head(), '\n')

    print("columns: ")
    print(dfDataBeta.columns.values)

    return dfDataBeta.values, list(dfDataBeta.columns.values)


def returnSizeDataX(fnameDataBeta):
    X, feature_names = LoadingDataBeta(fnameDataBeta)
    return X.shape[1], feature_names


def LoadingDataLabels(fnameDataLabels):
    dfLabels = pd.read_csv(fnameDataLabels, header=None)
    dfLabels.rename(columns={0: 'labels'}, inplace=True)

    print("Data Labels:\n")
    print("Shape: ", dfLabels.shape, '\n')
    print(dfLabels.head(), '\n')

    return dfLabels.values


def CreateTrainTest(fnameDataBeta, fnameDataLabels):
    X, feature_names = LoadingDataBeta(fnameDataBeta)
    y = LoadingDataLabels(fnameDataLabels)

    X_train, X_trainval, y_train, y_trainval = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_trainval, y_trainval, test_size=0.5, stratify=y_trainval,
                                                    random_state=42)

    y_train, y_val, y_test = np.asarray(y_train).reshape(-1), np.asarray(y_val).reshape(-1), np.asarray(y_test).reshape(
        -1)

    return X_train, y_train, X_test, y_test, X_val, y_val, feature_names


def f1_metrix(y_train, train_preds, y_test, test_preds):
    print("\nTest F1: %.2f" % f1_score(y_test, test_preds))
    print("Train F1: %.2f" % f1_score(y_train, train_preds))


def confusion_matrix_plot(y_test, y_pred, name):
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred))

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix_df, annot=True)
    fig.savefig(fnamesavegraph + '/graph_' + name + 'matrix.png')
    plt.close()


def graph(labels, approxes, iterations):
    accuracy = eval_metric(labels, approxes, 'Accuracy')
    f1 = eval_metric(labels, approxes, 'F1')
    fig, ax = plt.subplots()
    sns.set_theme(style="darkgrid")
    sns.lineplot(iterations, accuracy)
    fig.savefig(fnamesavegraph + '/graph_' + 'accuracy.png')
    plt.close()

    fig, ax = plt.subplots()
    sns.set_theme(style="darkgrid")
    sns.lineplot(iterations, f1)
    fig.savefig(fnamesavegraph + '/graph_' + 'f1.png')
    plt.close()


def catboost_def(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, n_class):
    iterations = 150
    y_pred_test, y_pred_train = catboost_train.catboost_CatBoost(X_train, y_train, X_test, y_test, X_val, y_val,
                                                           feature_names, n_class, iterations)
    confusion_matrix_plot(y_test, y_pred_test, 'catboost_CatBoost')

    y_pred_test1, y_pred_train1 = catboost_train.catboost_CatBoostClassifier(X_train, y_train, X_test, y_test, X_val, y_val,
                                                                       feature_names, iterations)
    confusion_matrix_plot(y_test, y_pred_test1, 'catboost_CatBoostClassifier')



def lightgbm(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, n_class):
    y_pred_test, y_pred_train = lightgbm.lightgbm_train(X_train, y_train, X_test, y_test, X_val, y_val, feature_names,
                                                        n_class)
    confusion_matrix_plot(y_test, y_pred_test, 'lightgbm_train')

    y_pred_test1, y_pred_train1 = lightgbm.lightgbm_LGBMClassifier(X_train, y_train, X_test, y_test, X_val, y_val,
                                                                 feature_names, n_class)
    confusion_matrix_plot(y_test, y_pred_test1, 'lightgbm_LGBMClassifier')


def tabnet_main(X_train, y_train, X_test, y_test):
    clf = TabNetMultiTaskClassifier()
    # eval_set=[(X_train, y_train), (X_val, y_val)]
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)]
    )
    preds = clf.predict(X_test)


def main():
    n_class = 25
    X_train, y_train, X_test, y_test, X_val, y_val, feature_names = CreateTrainTest(fnameDataBeta, fnameDataLabels)
    catboost_def(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, n_class)


if __name__ == "__main__":
    main()