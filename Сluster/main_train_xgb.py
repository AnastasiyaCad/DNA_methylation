import xgboost_t
import catboost_
import lightgbm_
import tabnet_

import pickle5 as pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics

fnameDataBeta = r'E:\train_meth\X_val.pickle'
fnameDataLabels = r'E:\train_meth\y_val.pickle'
fnamesavegraph = r'E:\train_meth'


def LoadingDataBeta(fnameDataBeta):
    with open(fnameDataBeta, 'rb') as f:
        f_new = pickle.load(f)
    dfDataBeta = pd.DataFrame(f_new)
    dfDataBeta = dfDataBeta.iloc[:, :500]
    categorical_features_indices = np.where(dfDataBeta.dtypes != float)[0]
    return dfDataBeta.values, list(dfDataBeta.columns.values), categorical_features_indices


def returnSizeDataX(fnameDataBeta):
    X, feature_names, categorical_features_indices = LoadingDataBeta(fnameDataBeta)
    return X.shape[1], feature_names, categorical_features_indices


def LoadingDataLabels(fnameDataLabels):
    with open(fnameDataLabels, 'rb') as f:
        dfLabels = pickle.load(f)
    return dfLabels


def CreateTrainTest(fnameDataBeta, fnameDataLabels):
    X, feature_names, categorical_features_indices = LoadingDataBeta(fnameDataBeta)
    y = LoadingDataLabels(fnameDataLabels)

    X_train, X_trainval, y_train, y_trainval = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_trainval, y_trainval, test_size=0.5, stratify=y_trainval,
                                                    random_state=42)

    y_train, y_val, y_test = np.asarray(y_train).reshape(-1), np.asarray(y_val).reshape(-1), np.asarray(y_test).reshape(
        -1)

    return X_train, y_train, X_test, y_test, X_val, y_val, feature_names, categorical_features_indices


def f1_metrix(y_train, train_preds, y_test, test_preds):
    print("\nTest F1: %.2f" % metrics.f1_score(y_test, test_preds))
    print("Train F1: %.2f" % metrics.f1_score(y_train, train_preds))


def confusion_matrix_plot(y_test, y_pred, name):
    confusion_matrix_df = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred))

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix_df, annot=True)
    fig.savefig(fnamesavegraph + '/graph_' + name + 'matrix.png')
    plt.close()


def auc_graph(y_test, proba_test, y_train, proba_train, name):
    train_fpr, train_tpr, train_threshold = metrics.roc_curve(y_train, proba_train[:, 1])
    test_fpr, test_tpr, test_threshold = metrics.roc_curve(y_test, proba_test[:, 1])

    train_roc_auc = metrics.auc(train_fpr, train_tpr)
    test_roc_auc = metrics.auc(test_fpr, test_tpr)

    fig, ax = plt.subplots()
    plt.title('Receiver Operating Characteristic')
    plt.plot(train_fpr, train_tpr, 'b', label='Train AUC = %0.2f' % train_roc_auc)
    plt.plot(test_fpr, test_tpr, 'g', label='Test AUC = %0.2f' % test_roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig.savefig(fnamesavegraph + '/graph_' + name + '_auc.png')
    plt.close()


def loss_graph(loss_info, name):
    fig, ax = plt.subplots()
    sns.lineplot(loss_info['epoch'], loss_info['train/loss'], label='train')
    sns.lineplot(loss_info['epoch'], loss_info['val/loss'], label='val')
    plt.ylabel('value')
    plt.xlabel('epochs')
    ax.legend()
    ax.set_title('Train/Val Loss/Epoch')
    fig.savefig(fnamesavegraph + '/graph_' + name + '_loss.png')
    plt.close()


def xgboost_main(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, n_class, max_epochs, batch_size):
    test_preds, train_preds, loss_info = xgboost_t.xgboost_train_(X_train, y_train, X_test, y_test,
                                                                                   X_val, y_val, feature_names, n_class)
    confusion_matrix_plot(y_test, test_preds, 'xgboost_train')
    print('xgboost_train')
    loss_graph(loss_info, 'XGBClassifier')

    #y_pred_test1, y_pred_train1, proba_test1, proba_train1, history_loss1 = xgboost_.xgboost_XGBClassifier(X_train, y_train, X_test, y_test,
    #                                                                              X_val, y_val, n_class, max_epochs,
    #                                                                              batch_size)
    #confusion_matrix_plot(y_test, y_pred_test1, 'xgboost_XGBClassifier')

    #print('XGBClassifier')
    #f1_metrix(y_train, y_pred_train1, y_test, y_pred_test1)
    #loss_graph(max_epochs, history_loss1, 'XGBClassifier')
    #auc_graph(y_test, proba_test1, y_train, proba_train1, 'XGBClassifier')


def catboost_main(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, n_class, max_epochs, batch_size, categorical_features_indices):
    iterations = 50

    # y_pred_test, y_pred_train, proba_test, proba_train, history_loss = catboost_.catboost_CatBoost(X_train, y_train, X_test, y_test,
    #                                                                            X_val, y_val, feature_names, n_class,
    #                                                                            iterations, max_epochs, batch_size)
    # confusion_matrix_plot(y_test, y_pred_test, 'catboost_CatBoost')
    #
    # print('CatBoost_train')
    # loss_graph(max_epochs, history_loss, 'CatBoost')
    test_preds, train_preds, loss_info = catboost_.cat(X_train, y_train, X_test, y_test, X_val,
                                                                                  y_val, feature_names, iterations, max_epochs,
                                                                                  batch_size, categorical_features_indices)

    confusion_matrix_plot(y_test, test_preds, 'catboost_CatBoostClassifier')

    print('CatBoostClassifier')
    loss_graph(loss_info, 'CatBoostClassifie')


def lightgbm_main(X_train, y_train, X_test, y_test, X_val, y_val, n_class, max_epochs, batch_size):
    test_preds, train_preds, loss_info = lightgbm_.lightgbm_train(X_train, y_train, X_test, y_test, X_val, y_val, n_class)
    confusion_matrix_plot(y_test, test_preds, 'lightgbm_train')
    print('lightgbm_train')
    loss_graph(loss_info, 'LGBMClassifier')

    # y_pred_test1, y_pred_train1, proba_test1, proba_train1, history_loss = lightgbm_.lightgbm_LGBMClassifier(X_train, y_train, X_test, y_test,
    #                                                                                    X_val, y_val, n_class,
    #                                                                                    max_epochs, batch_size)
    # confusion_matrix_plot(y_test, y_pred_test1, 'lightgbm_LGBMClassifier')
    # print('LGBMClassifier')
    # loss_graph(max_epochs, history_loss, 'LGBMClassifier')


def tabnet_main(X_train, y_train, X_test, y_test, X_val, y_val, max_epochs, batch_size):
    y_pred_test, y_pred_train, loss_info = tabnet_.tabnet_TabNetMultiTaskClassifier(X_train, y_train, X_test, y_test,
                                                                                    X_val, y_val, max_epochs, batch_size)
    confusion_matrix_plot(y_test, y_pred_test, 'tabnet_TabNetMultiTaskClassifier')
    print('TabNetMultiTaskClassifier')
    loss_graph(loss_info, 'TabNetMultiTaskClassifier')


def main():
    n_class = 25
    max_epochs = 500
    batch_size = 64

    X_train, y_train, X_test, y_test, X_val, y_val, feature_names, categorical_features_indices = CreateTrainTest(fnameDataBeta, fnameDataLabels)
    #xgboost_main(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, n_class, max_epochs, batch_size)
    #catboost_main(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, n_class, max_epochs, batch_size, categorical_features_indices)
    #lightgbm_main(X_train, y_train, X_test, y_test, X_val, y_val, n_class, max_epochs, batch_size)
    tabnet_main(X_train, y_train, X_test, y_test, X_val, y_val, max_epochs, batch_size)


if __name__ == "__main__":
    main()