
import pickle5 as pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics


def xgboost_main(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, n_class, max_epochs, batch_size):
    test_preds, train_preds, loss_info = xgboost_t.xgboost_train_(X_train, y_train, X_test, y_test,
                                                                                   X_val, y_val, feature_names, n_class)


    #y_pred_test1, y_pred_train1, proba_test1, proba_train1, history_loss1 = xgboost_.xgboost_XGBClassifier(X_train, y_train, X_test, y_test,
    #                                                                              X_val, y_val, n_class, max_epochs,
    #                                                                              batch_size)
    #confusion_matrix_plot(y_test, y_pred_test1, 'xgboost_XGBClassifier')

    #print('XGBClassifier')
    #f1_metrix(y_train, y_pred_train1, y_test, y_pred_test1)
    #loss_graph(max_epochs, history_loss1, 'XGBClassifier')
    #auc_graph(y_test, proba_test1, y_train, proba_train1, 'XGBClassifier')


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