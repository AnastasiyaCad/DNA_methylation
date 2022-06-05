import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score

# pip install catboost
import catboost
from catboost import CatBoostClassifier
from catboost.utils import eval_metric


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

    y_train, y_test = np.asarray(y_train).reshape(-1), np.asarray(y_test).reshape(-1)

    return X_train, y_train, X_test, y_test, feature_names


def xgboost_main(X_train, y_train, X_test, y_test, n_class):

    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test)

    num_boost_round = 16
    # param train model
    params = {
        # максимальная глубина дерева по умолч 6
        'max_depth': 3,
        'objective': 'multi:softmax',  # error evaluation for multiclass training
        'num_class': n_class,
        'n_gpus': 0
    }
    # train model's
    bst = xgb.train(params, um_boost_round=num_boost_round, dtrain=dtrain,
                    evals=[(dtrain, "train"), (dtest, "test")])
    print('booster', bst)

    y_pred = bst.predict(dtest)

    with plt.style.context("ggplot"):
        fig, ax = plt.subplots()
        xgb.plotting.plot_importance(bst, ax=ax, height=0.6, importance_type="weight")
    fig.savefig(fnamesavegraph + '/graph_' + 'plot_importance.png')
    plt.close()

    print(classification_report(y_test, y_pred))

    return bst, y_pred


def confusion_matrix_plot(y_test, y_pred):
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred))

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix_df, annot=True)
    fig.savefig(fnamesavegraph + '/graph_' + 'matrix.png')
    plt.close()


def catboost_main(X_train, y_train, X_test, y_test, feature_names, iterations):
    # booster = CatBoost(params={'iterations': 150,
    #                            'verbose': 10,
    #                            'loss_function': 'MultiClass',
    #                            'classes_count': n_class})

    booster = CatBoostClassifier(
        iterations=iterations,
        random_seed=43,
        loss_function='MultiClass'
    )
    booster.fit(X_train, y_train, eval_set=(X_test, y_test))
    # booster.fit(
    #     X_train, y_train,
    #     cat_features=feature_names,
    #     verbose=50
    # )
    booster.set_feature_names(feature_names)

    test_preds = booster.predict(X_test)
    train_preds = booster.predict(X_train)

    print("\nTest  Accuracy : %.2f" % booster.score(X_test, y_test))
    print("Train Accuracy : %.2f" % booster.score(X_train, y_train))

    return test_preds, train_preds


def catboost_graph(labels, approxes, iterations):
    accuracy = eval_metric(labels, approxes, 'Accuracy')
    #f1 = eval_metric(labels, approxes, 'F1')
    fig, ax = plt.subplots()
    sns.set_theme(style="darkgrid")
    sns.lineplot(iterations, accuracy)
    fig.savefig(fnamesavegraph + '/graph_' + 'accuracy.png')
    plt.close()


def main():
    n_class = 25
    iterations = 150
    X_train, y_train, X_test, y_test, feature_names = CreateTrainTest(fnameDataBeta, fnameDataLabels)
    y_pred_test, y_pred_train = catboost_main(X_train, y_train, X_test, y_test, feature_names, iterations)
    confusion_matrix_plot(y_test, y_pred_test)
    catboost_graph(y_test, y_pred_test, iterations)


if __name__ == "__main__":
    main()