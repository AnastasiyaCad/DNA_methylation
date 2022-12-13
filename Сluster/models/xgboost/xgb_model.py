import os
import xgboost as xgb
import numpy as np
import torch

import globalConstants
from metrixs.metrixsPyTorch import torchConfusionMatrix
from getTable import tableMetrics, tableEpochLoss, tableGroundTruth
from visualData.visualMetrics import stacked_bar_graph
from loadCreateData.lodingData import LoadingDataHome, LoadingDataLabelsHome


def XGBoostTrainModel(X_train, y_train, X_val, y_val, path=globalConstants.fNameSaveOutput):
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dval = xgb.DMatrix(data=X_val, label=y_val)

    model_params = {'max_depth': 6, 'objective': 'multi:softprob', 'num_class': globalConstants.NUM_CLASSES}
    #'multi:softmax'

    evals_result = {}

    booster = xgb.train(
        params=model_params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dval, "val")],
        num_boost_round=globalConstants.EPOCHS,
        evals_result=evals_result,
        verbose_eval=False
    )

    train_preds = booster.predict(dtrain)
    val_preds = booster.predict(dval)

    loss_info = {
        'epoch': list(range(len(evals_result['train']['mlogloss']))),
        'train/loss': evals_result['train']['mlogloss'],
        'val/loss': evals_result['val']['mlogloss']
    }

    # сохранение модели
    booster.save_model(path + '/XGBoostModelClassification.model')

    return booster, train_preds, val_preds, loss_info


# функция принимает на вход модель, а
def getPredict(train_preds, val_preds):
    if type(train_preds) != 'tensor' and type(val_preds) != 'tensor':
        train_preds, val_preds = torch.tensor(train_preds), torch.tensor(val_preds)
    _, trainClassPredicted = torch.max(train_preds, 1)
    _, valClassPredicted = torch.max(val_preds, 1)
    return trainClassPredicted.numpy(), valClassPredicted.numpy()


def getIndexBestSlit(X_trainSet, y_trainSet, X_valSet, y_valSet, pathSave):
    list_f1_weighted_val_metrics = []

    # проверка, какой сплит лучший
    for ind in range(len(y_trainSet)):
        X_train, y_train, X_val, y_val = X_trainSet[ind], y_trainSet[ind], X_valSet[ind], y_valSet[ind]

        pathSaveModel = pathSave + "/ModelSplit" + str(ind + 1)
        if not os.path.exists(pathSaveModel):
            os.makedirs(pathSaveModel)

        # получаем модель, векторы достоверности, loss от эпох
        xgbModel, train_preds, val_preds, loss_info = XGBoostTrainModel(X_train, y_train, X_val, y_val, pathSaveModel)

        # получение таблицы с метриками данных train и val
        dfTable = tableMetrics(y_train, train_preds, y_val, val_preds, path=pathSaveModel)

        # будем сравнивать какой сплит лучше, сравнивая метрику f1 по усрднению weighted
        list_f1_weighted_val_metrics.append(dfTable['val']['weighted']['f1'])

    max_f1 = max(list_f1_weighted_val_metrics)
    max_index_f1 = list_f1_weighted_val_metrics.index(max_f1)

    return max_index_f1


def getTablesGraphXGBoost(X_trainSet, y_trainSet, X_valSet, y_valSet, DataBetaNamePersonTrainSet, DataBetaNamePersonTestSet):
    pathSave = globalConstants.fNameSaveOutput + "/XGBoostModel"
    if not os.path.exists(pathSave):
        os.makedirs(pathSave)

    # получение индекса лучшего сплита
    indexBestSlit = getIndexBestSlit(X_trainSet, y_trainSet, X_valSet, y_valSet, pathSave)

    # получаем "лучшие" данные
    X_train = X_trainSet[indexBestSlit]
    y_train = y_trainSet[indexBestSlit]
    X_val = X_valSet[indexBestSlit]
    y_val = y_valSet[indexBestSlit]
    DataBetaNamePersonTrain = DataBetaNamePersonTrainSet[indexBestSlit]
    DataBetaNamePersonVal = DataBetaNamePersonTestSet[indexBestSlit]

    # получаем модель, векторы достоверности, loss от эпох для лучих данныъ
    xgbModel, trainPredsVect, valPredsVect, loss_info = XGBoostTrainModel(X_train, y_train, X_val, y_val, pathSave)

    # получение таблицы с метриками данных train и val
    _ = tableMetrics(y_train, trainPredsVect, y_val, valPredsVect, path=pathSave)

    # получаем предсказанные классы
    trainLabelPredicted, valLabelPredicted = getPredict(trainPredsVect, valPredsVect)

    tableGroundTruth(y_train, trainLabelPredicted, y_val, valLabelPredicted, DataBetaNamePersonTrain,
                     DataBetaNamePersonVal, trainPredsVect, valPredsVect, "XGBoost", pathSave)

    tableEpochLoss(loss_info, "XGBoost", path=pathSave)
    torchConfusionMatrix(y_train, trainPredsVect, "XGBoostTrain", path=pathSave)
    torchConfusionMatrix(y_val, valPredsVect, "XGBoostVal", path=pathSave)

    stacked_bar_graph(y_train, y_val, pathSave)
