import pandas as pd

import globalConstants
from metrixs.metrixsPyTorch import get_dataFrame_metrics
from visualData.visualMetrics import loss_graph
from loadCreateData.lodingData import LoadingDataHome, LoadingDataLabelsHome


def tableMetrics(y_train_true, y_train_predict, y_val_true, y_val_predict, y_test_true=None,
                 y_test_predict=None, path=globalConstants.fNameSaveOutput):
    print("Metrics table...")
    dfTableMetricsTrain = get_dataFrame_metrics(y_train_true, y_train_predict)
    dfTableMetricsVal = pd.DataFrame()
    dfTableMetricsTest = pd.DataFrame()
    keys = ["train"]
    if y_val_true is not None and y_val_predict is not None:
        dfTableMetricsVal = get_dataFrame_metrics(y_val_true, y_val_predict)
        keys.append("val")
    if y_test_true is not None and y_test_predict is not None:
        dfTableMetricsTest = get_dataFrame_metrics(y_test_true, y_test_predict)
        keys.append("test")
    if y_test_true is None and y_test_predict is None:
        TableAll = pd.concat([dfTableMetricsTrain, dfTableMetricsVal], keys=keys, axis=1)
    else:
        TableAll = pd.concat([dfTableMetricsTrain, dfTableMetricsVal, dfTableMetricsTest], keys=keys, axis=1)

    TableAll.to_excel(path + '/TableMetrics.xlsx')

    print("Done")
    return TableAll


def tableEpochLoss(loss_info, name, path=globalConstants.fNameSaveOutput):
    print("Loss table...")
    tableLoss = pd.DataFrame(loss_info)
    tableLoss.columns = ['epoch', 'train', 'validation']
    tableLoss.to_excel(path + '/TableEpochLoss' + name + '.xlsx', index=False)

    loss_graph(loss_info, name, path)
    print("Done")
    return


def tableGroundTruth(trueLabelTrain, predictLabelTrain, trueLabelVal, predictLabelVal, DataBetaNamePersonTrain,
                     DataBetaNamePersonVal, trainPredsVect, valPredsVect, name, path=globalConstants.fNameSaveOutput):
    dictPerson = {}
    InfoPerson = []
    # dictPerson = {DataBetaNamePerson[num]: }

    for idx in range(len(DataBetaNamePersonTrain)):
        listInfoPerson = [DataBetaNamePersonTrain[idx], trueLabelTrain[idx], predictLabelTrain[idx]]
        for predict in trainPredsVect[idx]:
            listInfoPerson.append(predict)
        listInfoPerson.append("Train")
        InfoPerson.append(listInfoPerson)

    for idx in range(len(DataBetaNamePersonVal)):
        listInfoPerson = [DataBetaNamePersonVal[idx], trueLabelVal[idx], predictLabelVal[idx]]
        for predict in valPredsVect[idx]:
            listInfoPerson.append(predict)
        listInfoPerson.append("Val")
        InfoPerson.append(listInfoPerson)

    columns = ['id_person', 'ground_truth', 'predicted_label']
    for class_num in range(globalConstants.NUM_CLASSES):
        columns.append('class_' + str(class_num))
    columns.append('Train|Val')

    TableInfoPerson = pd.DataFrame(InfoPerson, columns=columns)
    TableInfoPerson.to_excel(path + '/TableInfoPerson' + name + '.xlsx', index=False)
