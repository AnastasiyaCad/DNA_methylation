import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import globalConstants

from sklearn import metrics
import pickle5 as pickle
import openpyxl

import torch
from torchmetrics import Specificity, Precision, Recall, F1Score, Accuracy, CohenKappa, MatthewsCorrCoef, AUROC, ConfusionMatrix


# график confusion matrix
def confusionMatrix(y_true, y_pred, name):
    confusion_matrix_df = pd.DataFrame(metrics.confusion_matrix(y_true, y_pred))
    confusion_matrix_df.to_excel(globalConstants.fNameSaveOutput + '/ConfusionMatrix.xlsx')

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix_df, annot=True)
    fig.savefig(globalConstants.fnamesavegraph + '/graph' + name + 'ConfusionMatrix.png')
    plt.close()


def torchConfusionMatrix(y_true, y_pred, name='', path=globalConstants.fNameSaveOutput):
    if type(y_true) != 'tensor' and type(y_pred) != 'tensor':
        y_pred, y_true = torch.tensor(y_pred), torch.tensor(y_true)
    y_pred = torch.argmax(y_pred, dim=1)
    matrix = ConfusionMatrix(num_classes=globalConstants.NUM_CLASSES)
    mc = matrix(y_true, y_pred)
    confusion_matrix_df = pd.DataFrame(mc.numpy())
    confusion_matrix_df.to_excel(path + '/ConfusionMatrix.xlsx')

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix_df, annot=True)
    fig.savefig(path + '/graph' + name + 'ConfusionMatrix.png')
    plt.close()


def multi_acc(y_true, y_pred):
    yPredSoftmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(yPredSoftmax, 1)

    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc


# график acc, loss
# noinspection PyUnresolvedReferences
def plot_val_acc_loss(train_val_acc_df, train_val_loss_df, path=globalConstants.fNameSaveOutput):
    fig, ax = plt.subplots()
    sns.lineplot(data=train_val_acc_df, x="epochs", y="value", hue="variable").set_title(
        'Train-Val Accuracy/Epoch')
    fig.savefig(path + '/graph_' + 'train_val_acc_df.png')
    plt.close()

    fig, ax = plt.subplots()
    sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable").set_title(
        'Train-Val Loss/Epoch')
    fig.savefig(path + '/graph_' + 'train_val_loss_df.png')
    plt.close()


# метрика auroc
# переделать
def auroc_metrics_torch(y_true, y_pred):
    AUROC_list = [None, None]
    average_list = ['macro', 'weighted']
    if type(y_true) != 'tensor' and type(y_pred) != 'tensor':
        y_pred, y_true = torch.tensor(y_pred), torch.tensor(y_true)
    for average in average_list:
        auroc = AUROC(num_classes=globalConstants.NUM_CLASSES, average=average)
        AUROC_list.append(auroc(y_pred, y_true).numpy())
    return AUROC_list


# метрика precision
def precision_metrics_torch(y_true, y_pred):
    precision_list = [None]
    average_list = ['micro', 'macro', 'weighted']
    if type(y_true) != 'tensor' and type(y_pred) != 'tensor':
        y_pred, y_true = torch.tensor(y_pred), torch.tensor(y_true)
    for average in average_list:
        precision = Precision(average=average, num_classes=globalConstants.NUM_CLASSES)
        precision_list.append(precision(y_pred, y_true).numpy())
    return precision_list


# метрика recall
def recall_metrics_torch(y_true, y_pred):
    recall_list = [None]
    average_list = ['micro', 'macro', 'weighted']
    if type(y_true) != 'tensor' and type(y_pred) != 'tensor':
        y_pred, y_true = torch.tensor(y_pred), torch.tensor(y_true)
    y_pred = torch.argmax(y_pred, dim=1)
    for average in average_list:
        recall = Recall(average=average, num_classes=globalConstants.NUM_CLASSES)
        recall_list.append(recall(y_true, y_pred).numpy())
    return recall_list


def f1_metrics_torch(y_true, y_pred):
    F1_list = [None]
    average_list = ['micro', 'macro', 'weighted']
    if type(y_true) != 'tensor' and type(y_pred) != 'tensor':
        y_pred, y_true = torch.tensor(y_pred), torch.tensor(y_true)
    y_pred = torch.argmax(y_pred, dim=1)
    for average in average_list:
        f1 = F1Score(average=average, num_classes=globalConstants.NUM_CLASSES)
        F1_list.append(f1(y_true, y_pred).numpy())
    return F1_list


def accuracy_metrics_torch(y_true, y_pred):
    acc_list = [None]
    average_list = ['micro', 'macro', 'weighted']
    if type(y_true) != 'tensor' and type(y_pred) != 'tensor':
        y_pred, y_true = torch.tensor(y_pred), torch.tensor(y_true)
    y_pred = torch.argmax(y_pred, dim=1)
    for average in average_list:
        acc = Accuracy(num_classes=globalConstants.NUM_CLASSES, average=average)
        acc_list.append(acc(y_true, y_pred).numpy())
    return acc_list


def cohen_kappa_metrics_torch(y_true, y_pred):
    cohen_kappa_list = []
    if type(y_true) != 'tensor' and type(y_pred) != 'tensor':
        y_pred, y_true = torch.tensor(y_pred), torch.tensor(y_true)
    y_pred = torch.argmax(y_pred, dim=1)
    cohenKappa = CohenKappa(num_classes=globalConstants.NUM_CLASSES)
    return [cohenKappa(y_true, y_pred).numpy(), None, None, None]


def matthews_metrics_torch(y_true, y_pred):
    cohen_kappa_list = []
    if type(y_true) != 'tensor' and type(y_pred) != 'tensor':
        y_pred, y_true = torch.tensor(y_pred), torch.tensor(y_true)
    y_pred = torch.argmax(y_pred, dim=1)
    matthewsCorr = MatthewsCorrCoef(num_classes=globalConstants.NUM_CLASSES)
    return [matthewsCorr(y_true, y_pred).numpy(), None, None, None]


def specificity_metrics_torch(y_true, y_pred):
    specificity_list = [None]
    average_list = ['micro', 'macro', 'weighted']
    if type(y_true) != 'tensor' and type(y_pred) != 'tensor':
        y_pred, y_true = torch.tensor(y_pred), torch.tensor(y_true)
    y_pred = torch.argmax(y_pred, dim=1)
    for average in average_list:
        specificity = Specificity(num_classes=globalConstants.NUM_CLASSES, average=average)
        specificity_list.append(specificity(y_true, y_pred).numpy())
    return specificity_list


# данная функция возвращает таблицу DataFrame, где строки - метрики, столбцы - усреднения
def get_dataFrame_metrics(y_true, y_pred):
    dict_metrics = {}
    list_name_metrics = ['f1', 'auroc', 'accuracy', 'precision', 'recall', 'specificity', 'cohen_kappa', 'matthews']

    dict_metrics['f1'] = f1_metrics_torch(y_true, y_pred)
    dict_metrics['auroc'] = auroc_metrics_torch(y_true, y_pred)
    dict_metrics['accuracy'] = accuracy_metrics_torch(y_true, y_pred)
    dict_metrics['precision'] = precision_metrics_torch(y_true, y_pred)
    dict_metrics['recall'] = recall_metrics_torch(y_true, y_pred)
    dict_metrics['specificity'] = specificity_metrics_torch(y_true, y_pred)
    dict_metrics['cohen_kappa'] = cohen_kappa_metrics_torch(y_true, y_pred)
    dict_metrics['matthews'] = matthews_metrics_torch(y_true, y_pred)

    dfMetrics = pd.DataFrame(dict_metrics, index=['no_averaging', 'micro', 'macro', 'weighted']) # columns=list_name_metrics
    dfMetrics = dfMetrics.T
    return dfMetrics
    #
    # with open(globalConstants.fNameSaveOutput + '/MetrixTable.pkl', 'wb') as handle:
    #     pickle.dump(dfMetrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # dfMetrics.to_pickle(globalConstants.fNameSaveOutput + '/dfMetrixTable.pkl')
    # dfMetrics.to_excel(globalConstants.fNameSaveOutput + '/MetrixTable.xlsx')

