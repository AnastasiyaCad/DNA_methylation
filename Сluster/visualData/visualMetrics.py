import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import globalConstants


def loss_graph(loss_info, name, path=globalConstants.fNameSaveOutput):
    fig, ax = plt.subplots()
    sns.lineplot(loss_info['epoch'], loss_info['train/loss'], label='train')
    sns.lineplot(loss_info['epoch'], loss_info['val/loss'], label='val')
    plt.ylabel('value')
    plt.xlabel('epochs')
    ax.legend()
    ax.set_title('Train/Val Loss/Epoch')
    fig.savefig(path + '/graph_' + name + '_loss.png')
    plt.close()


def stacked_bar_graph(train_data, val_data, pathSave):
    train_data = list(train_data)
    val_data = list(val_data)
    listLabelsTrainValNum = []
    labels = ['Class']

    for num_class in range(globalConstants.NUM_CLASSES):
        labels.append(str(num_class))
        countTrain = train_data.count(num_class)
        countVal = val_data.count(num_class)
        listLabelsTrainValNum.append([str(num_class), countTrain, countVal])

    data_frame = pd.DataFrame(listLabelsTrainValNum, columns=['Class', 'Train', 'Val'])
    fig = data_frame.plot(x='Class', kind='bar', stacked=True, title=' Tran Val Class').get_figure()
    fig.savefig(pathSave + '/barGraph.png')


# взяла из старого xgbust

# def auc_graph(y_test, proba_test, y_train, proba_train, name):
#     train_fpr, train_tpr, train_threshold = metrics.roc_curve(y_train, proba_train[:, 1])
#     test_fpr, test_tpr, test_threshold = metrics.roc_curve(y_test, proba_test[:, 1])
#
#     train_roc_auc = metrics.auc(train_fpr, train_tpr)
#     test_roc_auc = metrics.auc(test_fpr, test_tpr)
#
#     fig, ax = plt.subplots()
#     plt.title('Receiver Operating Characteristic')
#     plt.plot(train_fpr, train_tpr, 'b', label='Train AUC = %0.2f' % train_roc_auc)
#     plt.plot(test_fpr, test_tpr, 'g', label='Test AUC = %0.2f' % test_roc_auc)
#     plt.legend(loc='lower right')
#     plt.plot([0, 1], [0, 1], 'r--')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     fig.savefig(fnamesavegraph + '/graph_' + name + '_auc.png')
#     plt.close()



