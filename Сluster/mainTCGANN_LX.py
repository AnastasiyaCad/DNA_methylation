import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# Загрузка данных

fnameDataBeta = r'D:\Nerobova_Anastasiya\DataTCGA\MatrixBetaNoNanNoBadCpGandPerson.pkl'
fnameDataLabels = 'D:/Nerobova_Anastasiya/DataTCGA/markerList.txt'
fnamesavegraph = r'D:\Nerobova_Anastasiya\DataTCGA\graph'


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


# разделение данных на train, test, val


def CreateTrainValTest(fnameDataBeta, fnameDataLabels):
    X = LoadingDataBeta(fnameDataBeta)
    y = LoadingDataLabels(fnameDataLabels)

    X_train, X_trainval, y_train, y_trainval = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_trainval, y_trainval, test_size=0.5, stratify=y_trainval,
                                                    random_state=42)

    y_train, y_val, y_test = np.asarray(y_train).reshape(-1), np.asarray(y_val).reshape(-1), np.asarray(y_test).reshape(
        -1)

    return X_train, y_train, X_test, y_test, X_val, y_val


# визуализация классов

def get_class_distribution(obj):
    count_dict = {
        "rating_0": 0,
        "rating_1": 0,
        "rating_2": 0,
        "rating_3": 0,
        "rating_4": 0,
        "rating_5": 0,
        "rating_6": 0,
        "rating_7": 0,
        "rating_8": 0,
        "rating_9": 0,
        "rating_10": 0,
        "rating_11": 0,
        "rating_12": 0,
        "rating_13": 0,
        "rating_14": 0,
        "rating_15": 0,
        "rating_16": 0,
        "rating_17": 0,
        "rating_18": 0,
        "rating_19": 0,
        "rating_20": 0,
        "rating_21": 0,
        "rating_22": 0,
        "rating_23": 0,
        "rating_24": 0
    }

    for i in obj:
        count_dict['rating_' + str(i)] += 1

    return count_dict


def VisualClassData(y_train, y_val, y_test):
    # Train
    a = get_class_distribution(y_train)
    print("Train barplot")
    sns_plot = sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(y_train)]).melt(), x="variable",
                y="value")
    sns_plot.figure.savefig("output.png")

    # Test
    print("Test barplot")
    sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(y_test)]).melt(), x="variable",
                y="value")

    # Val
    print("Val barplot")
    sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(y_val)]).melt(), x="variable",
                y="value")


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def ReturnTargetList(train_dataset):
    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)
    return target_list


def ReturnClassWeightsAll(y_train, target_list):
    class_count = [i for i in get_class_distribution(y_train).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    print("Class_weights: ", class_weights)

    class_weights_all = class_weights[target_list]
    return class_weights, class_weights_all


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def DatasetTrainTestVal():
    X_train, y_train, X_test, y_test, X_val, y_val = CreateTrainValTest(fnameDataBeta, fnameDataLabels)

    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    return train_dataset, test_dataset, val_dataset, y_train, y_test


def ReturnWeightedSampler(class_weights_all):
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )
    return weighted_sampler


def DeviceTorch():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    return device


def LoaderTrainTestVal(train_dataset, test_dataset, val_dataset, BATCH_SIZE, weighted_sampler):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler=weighted_sampler
                              )
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    return train_loader, test_loader, val_loader


def ModelCreate(class_weights, device, NUM_FEATURES, NUM_CLASSES, LEARNING_RATE):

    model = MulticlassClassification(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)

    return model, criterion, optimizer


def TrainModel(train_loader, val_loader, model, criterion, optimizer, device, EPOCHS):
    print("TRAIN MODEL")
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    for e in range(1, EPOCHS + 1):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()

        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        print(
            f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(val_loader):.3f}')

    i = 0
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()

        y_train_pred = model(X_train_batch)
        print(y_train_batch, '\n', y_train_pred)
        if i == 0:
            break

    return accuracy_stats, loss_stats


def ModelTest(test_loader, model, device):
    print("TEST MODEL")

    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    return y_pred_list


def main():

    EPOCHS = 7
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_FEATURES = returnSizeDataX(fnameDataBeta)
    NUM_CLASSES = 25

    train_dataset, test_dataset, val_dataset, y_train, y_test = DatasetTrainTestVal()

    target_list = ReturnTargetList(train_dataset)

    class_weights, class_weights_all = ReturnClassWeightsAll(y_train, target_list)
    weighted_sampler = ReturnWeightedSampler(class_weights_all)

    device = DeviceTorch()

    train_loader, test_loader, val_loader = LoaderTrainTestVal(train_dataset, test_dataset, val_dataset, BATCH_SIZE, weighted_sampler)

    model, criterion, optimizer = ModelCreate(class_weights, device, NUM_FEATURES, NUM_CLASSES, LEARNING_RATE)

    accuracy_stats, loss_stats = TrainModel(train_loader, val_loader, model, criterion, optimizer, device, EPOCHS)

    # Create dataframes
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})

    # Plot the dataframes
    fig, ax = plt.subplots()
    sns.lineplot(data=train_val_acc_df, x="epochs", y="value", hue="variable").set_title(
        'Train-Val Accuracy/Epoch')
    fig.savefig(fnamesavegraph + '/graph_' + 'train_val_acc_df.png')

    fig, ax = plt.subplots()
    sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable").set_title(
        'Train-Val Loss/Epoch')
    fig.savefig(fnamesavegraph + '/graph_' + 'train_val_loss_df.png')

    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
    sns.lineplot(data=train_val_acc_df, x="epochs", y="value", hue="variable", ax=axes[0]).set_title(
        'Train-Val Accuracy/Epoch')
    fig.savefig(fnamesavegraph + '/graph_' + 'train_val_acc_df.png')
    sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable", ax=axes[1]).set_title(
        'Train-Val Loss/Epoch')
    fig.savefig(fnamesavegraph + '/graph_' + 'train_val_loss_df.png')
    """

    y_pred_list = ModelTest(test_loader, model, device)

    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred_list))  # rename(columns=idx2class, index=idx2class)

    sns.heatmap(confusion_matrix_df, annot=True)
    fig.savefig(fnamesavegraph + '/graph_' + 'matrix.png')
    print(classification_report(y_test, y_pred_list))


if __name__ == "__main__":
    main()
