import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from tabnet.tabnetClass import TabNetModelClassification


def get_class_distribution_new(obj):
    count_dict = {}

    for obj_ in obj:
        if obj_ not in count_dict.keys():
            count_dict['rating_' + str(obj_)] = 0
        else:
            count_dict['rating_' + str(obj_)] += 1

    return count_dict


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


class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.X_data.shape[0]


# функции метрик
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def CreateModel(X_train, y_train, X_test, y_test, X_val, y_val, BATCH_SIZE, LEARNING_RATE, EPOCHS, fnamesavegraph):
    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_daatset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_daatset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    target_list = []

    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)

    class_count = [i for i in get_class_distribution(y_train).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    print(class_weights)

    class_weights_all = class_weights[target_list]

    weighted_sampler = WeightedRandomSampler(weights=class_weights_all,
                                             num_samples=len(class_weights_all),
                                             replacement=True)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler=weighted_sampler)

    val_loader = DataLoader(dataset=val_daatset, batch_size=1)
    test_loader = DataLoader(dataset=test_daatset, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    input_dim = X_train.shape[1]
    output_dim = 25
    model = TabNetModelClassification(
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax")
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(model)

    accuracy_stats = {'train': [], 'val': []}
    lossy_stats = {'train': [], 'val': []}

    print("Begin training")

    for e in range(1, EPOCHS + 1):
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

        lossy_stats['train'].append(train_epoch_loss/len(train_loader))
        lossy_stats['val'].append(val_epoch_loss/len(val_loader))

        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'] .append(val_epoch_acc/len(val_loader))

        print(
            f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f}'
            f' | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}'
            f'| Val Acc: {val_epoch_acc / len(val_loader):.3f}')

    # Create dataframes
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().\
        melt(id_vars=['index']).rename(columns={'index': 'epochs'})

    train_val_loss_df = pd.DataFrame.from_dict(lossy_stats).reset_index().\
        melt(id_vars=['index']).rename(columns={'index': 'epochs'})

    fig, ax = plt.subplots()
    sns.lineplot(data=train_val_acc_df, x="epochs", y="value", hue="variable").set_title(
        'Train-Val Accuracy/Epoch')
    fig.savefig(fnamesavegraph + '/graph_' + 'train_val_acc_df.png')
    plt.close()

    fig, ax = plt.subplots()
    sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable").set_title(
        'Train-Val Loss/Epoch')
    fig.savefig(fnamesavegraph + '/graph_' + 'train_val_loss_df.png')
    plt.close()

