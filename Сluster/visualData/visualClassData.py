import pandas as pd
import seaborn as sns
import globalConstants


def get_class_distribution(obj):
    count_dict = {}
    for i in range(globalConstants.NUM_CLASSES):
        count_dict["rating_"+str(i)] = 0
    # count_dict = {
    #     "rating_0": 0,
    #     "rating_1": 0,
    #     "rating_2": 0,
    #     "rating_3": 0,
    #     "rating_4": 0,
    #     "rating_5": 0,
    #     "rating_6": 0,
    #     "rating_7": 0,
    #     "rating_8": 0,
    #     "rating_9": 0,
    #     "rating_10": 0,
    #     "rating_11": 0,
    #     "rating_12": 0,
    #     "rating_13": 0,
    #     "rating_14": 0,
    #     "rating_15": 0,
    #     "rating_16": 0,
    #     "rating_17": 0,
    #     "rating_18": 0,
    #     "rating_19": 0,
    #     "rating_20": 0,
    #     "rating_21": 0,
    #     "rating_22": 0,
    #     "rating_23": 0,
    #     "rating_24": 0
    # }

    for i in obj:
        count_dict['rating_' + str(i)] += 1

    return count_dict


def VisualClassData(y_train, y_val, y_test):
    a = get_class_distribution(y_train)
    print("Train barplot")
    sns_plot = sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(y_train)]).melt(), x="variable",
                y="value")
    sns_plot.figure.savefig("output.png")

    print("Test barplot")
    sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(y_test)]).melt(), x="variable",
                y="value")

    print("Val barplot")
    sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(y_val)]).melt(), x="variable",
                y="value")
