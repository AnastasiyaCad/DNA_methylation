import pandas as pd
import numpy as np
import os.path
import pickle5 as pickle

path_matrixBetaValue = '/common/home/nerobova_a/DataMethy/MatrixBetaNoNanNoBadCpGandPerson.pkl'

matrixBetaValue = pd.read_pickle(path_matrixBetaValue)

print(matrixBetaValue.shape)

matrixBetaValueMini = matrixBetaValue.iloc[:, :500]

print(matrixBetaValueMini.shape)

matrixBetaValueMini.to_pickle('/common/home/nerobova_a/DataMethy/MatrixBetaMini.pkl')
