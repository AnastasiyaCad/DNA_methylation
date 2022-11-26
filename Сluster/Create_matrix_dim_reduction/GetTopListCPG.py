import pandas as pd
import numpy as np
import os.path
import pickle5 as pickle


path_files = '/common/home/nerobova_a/DataMethy/Data/'
path_27k = '/common/home/nerobova_a/DataMethy/Data/ann27k.txt'
path_450k = '/common/home/nerobova_a/DataMethy/Data/ann450k.txt'
path_850k = '/common/home/nerobova_a/DataMethy/Data/ann850k.txt'
path_matrixBetaValue = '/common/home/nerobova_a/DataMethy/MatrixBetaNoNanNoBadCpGandPerson.pkl'


# считывание txt файла в DataFrame
def readCSVToDataFrame(path):
    return pd.read_csv(path, sep=" ", header=None)


def readPklToDataFrame(path):
    return pd.read_pickle(path)


# загрузка трех массивов CpG сайтов (27к, 450k, 850k)
# создание массива их объединения и сохранение этого массива в различных форматах
def getNewCrossingStandards():
    if os.path.exists(path_27k) and os.path.exists(path_450k) and os.path.exists(path_850k):
        data27k = readCSVToDataFrame(path_27k)
        data450k = readCSVToDataFrame(path_450k)
        data850k = readCSVToDataFrame(path_850k)
    else:
        raise FileNotFoundError()

    result = list(set.intersection(set(data27k[0]), set(data450k[0]), set(data850k[0])))
    print("Result array size:  ", len(result))
    print("Saving the result ...")
    with open(path_files + '/CrossingStandards.pkl', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done")
    return result


# получение объединенного массива из файла
# если такого файла нет в рабочей директории, создать его
def getCrossingStandards():
    pathStand = path_files + '/CrossingStandards.pkl'
    if os.path.exists(pathStand):
        with open(pathStand, 'rb') as f:
            arrayDataCpGCross = pickle.load(f)
    else:
        arrayDataCpGCross = getNewCrossingStandards()
    return arrayDataCpGCross


# загрузка словаря, где ключ - имя столбца, значение - номер соответсвующего столбца
# если словаря нет, то он будет создан
def getDictCpGToNumColumns(headMatrixBetaValue):
    if os.path.exists(path_files + '/dictCpGToNumColumns.pkl'):
        with open(path_files + '/dictCpGToNumColumns.pkl', "rb") as fh:
            dictCpGToNumColumns = pickle.load(fh)
        return dictCpGToNumColumns
    else:
        dictCpGToNumColumns = {}
        numColumns = 0
        for nameColumn in headMatrixBetaValue:
            dictCpGToNumColumns[nameColumn] = numColumns
            numColumns += 1
        with open(path_files + '/dictCpGToNumColumns.pkl', 'wb') as handle:
            pickle.dump(dictCpGToNumColumns, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return dictCpGToNumColumns


# создание новой матрицы меньшей размерности
# перебор элементов массива из объединенных стандартов
# проверяется, есть ли данные эелементы в массиве отфильтрованного стандарта 450к TCGA
# из исходной матрицы бета значений выбираются только те столбцы CpG, что есть в объединенном массиве
def GetNewBetaMatrixCrossingStandards():
    arrayDataCpGCross = getCrossingStandards()
    matrixBetaValue = readPklToDataFrame(path_matrixBetaValue)
    headMatrixBetaValue = list(matrixBetaValue.columns.values)
    dictCpGToNumColumns = getDictCpGToNumColumns(headMatrixBetaValue)

    listNumColumnNewMatrix = []

    listCpG = list(dictCpGToNumColumns.keys())
    for nameCpG in arrayDataCpGCross:
        if nameCpG in listCpG:
            listNumColumnNewMatrix.append(dictCpGToNumColumns[nameCpG])
    
    print("Columns of new matrix: ", len(listNumColumnNewMatrix))

    matrixBetaValueDimReduction = matrixBetaValue[matrixBetaValue.columns[listNumColumnNewMatrix]]
    matrixBetaValueDimReduction.to_pickle(path_files + '/matrixBetaValueDimReduction.pkl')
    # сохранение списка CpG старой матрицы
    # headMatrixBetaValue.to_csv(path_files + '/columnsMatrixBeta.txt', index=False, header=None)


def main():
    GetNewBetaMatrixCrossingStandards()
    return 0


if __name__ == '__main__':
    main()
