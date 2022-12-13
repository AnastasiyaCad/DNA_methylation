import pandas as pd
import numpy as np
import os.path
import pickle5 as pickle


path_files = r'E:\MLTCGAdata\RSrudioStandartIllumuna'
path_27k = r'E:\MLTCGAdata\RSrudioStandartIllumuna\ann27k.txt'
path_450k = r'E:\MLTCGAdata\RSrudioStandartIllumuna\ann450k.txt'
path_850k = r'E:\MLTCGAdata\RSrudioStandartIllumuna\ann850k.txt'
path_matrixBetaValue = r'E:\MLTCGAdata\MatrixCpGAllCancer\MatrixCpGAllCancer.pkl'
path_markerList = r'E:\MLTCGAdata\MatrixCpGAllCancer\markerList.txt'


# считывание txt файла в DataFrame
def readCSVToDataFrame(path):
    return pd.read_csv(path, sep=" ", header=None)


def readPklToDataFrame(path):
    return pd.read_pickle(path)


def readTxtToList(path, ras):
    text_file = open(path, "r")
    lines = text_file.read().split(ras)
    return lines


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


def GetNewBetaMatrixCrossingStandards():
    arrayDataCpGCross = getCrossingStandards()
    #matrixBetaValue = readPklToDataFrame(path_matrixBetaValue)
    #headMatrixBetaValue = list(matrixBetaValue.columns.values)
    headMatrixBetaValue = readTxtToList(path_files + '/colums_matrix.txt', '\n')
    dictCpGToNumColumns = getDictCpGToNumColumns(headMatrixBetaValue)

    listNumColumnNewMatrix = []

    listCpG = list(dictCpGToNumColumns.keys())
    for nameCpG in arrayDataCpGCross:
        if nameCpG in listCpG:
            listNumColumnNewMatrix.append(dictCpGToNumColumns[nameCpG])

    print("Columns of new matrix: ", len(listNumColumnNewMatrix))

    # matrixBetaValueDimReduction = matrixBetaValue.drop(matrixBetaValue.columns[listNumColumnNewMatrix], axis=1)
    #print(matrixBetaValueDimReduction.head)
    #matrixBetaValueDimReduction.to_pickle(path_files + '/matrixBetaValueDimReduction.pkl')

    # headMatrixBetaValue.to_csv(path_files + '/columnsMatrixBeta.txt', index=False, header=None)


def main():
    GetNewBetaMatrixCrossingStandards()
    return 0


if __name__ == '__main__':
    main()