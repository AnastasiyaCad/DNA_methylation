import os
import urllib
import pandas as pd
from os import path
from os import listdir
from GEOparse import GEOparse
import numpy as np
from numpy import *

# путь до базы
base_path = "C:/Users/PC/Documents/DNA/Work/GPL13534/filtered/liver/GSE48325"
base_path_gsms = base_path + "/gsms"
# путь, куда сохранить бета-значения
beta_save_path = base_path + '/data'
# путь, куда сохранить idat файлы
idat_save_path = beta_save_path
# список файлов в gsms
name_file = []

for f_d in listdir(base_path_gsms):
    full_path = path.join(base_path_gsms, f_d)
    if path.isfile(full_path):
        name_file.append(f_d)

# матрица значений
matrix_beta = np.zeros((len(name_file)))

# словарь cpg с бета значениями
dic_cpg_num = {}

# создаем лист имен idat файлов
list_idat_name = []
# флаг для создания txt файла для dataset с пустыми адресами для скачивния idat files
flag = 0

# лист имен людей
list_name_people = []
# лист cpg
list_cpg = []
list_betas = []

num_person = -1

for file in name_file:
    # открыли файл
    file_path = base_path_gsms + '/' + file
    file_person = open(file_path, 'r')
    num_person += 1
    # в поисках нужных строк
    for line in file_person:
        if "!Sample_supplementary_file" in line:
            download_address = line[29:]
            download_address = download_address[:-1]
            # если адресс пустой, то качаем бета-значения в указанную путь
            # иначе скачиваем idat files
            if download_address == "NONE":
                gsm_data = GEOparse.get_GEO(geo=file[:-4], destdir=beta_save_path, how="full")
                # добавляем имя в лист
                list_name_people.append(file[:-4])
                # от 1 до кол cpg будем брать бета значения
                for num in range(len(gsm_data.table.ID_REF) + 1):
                    cpg_gsm = gsm_data.table.ID_REF[num]
                    # если такого cpg ещё не было
                    if cpg_gsm not in list_cpg:
                        # добавляем в словарь значение, где каждому cpg ставится номер, соответсвующий номеру
                        # строки матрицы с его бета-значенями
                        dic_cpg_num[cpg_gsm] = len(list_cpg)
                        # записываем его
                        list_cpg.append(cpg_gsm)
                        # создаем одномерную матрицу длиной равной количеству людей
                        list_beta = np.zeros(len(name_file), dtype=float)
                        # где соответсвующему номеру человека ставится определённое бета значение
                        list_beta[num_person] = gsm_data.table.VALUE[num]
                        # присоединяем массив как строчку к матрице
                        matrix_beta = vstack((matrix_beta, list_beta))
                        print(num)
                    else:
                        matrix_beta[dic_cpg_num[cpg_gsm]][num_person] = gsm_data.table.VALUE[num]

                    """
                    if cpg_gsm not in list_cpg:
                        list_cpg.append(cpg_gsm)
                        matrix_beta = np.zeros(len(name_file), dtype=float)
                        matrix_beta[num_name_file] = gsm_data.table.VALUE[num]
                        dic_cpg_beta[cpg_gsm] = matrix_beta
                        print(num)
                    else:
                        matrix_beta = dic_cpg_beta[cpg_gsm]
                        matrix_beta[num_name_file] = gsm_data.table.VALUE[num]
                        dic_cpg_beta[cpg_gsm] = matrix_beta

                    """

                flag = 1
            else:
                name_idat = download_address[67:]
                name_idat = name_idat[:-3]
                # urllib.request.urlretrieve(download_address, idat_save_path + '/' + name_idat)
                name_idat = name_idat[:-9]
                if name_idat not in list_idat_name:
                    list_idat_name.append(name_idat)
    file_person.close()

# создание txt файла с "пустым" датасетом
if flag == 1:
    fale_txt = open(base_path + '/' + 'empty_dataset.txt', 'a')
    fale_txt.write(base_path[-9:])
    fale_txt.close()

# открываем excel
xl = pd.ExcelFile(base_path + "/observables.xlsx")
xl.sheet_names
df = xl.parse("Sheet1")
df['basename'] = list_idat_name
print(df)
df.to_csv(base_path + "/observables_new.csv")