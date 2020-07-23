import os
import urllib
import urllib.request
import pandas as pd
from os import path
from os import listdir
from GEOparse import GEOparse
import gzip
import shutil

# путь до базы
base_path = "C:/Users/PC/Documents/DNA/Work/GPL13534/filtered/liver/GSE61446"
base_path_gsms = base_path + "/gsms"
# путь, куда сохранить бета-значения
beta_save_path = 'C:/Users/PC/Documents/DNA/Work/GPL13534/filtered/liver/GSE61446/raw_data'
# путь, куда сохранить idat файлы
idat_save_path = "C:/Users/PC/Documents/DNA/Work/GPL13534/filtered/liver/GSE61446/idat_zip"
# список файлов в gsms
name_file = []

for f_d in listdir(base_path_gsms):
    full_path = path.join(base_path_gsms, f_d)
    if path.isfile(full_path):
        name_file.append(f_d)


# создаем лист имен idat файлов
list_idat_name = []
# флаг для создания txt файла для dataset с пустыми адресами для скачивния idat files
flag = 0

# лист имен людей
list_name_people = []
# лист cpg
list_cpg = []
list_cpg.append('ID_REF')
list_betas = []

list_cpg_beta = []
#list_cpg_beta.append(list_cpg)

num_person = -1
flag_idat = 0

for file in name_file:
    # открыли файл
    file_path = base_path_gsms + '/' + file
    file_person = open(file_path, 'r')
    num_person += 1
    flag_idat = 0
    # в поисках нужных строк
    for line in file_person:
        if flag_idat == 2:
            continue
        if "!Sample_supplementary_file" in line:
            download_address = line[29:]
            download_address = download_address[:-1]
            # если адресс пустой, то качаем бета-значения в указанную путь
            # иначе скачиваем idat files
            if download_address == "NONE":
                gsm_data = GEOparse.get_GEO(geo=file[:-4], destdir=beta_save_path, how="full")
                # добавляем имя в лист
                if flag == 0:
                    list_cpg.append(file[:-4])
                    print(file[:-4])
                    list_cpg_beta.append(list_cpg)
                    list_beta_str = list(map(str, gsm_data.table.VALUE))
                    for i in range(len(gsm_data.table.ID_REF)):
                        list_tmp = []
                        list_tmp.append(gsm_data.table.ID_REF[i])
                        list_tmp.append(list_beta_str[i])
                        list_cpg_beta.append(list_tmp)
                        flag = 1
                    print(list_cpg_beta)
                else:
                    list_cpg_beta[0].append(file[:-4])
                    print(file[:-4])
                    list_beta_str = list(map(str, gsm_data.table.VALUE))
                    for i in range(len(gsm_data.table.ID_REF)):
                        if list_cpg_beta[i + 1][0] == gsm_data.table.ID_REF[i]:
                            list_cpg_beta[i + 1].append(list_beta_str[i])
                        else:
                            if gsm_data.table.ID_REF[i] in list_cpg:
                                """"
                                list_tmp = []
                                list_tmp.append(gsm_data.table.ID_REF[i])
                                for j in range(1, len(name_file)):
                                    list_tmp.append(list_tmp)
                                """
                    for cg in list_cpg_beta:
                        print(cg)
            else:
                name_idat = download_address[67:]
                print(name_idat)
                urllib.request.urlretrieve(download_address, idat_save_path + '/' + name_idat)
                with gzip.open(idat_save_path + '/' + name_idat, 'rb') as f_in:
                    with open(beta_save_path + '/' + name_idat[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                name_idat = name_idat[:-12]
                if name_idat not in list_name_people:
                    list_name_people.append(name_idat)
                flag_idat += 1
                if name_idat not in list_idat_name:
                    list_idat_name.append(name_idat)

    file_person.close()

print(list_cpg_beta)

# создание txt файла с "пустым" датасетом

if flag == 1:
    fale_txt = open(base_path + '/' + 'empty_dataset.txt', 'a')
    fale_txt.write(base_path[-9:])
    fale_txt.close()

    # создание txt файла
    f = open(base_path + '/' + 'table_beta.txt', 'a')
    for l in list_cpg_beta:
        for elem in l:
            f.write(elem + " ")
        f.write("\n")

# открываем excel
xl = pd.ExcelFile(base_path + "/observables.xlsx")
xl.sheet_names
df = xl.parse("Sheet1")
df['Basename'] = list_idat_name
df.to_csv(base_path + "/observables_new.csv")
