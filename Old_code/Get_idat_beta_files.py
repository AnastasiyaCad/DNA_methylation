import os
import urllib
import pandas as pd
from os import path
from os import listdir
from GEOparse import GEOparse

# путь до базы
base_path = "C:/Users/PC/Documents/DNA/Work/GPL13534/filtered/liver/GSE48325"
base_path_gsms = base_path + "/gsms"
# путь, куда сохранить бета-значения
beta_save_path = 'C:/Users/PC/Documents/DNA/Work/GPL13534/filtered/liver/GSE48325/bb'
# путь, куда сохранить idat файлы
idat_save_path = "C:/Users/PC/Documents/DNA/Work/GPL13534/filtered/liver/GSE48325/bb"
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

for file in name_file:
    # открыли файл
    file_path = base_path_gsms + '/' + file
    file_person = open(file_path, 'r')
    # в поисках нужных строк
    for line in file_person:
        if "!Sample_supplementary_file" in line:
            download_address = line[29:]
            download_address = download_address[:-1]
            # если адресс пустой, то качаем бета-значения в указанную путь
            # иначе скачиваем idat files
            if download_address == "NONE":
                gsm_data = GEOparse.get_GEO(geo=file[:-4], destdir=beta_save_path, how="full")
                flag = 1
            else:
                name_idat = download_address[67:]
                name_idat = name_idat[:-3]
                #urllib.request.urlretrieve(download_address, idat_save_path + '/' + name_idat)
                name_idat = name_idat[:-9]
                if name_idat not in list_idat_name:
                    list_idat_name.append(name_idat)
    file_person.close()

#создание txt файла с "пустым" датасетом
if flag == 1:
    fale_txt = open(base_path + '/' + 'empty_dataset.txt', 'a')
    fale_txt.write(base_path[-9:])
    fale_txt.close()

#открываем excel
xl = pd.ExcelFile(base_path+"/observables.xlsx")
xl.sheet_names
df = xl.parse("Sheet1")
df['basename'] = list_idat_name
print(df)
df.to_csv(base_path+"/observables_new.csv")