import pandas as pd
import statsmodels.formula.api as smf

path_person_info = "E:/Work/unn_epic/result/person_info.txt"

file = open(path_person_info, 'r')
line_first = file.readline().rstrip().split('\t')

dict_person_sex = {}
dict_person_age = {}
dict_person_status = {}

list_Sample_Name = []
list_Aux = []

for line in file:
    list_data = line.rstrip().rstrip().split('\t')
    dict_person_status[list_data[0]] = list_data[4]
    dict_person_sex[list_data[0]] = list_data[8]
    dict_person_age[list_data[0]] = int(list_data[9])
    list_Sample_Name.append(list_data[1])
    list_Aux.append(list_data[0])


path_cell = "E:/Work/unn_epic/result/cell_txt.txt"
file = open(path_cell, 'r')
line_first = file.readline().rstrip().split('\t')

list_person_cell_CD8T = []
list_person_cell_CD4T = []
list_person_cell_NK = []
list_person_cell_Bcell = []
list_person_cell_Neu = []

for line in file:
    list_data = line.rstrip().rstrip().split(',')
    list_person_cell_CD8T.append(list_data[1])
    list_person_cell_CD4T.append(list_data[2])
    list_person_cell_NK.append(list_data[3])
    list_person_cell_Bcell.append(list_data[4])
    list_person_cell_Neu.append(list_data[6])

# path cpg
path_beta_norm = "E:/Work/unn_epic/result/R/beta_funnorm_filtered.txt"

file = open(path_beta_norm, 'r')
list_person_of_file = file.readline().rstrip().split('\t')

list_sex_person_of_file = []
list_age_person_of_file = []
list_status_person_of_file = []


for person in list_person_of_file[1:]:
    list_sex_person_of_file.append(dict_person_sex[person[1:].replace('.', '-')])
    list_age_person_of_file.append(dict_person_age[person[1:].replace('.', '-')])
    list_status_person_of_file.append(dict_person_status[person[1:].replace('.', '-')])


p_value = []
num = -1

for line in file:
    num += 1
    list_cg = line.rstrip().split('\t')
    cg = []
    for i in list_cg[1:]:
        cg.append(float(i))
    data = {}
    data[list_cg[0]] = cg
    data['sex'] = list_sex_person_of_file
    data['age'] = list_age_person_of_file
    data['status'] = list_status_person_of_file
    data['cell_CD8T'] = list_person_cell_CD8T
    data['cell_CD4T'] = list_person_cell_CD4T
    data['cell_NK'] = list_person_cell_NK
    data['cell_Bcell'] = list_person_cell_Bcell
    data['cell_Neu'] = list_person_cell_Neu

    df = pd.DataFrame(data)
    formula = list_cg[0] + ' ~ C(status) + age + sex + cell_CD8T + cell_CD4T + cell_NK + cell_Bcell + cell_Neu'
    res = smf.ols(formula=formula, data=df).fit()
    p_value.append(res.pvalues[1])
    print(num)

file_p_value = "E:/Work/unn_epic/result/p_value.txt"
f = open(file_p_value, 'w')
for item in p_value:
    f.write(str(item))
    f.write("\n")


