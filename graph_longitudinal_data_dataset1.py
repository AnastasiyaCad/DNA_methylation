import random

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy

# путь
path = "E:/Datasets/E-MTAB-7069 (0, 5, 10)"
file_beta_path = path + "/E-MTAB-7069_beta.txt"
file_targets_path = path + "/E-MTAB-7069_targets.txt"

file = open(file_beta_path, 'r')
line_first = file.readline()

line_name_persons = line_first.rstrip().split('\t')

# cg16867657(ELOVL2) cg06639320(FHL2) cg16419235(PENK)
name_cpg = "cg16419235"

for line in file:
    list_line = line.rstrip().split('\t')
    if list_line[0] == name_cpg:
        list_cpg = list_line[1:]
        break

plt.figure()
num_person = 1
for num_person in range(1, 12):
    name_patient = 'patient' + str(num_person)
    list_age = []
    list_cpg_age = []

    file = open(file_targets_path, 'r')
    line = file.readline().split('\t')
    
    for line in file:
        list_line = line.split('\t')
        if name_patient == list_line[4]:
            list_age.append(list_line[2])
            a = list_line[27][:-1]
            ind = line_name_persons.index(a)
            cp = list_cpg[ind]
            list_cpg_age.append(float(cp))
    color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    plt.plot(list_age, list_cpg_age, marker="o")

plt.title(name_cpg + " (PENK)")
plt.xlabel('age')
plt.ylabel('beta-value')
plt.show()