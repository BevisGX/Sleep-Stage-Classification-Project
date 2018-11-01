
# transfer label data to sleeping stage data
import numpy as np

label_file = "F:\\Data\\模型验证_儿童医院数据\\sleep_stage_137.csv"
raw_labels = np.loadtxt(label_file, dtype=str,delimiter=',')

print(raw_labels.shape)

labels = np.zeros( (raw_labels.shape[0], 5), dtype=int)
for ii in range(len(raw_labels)):
    #if(ii == 0 or ii == 1):
    #    continue;
    value = raw_labels[ii]
    print(value)
    if(value[1] == "W"):
        labels[ii][0] = 1
    elif(value[1] == "N1"):
        labels[ii][1] = 1
    elif(value[1] == "N2"):
        labels[ii][2] = 1
    elif(value[1] == "N3"):
        labels[ii][3] = 1
    elif(value[1] == "R"):
        labels[ii][4] = 1

print(labels)
labels = labels.astype(np.int)

label_file_2 = "F:\\Data\\模型验证_儿童医院数据\\sleep_stage_137_2.csv"
np.savetxt(label_file_2, labels, delimiter=',', fmt='%1d')
        