import json
import os
from PIL import Image
import numpy as np
label = "goal"

with open("/root/workspace/XU/MED_data/VSPW480p/VSPW_480p/label_num_dic_final.json",'r') as f:
    label_dic = json.load(f)
    index = label_dic[label]
print(label,"-",index)
with open("/root/workspace/XU/MED_data/VSPW480p/VSPW_480p/val.txt",'r') as f:
    val_list = f.readlines()
    for i in range(len(val_list)):
        val_list[i] = val_list[i].strip()

root = "/root/workspace/XU/MED_data/VSPW480p/VSPW_480p/data/"
ans_list = []
all_list_values = []
for i in range(len(val_list)):
    dir_path = val_list[i]
    print(dir_path)
    abs_path = root + dir_path
    mask_path = os.path.join(abs_path,"mask")
    mask_list = os.listdir(mask_path)
    ans_list.append(abs_path)
    list_values = []
    
    for mask_name in mask_list:
        label = Image.open(mask_path + "/" + mask_name)
        label = np.array(label)
        unique_value = np.unique(label)
        list_values.append(unique_value)
        # print(unique_value)
    all_list_values.append(list_values)

print(ans_list)
with open("find_label_in.txt",'w') as f:
    for names, values in zip(ans_list, all_list_values):
        f.write(names + ":\n")
        for value in values:
            f.write(str(value) + "\n")


    