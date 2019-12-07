# -*- coding: utf-8 -*-
# @Time :2018/8/25   20:18
# @Author : ELEVEN
# @File : 011_批量重命名文件.py
# @Software: PyCharm
import os

# 1. 获取一个要重命名的文件夹的名字
folder_name = "/home/root123/data/datasets/source/crescent"

# 2. 获取那个文件夹中所有的文件名字
file_names = os.listdir(folder_name)
print(file_names)
# 第1中方法
# os.chdir(folder_name)

# 3. 对获取的名字进行重命名即可
# for name in file_names:
#    print(name)
#    os.rename(name,"[京东出品]-"+name)
i = 1 # 可以让每个文件名字都不一样

for name in file_names:
    print(name)
    old_file_name = folder_name + "/" + name
    new_file_name = folder_name + "/" + str(i) + "crescent"+".jpg"
    os.rename(old_file_name, new_file_name)
    i += 1