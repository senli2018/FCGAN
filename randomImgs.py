import random
import os
import shutil
from os import scandir
source_path='/home/root123/data/datasets/target/toxo40/toxo40'
target_path='/home/root123/data/datasets/source/toxo40X/toxo'
ratio=0.1
file_paths=[]
for file in scandir(source_path):
    file_paths.append(file.path)
target_count=int(len(file_paths)*ratio)
random_idx=random.sample(range(0,len(file_paths)),target_count)
for i in random_idx:
    file=file_paths[i]
    file_name=file.split("/")[-1]
    target_file=os.path.join(target_path,file_name)
    shutil.copyfile(file,target_file)
print('success')


