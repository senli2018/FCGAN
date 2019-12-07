import matplotlib.pyplot as plt
import numpy as np
import data_reader as dr
import random
'''
UC_name = "banana"
model = "20190411-2227"
Uy = np.loadtxt('checkpoints/' + model + '/Uy' + UC_name + '.txt', delimiter=",")
x=Uy[:,0]
y=Uy[:,1]

plt.plot(x[0:10],y[0:10],'ro')
plt.show()
'''

# _*_ coding:utf-8 -*-
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class data():
    def __init__(self, data, target):
        self.data = data
        self.target = target
y_images, y_id_list, y_len, y_labels,_,_ = dr.get_target_batch(0, 256, 256, target_dir='/home/root123/data/datasets/target/toxo40_test/')
randIdx=random.sample(range(0,len(y_labels)),500)

# 加载数据集
feature=np.load('./checkpoints/20190424-2211/max/feature_fcgan.npy')
#feature=np.random.rand(20000,100)
t_features=[]
t_labels=[]
for i in range(len(randIdx)):
    t_features.append(feature[randIdx[i]])
    t_labels.append(y_labels[randIdx[i]])
#print(t_feature.sum())
print(np.shape(t_features))
print(np.shape(t_labels))
iris = load_iris()
# 共有150个例子， 数据的类型是numpy.ndarray
print(iris.data.shape)
# 对应的标签有0,1,2三种
print(iris.target)
# 使用TSNE进行降维处理。从4维降至2维。
#tsne = TSNE(n_components=2, learning_rate=100).fit_transform(feature)
tsne = TSNE(n_components=2, learning_rate=100).fit_transform(t_features)

# 使用PCA 进行降维处理
pca = PCA().fit_transform(t_features)
# 设置画布的大小
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(tsne[:, 0], tsne[:, 1], c=t_labels)
plt.subplot(122)
plt.scatter(pca[:, 0], pca[:, 1], c=t_labels)
plt.colorbar()#使用这一句就可以分辨出，颜色对应的类了！神奇啊。
plt.savefig('plot.pdf')