import data_reader as dr
import fuzzy
import numpy as np
import random
import copy
import math
from compute_accuracy import computeAccuracy

m = 2
def initialize_U(num_data, num_cluster):
    U = []
    for data in range(num_data):
        u_data = []
        sum_random = 0.0
        for cluster in range(num_cluster):
            u_data.append(random.random())
            sum_random += u_data[-1]
        for cluster in range(num_cluster):
            u_data[cluster] /= sum_random
        U.append(u_data)

    return U


def initialize_C(data, U):
    """
    :param data: features of the data
    :param U:
    :param m:
    :return:
    """
    num_cluster = len(U[0])
    C = []
    for j in range(num_cluster):
        current_cluster_center = []
        for i in range(len(data[0])):
            dummy_sum_num = 0.0
            dummy_sum_dum = 0.0
            for k in range(len(data)):
                # 分子
                dummy_sum_num += (U[k][j] ** m) * data[k][i]
                # 分母
                dummy_sum_dum += (U[k][j] ** m)
            # 第i列的聚类中心
            current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
            # 第j簇的所有聚类中心
        C.append(current_cluster_center)
    return C
def initialize_UC(y_data):
    print('y_data:',np.sum(y_data,axis=1))
    y_len=len(y_data)
    #Ux = initialize_U(x_len, 1)
    Uy = initialize_U(y_len, 2)
    #Cx = initialize_C(x_data, Ux)
    Cy = initialize_C(y_data, Uy)
    #print('Cx:',np.shape(Cx))
    print('Cy:',np.shape(Cy))
    return Uy,Cy
def end_conditon(U, U_old):
    """
    结束条件。当U矩阵随着连续迭代停止变化时，触发结束
    """
    epsm = 0.000000001
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > epsm:
                return False
    return True
def calculate_J(data, U, C):
    """
    计算目标函数J
    :param data:
    :param C:
    :return:
    """
    J = 0.0
    for num_C in range(len(C)):
        for num_data in range(len(data)):
            J += (np.power(U[num_data][num_C], m) * np.square(np.subtract(C[num_C], data[num_data])))
    return J


def distance(point, center):
    """
    该函数计算2点之间的距离（作为列表）。我们指欧几里德距离。        闵可夫斯基距离
    """
    if len(point) != len(center):
        return -1
    dummy = 0.0
    for i in range(0, len(point)):
        dummy += abs(point[i] - center[i]) ** 2
    return math.sqrt(dummy)

def updata_U(saveDir, data, U):
    cluster_number = len(U[0])

    for iter in range(100):
        print('第%d 次迭代更新U' % iter)
        U_old = copy.deepcopy(U)
        C = []
        for j in range(0, cluster_number):
            current_cluster_center = []
            for i in range(0, len(data[0])):
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(0, len(data)):
                    dummy_sum_num += (U[k][j] ** m) * data[k][i]
                    dummy_sum_dum += (U[k][j] ** m)
                current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
            C.append(current_cluster_center)

        distance_matrix = []
        for i in range(0, len(data)):
            current = []
            for j in range(0, cluster_number):
                current.append(distance(data[i], C[j]))
            distance_matrix.append(current)
        # 更新U
        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0, cluster_number):
                    dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2 / (m - 1))
                U[i][j] = 1 / dummy

        if end_conditon(U, U_old):
             print("结束聚类")
             break

    np.savetxt(saveDir + "/U.txt", U, fmt="%.20f", delimiter=",")
    np.savetxt(saveDir + "/C.txt", C, fmt="%.20f", delimiter=",")
    return U, C
if __name__=='__main__':
    #x_path="/home/root123/data/datasets/source/banana/"
    model='20190416-2037'
    dir='./checkpoints/+'+model+'/max/'
    y_path="/home/root123/data/datasets/target/toxo40/"
    #x_images, x_id_list, x_len, x_labels = dr.get_source_batch(0, 256, 256, source_dir=x_path)
    y_images, y_id_list, y_len, y_labels = dr.get_target_batch(0, 256, 256, target_dir=y_path)

    feature=np.load(dir+'feature_fcgan.npy')
    Uy,Cy= initialize_UC(feature)
    U,C=updata_U(dir,feature,Uy)
    print(np.shape(U))
    print(np.shape(C))
    
    y, y_idx_list, y_data_n, y_labels = dr.get_target_batch(0, 224, 224, target_dir=y_path)
    #Uy = np.loadtxt('checkpoints/' + model + '/Uy' + UC_name + '.txt', delimiter=",")
    #Uy = np.loadtxt(dir+'/U.txt', delimiter=",")
    accuracy=computeAccuracy(U,y_labels,0.5)
    print('accuracy:',accuracy)
