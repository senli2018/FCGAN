import numpy as np
import matplotlib.pyplot as plt

import data_reader as dr
import os

def computeAccuracy(Uy,y_labels,threshold=0.5):
    #threshold=0.5
    pre_postive=0
    pre_negative=0
    fp=fn=tp=tn=0
    #print(Uy)
    for i in range(len(Uy)):
        if Uy[i][0]>threshold:
            pre_postive+=1
            if y_labels[i]==0:
                tp+=1
            else:
                fp+=1
        elif Uy[i][0]<1-threshold:
            pre_negative+=1
            if y_labels[i]==1:
                tn+=1
            else:
                fn+=1
    #print('number:',tp+fp+tn+fn)
    #print('len:', len(Uy))
    temp=0
    if tp<fn:
        temp=tp
        tp=fn
        fn=temp
    if tn<fp:
        temp=tn
        tn=fp
        fp=temp
    accuracy=(tn + tp) / (tn + tp + fn + fp)
    sensitive=tp/(tp+fn)
    specificity=tn/(fp+tn)
    precision=tp/(tp+fp)
    recall=sensitive
    f1_score=2*precision*recall/(precision+recall)

    #if accuracy<0.5:
    #    accuracy=1-accuracy

    return accuracy, tp, tn, fp,fn,f1_score,recall,precision,specificity


if __name__ == '__main__':
    UC_name = "banana"
    model = "20190414-2137"
    target_dir = "/home/root123/data/datasets/target/toxo40/"

    y, y_idx_list, y_data_n, y_labels = dr.get_target_batch(0, 224, 224, target_dir=target_dir)
    Uy = np.loadtxt('./checkpoints/20190416-2037/max/U.txt', delimiter=",")
    accuracy=computeAccuracy(Uy,y_labels,0.5)
    print(accuracy)


'''
    distances = Uy

    if not os.path.exists("roc/"):
        os.makedirs("roc/")
    f = open("roc/" + UC_name + ".txt", 'w')
    f.seek(0)
    f.truncate()
    f.write("阈值\t正样本\t负样本\t预测正样本\t预测负样本\tFN\tFP\tTN\tTP\tFPR\tTPR\tPrecision\tRecall\tf1_score" + '\n')
    threshold = 0.55
    fps = []
    tps = []
    #while threshold <= 1:
    pre_postive = 0
    pre_negative = 0
    fn = 0
    fp = 0
    tn = 0
    tp = 0
    for index in range(len(distances)):
    distance = distances[index]
    if distance[0] > threshold:
        pre_postive += 1
        if y_labels[index] == 0:
            tp += 1
        else:
            fp += 1
    else:
        pre_negative += 1
        if y_labels[index] == 1:
            tn += 1
        else:
            fn += 1
print('number:',tp+fp+tn+fn)
accuracy=(tn + tp) / (tn + tp + fn + fp)
if accuracy<0.5:
    accuracy=1-accuracy
print("accuracy", accuracy)




fp /= 9.1
tn /= 9.1
if (tp + fp) != 0 and (tp + fn) != 0:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) != 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0
else:
    precision = 0
    recall = 0
    f1_score = 0
fps.append(fp / (fp + tn))
tps.append(tp / (fn + tp))
f.write("%f\t8159\t4979\t%d\t%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f" % (
threshold, pre_postive, pre_negative, fn, fp, tn, tp, fp / (fp + tn), tp / (fn + tp), precision, recall, f1_score) + '\n')
threshold += 0.01
print("accuracy", (tn + tp) / (tn + tp + fn + fp))


auc = 0
for distance1 in distances[:count_1-1]:
    for distance2 in distances[count_1:]:
        if distance1[0] < distance2[0]:
            auc += 1
        elif distance1[0] == distance2[0]:
            auc += 0.5
auc = auc / (count_1 * count_2)
f.write("AUC:%.5f auc" % auc + '\n')
for i in fps:
    f.write("%.5f" % i)
    f.write(" ")
f.write("\n")
for i in tps:
    f.write("%.5f" % i)
    f.write(" ")
f.write("\n")
print(auc)

f.close()
plt.plot(fps, tps)
plt.xticks(np.arange(0, 1, 0.1))
plt.yticks(np.arange(0, 1, 0.1))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# plt.title("A simple plot")
plt.show()
'''