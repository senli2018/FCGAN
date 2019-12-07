# FCGAN
This page shows the original code and data samples of the paper 'A Novel Transfer Learning Approach for Toxoplasma gondii Microscopic Image Recognition by Fuzzy Cycle Generative Adversarial Network'.


The image samples can be downloaded in 

Toxoplasma under 40X microscope:![Download](https://drive.google.com/open?id=1j7KpYeGyIEWOFKc7uwE91dk7ANHDW9x8)

Toxoplasma under 100X microscope:![Download](https://drive.google.com/open?id=1LeloHJE8p8OE93eYMNLiBDkDADwQvUrV)

Source images of banana:![Download](https://drive.google.com/open?id=19VIN0en_VRRKc927TGTDui34iGuJeW3E)

The code in this page is for 40X toxoplasma image data, and the code for 100X data can be downloaded in ![Download](https://drive.google.com/open?id=1EcQTRcA-hBc8f97uWWQwh2PDfIEBdK32)


# Before 2019.12.7

#FCGAN  Some supplementary results of the FCGAN approach.

## 1.Evaluation of different source objects
For IJCAI 2019 rebuttal, we conduct extensive experiments for different source objects and show their performance on this page. These experiments are divided as four parts, including Image generation in FCGAN, Feature map visualization for each convolutional layers, t-SNE and PCA plot for extracted features, and Occlusion test to find the important discriminative area FCGAN focusing.
 
### 1.1 Image generation in FCGAN

### 1.2 Feature map visualization for each convolutional layers

### 1.3 t-SNE and PCA plot for extracted features

### 1.4 Occlusion test

## 2.Additional compared experiments
Furthermore, we also conduct some other experiments to show the effectiveness of our model. The first is using a pre-trained mode by Imagenet to extract features for T.gondii images, and the second is to replace Cycle GAN in our FCGAN by a GAN. The results of these two compared methods are as follows:

### 2.1 Pre-trained by ImageNet

### 2.3 GAN based model

## 3.Average pulling loss

## 4.Detail architectures of three supervised methods

## For Reviewe #263597

### 1.Detail setting of supervised methods ResNet, VggNet, and GoogleNet
The following table is the structural parameters of the ResNet model.  
![Image text](https://github.com/senli2018/image/blob/master/ResNet.jpg)  
The following table is the structural parameters of the VggNet model.  
![Image text](https://github.com/senli2018/image/blob/master/VggNet.jpg)  
The following table is the structural parameters of the GoogleNet model.  
![Image text](https://github.com/senli2018/image/blob/master/GoogleNet.jpg)  

### 2.Evaluation criteria and correct F1-scores
TP (True Positive): predicting the correct answer  
FP (False Positive): wrong to predict other classes as this class  
FN (False Negative): This type of label is predicted to be other types of labels.  
Precision: refers to the proportion of positive samples in the positive case determined by the classifier:  
![Image text](https://github.com/senli2018/image/blob/master/precision.gif)  
Recall: refers to the proportion of the total positive case that is predicted to be positive:  
![Image text](https://github.com/senli2018/image/blob/master/recall.gif)  
Represents the classifier to determine the correct proportion of the entire sample:  
![Image text](https://github.com/senli2018/image/blob/master/acc.gif)  
F1-score under each category:  
![Image text](https://github.com/senli2018/image/blob/master/fi.gif)   
