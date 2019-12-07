from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from plot_til import utils
import pyheatmap.heatmap as heatmap
from PIL import Image, ImageOps
import cv2
import matplotlib.cm as cm
from PIL import Image
from sklearn import preprocessing
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
PLOT_DIR = './out/plots'


def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')

def plot_conv_output(conv_img, i,id_y_dir):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    #ori_dir = os.path.join(id_y_dir, 'original_map')

    #print(conv_img)
    if i in [0,1,2,10,11,12,13,14,15,23,24,25]:
        for j in range(np.shape(conv_img)[3]):
            conv_dir=os.path.join(id_y_dir,'layer_'+str(i))
            utils.prepare_dir(conv_dir)
            file_name=os.path.join(conv_dir,'_conv_'+str(j)+'.png')
            gray_img=np.array(conv_img[0,:,:,j])
            v_min=np.min(gray_img)
            v_max=np.max(gray_img)
            img=(gray_img-v_min)/(v_max-v_min)*255
            cv2.imwrite(file_name,img)
            #cv2.imwrite(file_name,gray_img*255)
            #cv2.imshow('gray',gray_img)
            #cv2.waitKey()
            #gray_img=cv2.cvtColor(conv_img[0,:,:,j],cv2.COLOR_RGB2GRAY)
            #cv2.imwrite(file_name,gray_img)
    # create directory if does not exist, otherwise empty it
    '''
    utils.prepare_dir(id_y_dir, empty=False)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, :, :,  l]
        # put it on the grid
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # save figure
    #(os.path.join(plot_dir, 'conv{}.png'.format(i)))
    plt.savefig(os.path.join(id_y_dir, 'conv{}.png'.format(i)), bbox_inches='tight')
    #return plot_dir
    '''
MASKed_dir='./out/masked'
def plot_masked(masked_img, step,idx,U_value,i):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(MASKed_dir, step)
    plot_dir=os.path.join(plot_dir,str(idx))
    #plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=False)
    file_name=os.path.join(plot_dir,str(idx)+'_'+str(int(U_value*1000))+'_'+str(i)+'.png')
    cv2.imwrite(file_name,masked_img)
    return plot_dir

def plot_fake_xy(fake_y, fake_x, id_x, id_y, ori_x,ori_y,fake_dir):
    #x_dir = os.path.join(fake_dir, str(id_x))
    #y_dir=os.path.join(fake_dir,str(id))
    #plot_dir=os.path.join(plot_dir,str(idx))
    fake_x_img = (np.array(fake_x) + 1.0) * 127.5
    fake_x_img = cv2.cvtColor(fake_x_img, cv2.COLOR_RGB2BGR)
    fake_y_img = (np.array(fake_y) + 1.0) * 127.5
    fake_y_img = cv2.cvtColor(fake_y_img, cv2.COLOR_RGB2BGR)
    #ori_x=cv2.cvtColor(ori_x,cv2.COLOR_RGB2BGR)
    #ori_y = cv2.cvtColor(ori_y, cv2.COLOR_RGB2BGR)

    utils.prepare_dir(fake_dir, empty=False)
    file_nameOX = os.path.join(fake_dir, str(id_x) + '_oriX.png')
    cv2.imwrite(file_nameOX, ori_x)
    file_nameX=os.path.join(fake_dir,str(id_x)+'_fakeX.png')
    cv2.imwrite(file_nameX,fake_y_img)
    file_nameOY = os.path.join(fake_dir, str(id_y) + '_oriY.png')
    cv2.imwrite(file_nameOY, ori_y)
    file_nameY = os.path.join(fake_dir, str(id_y) + '_fakeY.png')
    cv2.imwrite(file_nameY, fake_x_img)


def generate_occluded_imageset(image, width=256,height=256,occluded_size=16):
    data=np.empty((width*height+1,width,height,3),dtype="float32")
    data[0,:,:,:]=image
    cnt=1
    for i in range(height):
        for j in range(width):
            i_min = int(i - occluded_size / 2)
            i_max = int(i + occluded_size / 2)
            j_min = int(j - occluded_size / 2)
            j_max = int(j + occluded_size / 2)
            if i_min < 0:
                i_min = 0
            if i_max > height:
                i_max = height
            if j_min < 0:
                j_min = 0
            if j_max > width:
                j_max = width
            data[cnt,:,:,:]=image
            data[cnt,i_min:i_max,j_min:j_max,:]=255
            #print(data[i].shape)
            cnt += 1
    return data


def draw_heatmap(occ_map_path, ori_img,save_dir):
    #occ_map_path = '/home/root123/data/FCGAN/FCGAN_CODE/result/occ_test/occlusion_map.txt'
    occ_map = np.loadtxt(occ_map_path, dtype=np.float64)

    min_max_scaler = preprocessing.MinMaxScaler()
    occ_map = min_max_scaler.fit_transform((occ_map))
    #print(occ_map)
    #plt.imshow(abs(occ_map), cmap=cm.hot_r)
    #plt.colorbar()
    #plt.show()
    
    occ_map_img = Image.fromarray(np.uint8(cm.hot_r(abs(occ_map))*255))
    occ_map_img=cv2.cvtColor(np.asarray(occ_map_img),cv2.COLOR_RGB2BGR)
    frame=ori_img
    overlay=frame.copy()
    alpha = 0.5
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.addWeighted(occ_map_img, alpha, frame, 1 - alpha, 0, frame)
    cv2.imwrite(os.path.join(save_dir),frame)
    '''
    occ_map_img.save('occ_map.png')
    ori_image = Image.open(ori_path)
    
    blend_img=Image.blend(ori_image,occ_map_img.convert('RGB'),0.4)
    blend_img.show()
    blend_img.save(save_dir)
    
    hmap=cv2.imread('occ_map.png')
    cv2.imshow('hmap',hmap)
    ori_path = '/home/root123/data/FCGAN/FCGAN_CODE/result/occ_test/787.png'
    frame=cv2.imread(ori_path)
    overlay=frame.copy()
    alpha=0.5
    cv2.rectangle(overlay,(0,0),(frame.shape[1],frame.shape[0]),(255,0,0),-1)
    cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
    cv2.addWeighted(hmap,alpha,frame,1-alpha,0,frame)
    cv2.imshow('frame',frame)
    cv2.waitKey()
    '''

if __name__=='__main__':
    '''
    print('ok')
    img=cv2.imread('/home/root123/data/datasets/target/toxo40_test/toxo40/toxo_0000000.png')
    img=cv2.resize(img,(256,256))
    imgs=generate_occluded_imageset(np.array(img) / 127.5 - 1., width=256, height=256, occluded_size=16)

    for i in range(len(imgs)):
        cv2.imshow('img:{}'.format(i),imgs[i])
        cv2.waitKey(0)
    '''
    idx=245
    image=cv2.imread('/result/occ_test/'+idx+'.png')
    draw_heatmap('/home/root123/data/FCGAN/FCGAN_CODE/result/occ_test/occlusion_map_'+idx+'.txt', ori_img=image,save_dir='y_1.png')
    '''
    ori_path='/home/root123/data/FCGAN/FCGAN_CODE/result/occ_test/201.png'
    occ_map = np.array(np.loadtxt('/home/root123/data/FCGAN/FCGAN_CODE/result/occ_test/occlusion_map_201.txt', dtype=np.float64))
    occ_map=abs(occ_map)-2*0.8037269
    plt.imshow(abs(occ_map), cmap=cm.hot)
    plt.colorbar()
    plt.show()

    occ_map_img = Image.fromarray(np.uint8(cm.hot(abs(occ_map)) * 255))
    occ_map_img.save('occ_map_img.png')
    
    occ_map_img.show()
    ori_image = Image.open(ori_path)

    blend_img = Image.blend(ori_image, occ_map_img.convert('RGB'), 0.5)
    blend_img.show()
    '''
    #blend_img.save(save_dir)
    '''
    print('shape1:',np.shape(vocc_map))
    img=np.uint8(cm.hot(vocc_map) * 255)
    print('shape2:', np.shape(img))
    print(np.shape(img))
    img=cv2.applyColorMap(img,cv2.COLORMAP_JET)
    cv2.imshow('img',img)
    cv2.waitKey()
    print(img)
    '''

    #print(abs(occ_map))
