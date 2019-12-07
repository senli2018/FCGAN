import tensorflow as tf
from model import CycleGAN
from data_reader import get_source_batch, get_target_batch,get_single_img
from datetime import datetime
import os
import logging
import cv2
from utils import ImagePool
import fuzzy
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from compute_accuracy import computeAccuracy
from plot_til.plot_func import plot_fake_xy,plot_conv_output,generate_occluded_imageset,draw_heatmap
from plot_til import utils
from fuzzy import cal_U
import matplotlib.cm as cm
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('lambda1', 10,
                        'weight for forward cycle loss (X->Y->X), default: 10')
tf.flags.DEFINE_integer('lambda2', 10,
                        'weight for backward cycle loss (Y->X->Y), default: 10')
tf.flags.DEFINE_float('learning_rate', 2e-4,  # 2e-4
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('learning_rate2', 2e-6,  # 2e-6
                      'initial learning rate for Adam, default: 0.000002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('X', '/home/root123/data/datasets/source/',
                       'X tfrecords file for training, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('Y', '/home/root123/data/datasets/target/toxo40/',
                       'Y tfrecords file for training, default: data/tfrecords/orange.tfrecords')
tf.flags.DEFINE_string('load_model', '20190508-2326/max',
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('UC_name', "crescent",
                       'name of the source data, default: None')

Ux = Uy = Cx = Cy = None

x_img_path='/home/root123/data/datasets/source/banana/train/0_000004.jpg'
x_name=x_img_path.split('/')[-1].split('.')[0]
y_img_path='/home/root123/data/FCGAN/data/'
y_name=y_img_path.split('/')[-1].split('.')[0]

def train():
    if FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
        print(checkpoints_dir)
    else:
        logging.info('No model to test, stopped!')
        return

    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = CycleGAN(
            batch_size=FLAGS.batch_size,
            image_size=FLAGS.image_size,
            use_lsgan=FLAGS.use_lsgan,
            norm=FLAGS.norm,
            lambda1=FLAGS.lambda1,
            lambda2=FLAGS.lambda2,
            learning_rate=FLAGS.learning_rate,
            learning_rate2=FLAGS.learning_rate2,
            beta1=FLAGS.beta1,
            ngf=FLAGS.ngf
        )
        G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x, Disperse_loss, Fuzzy_loss, feature_x, feature_y = cycle_gan.model()

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()
    logging.info('Network Built!')

    with tf.Session(graph=graph) as sess:
        checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
        meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
        restore = tf.train.import_meta_graph(meta_graph_path)
        restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
        step = int(meta_graph_path.split("-")[2].split(".")[0])
        Ux = np.loadtxt(checkpoints_dir + "/Ux" + FLAGS.UC_name + '.txt', delimiter=",")
        Ux = [[x] for x in Ux]
        Uy = np.loadtxt(checkpoints_dir + "/Uy" + FLAGS.UC_name + '.txt', delimiter=",")
        Cx = np.loadtxt(checkpoints_dir + "/Cx" + FLAGS.UC_name + '.txt', delimiter=",")
        Cx = [Cx]
        Cy = np.loadtxt(checkpoints_dir + "/Cy" + FLAGS.UC_name + '.txt', delimiter=",")
        logging.info('Parameter Initialized!')

        #print('Ux',Ux)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            plots_count=10
            tsne_plot_count=1000
            result_dir='./result'
            fake_dir=os.path.join(result_dir,'fake_xy')
            roc_dir=os.path.join(result_dir,'roc_curves')
            plot_dir=os.path.join(result_dir,'tsne_pca')
            conv_dir=os.path.join(result_dir,'convs')
            #occ_dir=os.path.join(result_dir,'occ_test')
            #utils.prepare_dir(occ_dir)

            x_path = FLAGS.X + FLAGS.UC_name
            x_images, x_id_list, x_len, x_labels, oimg_xs, x_files = get_source_batch(0, 256, 256,
                                                                                      source_dir=x_path)
            y_images, y_id_list, y_len, y_labels, oimg_ys, y_files = get_target_batch(0, 256, 256,
                                                                                      target_dir=FLAGS.Y)
            #Compute Accuracy, tp, tn, fp, fn, f1_score, recall, precision, specificity#
            accuracy, tp, tn, fp, fn, f1_score, recall, precision, specificity=computeAccuracy(Uy,y_labels)
            print("accuracy:%.4f\ttp:%d\ttn:%d\tfp %d\tfn:%d\tf1_score:%.3f\trecall:%.3f\tprecision:%.3f\tspecicity:%.3f\t" %
                  (accuracy, tp, tn, fp, fn,f1_score, recall, precision, specificity))
            #cv2.imshow('201',oimg_ys[201])
            #cv 2.waitKey()
            #draw ROC curves
            '''
            print('y_labels:',np.shape(y_labels))
            print('y_scores:',np.shape(Uy[:,0]))
            fpr,tpr,thresholds=roc_curve(y_labels,Uy[:,1])
            roc_auc=auc(fpr,tpr)
            plt.plot(fpr, tpr)
            plt.xticks(np.arange(0, 1, 0.1))
            plt.yticks(np.arange(0, 1, 0.1))
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("A simple plot")
            plt.show()
            print('fpr:',np.shape(fpr))
            print('tpr:', np.shape(tpr))
            print('thresholds:', np.shape(thresholds))
            '''
            # t-SNE and PCA plots#
            for j in range(plots_count):
                feature_path=os.path.join(checkpoints_dir,'feature_fcgan.npy')
                feature=np.load(feature_path)
                print('feature:',len(feature))
                randIdx = random.sample(range(0, len(y_labels)), tsne_plot_count)
                t_features = []
                t_labels = []
                for i in range(len(randIdx)):
                    t_features.append(feature[randIdx[i]])
                    t_labels.append(y_labels[randIdx[i]])
                # 使用TSNE进行降维处理。从100维降至2维。
                tsne = TSNE(n_components=2, learning_rate=100).fit_transform(t_features)
                #pca = PCA().fit_transform(t_features)
                #设置画布大小
                plt.figure(figsize=(6, 6))
                #plt.subplot(121)
                plt.scatter(tsne[:, 0], tsne[:, 1], c=t_labels)
                #plt.subplot(122)
                #plt.scatter(pca[:, 0], pca[:, 1], c=t_labels)
                plt.colorbar()  # 使用这一句就可以分辨出，颜色对应的类了！神奇啊。
                utils.prepare_dir(plot_dir)
                plt.savefig(os.path.join(plot_dir,'plot{}.pdf'.format(j)))


            for i in range(10):
            #if True:
            #Cross Domain Image Generation#

                x_img,_,x_oimg = get_single_img(x_img_path)
                y_path=os.path.join(y_img_path,str(i+1)+'.jpg')
                y_img,_,y_oimg=get_single_img(y_path)
                fake_y_eval,fake_x_eval, conv_y_eval = sess.run(
                [fake_y, fake_x,  tf.get_collection('conv_output')],
                feed_dict={cycle_gan.x: x_img, cycle_gan.y: y_img})
                #print(np.shape(fake_y_eval))
                #print(np.shape(fake_x_eval))
                #print(np.shape(conv_y_eval))
                plot_fake_xy(fake_y_eval[0], fake_x_eval[0], x_name, str(i+1), x_oimg,y_oimg,fake_dir)
                print('processing:',i)

            #Feature Map Visualization#

                print('conv_len:', len(conv_y_eval))
                print('conv_shape:',np.shape(conv_y_eval[0]))
                id_y_dir=os.path.join(conv_dir, str(y_name))
                #utils.prepare_dir()
                for i, c in enumerate(conv_y_eval):
                    #conv_i_dir=os.path.join(id_y_dir,'_layer_'+str(i))
                    plot_conv_output(c,i,id_y_dir)
                #print(os.path.join(id_y_dir, 'y.png'))
                cv2.imwrite(os.path.join(id_y_dir, 'y.png'), y_oimg)

            #Occlusion Test#
            '''
            if True:
                t_imgs, t_lbs, t_img = get_single_img(t_img_path)
                #s_imgs, s_lbs, t_img = get_single_img(s_img_path)
                width=np.shape(t_imgs[0])[0]
                height=np.shape(t_imgs[0])[1]
                #print('width:',width)
                #print('height:', height)
                data=generate_occluded_imageset(t_imgs[0],width=width,height=height,occluded_size=16)
                #print(data.shape[0])
                #print('Cy:',Cy)
                u_ys=np.empty([data.shape[0]],dtype='float64')
                occ_map=np.empty((width,height),dtype='float64')
                print(occ_map.shape)
                cnt=0
                feature_y_eval = sess.run(
                    feature_y,
                    feed_dict={cycle_gan.y: [data[0]]})
                # print(feature_y_eval)
                idx_u = 0
                u_y0 = cal_U(feature_y_eval[0], Cy, 2, 2)[idx_u]
                occ_value=0
                print('u_y0:',u_y0)
                for i in range(width):
                    print('collum idx:',i)
                    print(str(cnt) + ':' + str(occ_value))
                    for j in range(height):
                        feature_y_eval = sess.run(
                            feature_y,
                            feed_dict={cycle_gan.y: [data[cnt+1]]})
                        # print(feature_y_eval)
                        u_y = cal_U(feature_y_eval[0], Cy, 2, 2)[idx_u]
                        #print('u_y0:', u_y0)
                        #print('u_y:',u_y)
                        occ_value=u_y0-u_y
                        occ_map[i,j]=occ_value
                        #print(str(cnt)+':'+str(occ_value))
                        cnt+=1

                occ_map_path=os.path.join(occ_dir,'occlusion_map.txt')
                np.savetxt(occ_map_path, occ_map, fmt='%0.20f')
                cv2.imwrite(os.path.join(occ_dir, 'occ_test.png'), oimg_ys[id_y])
                draw_heatmap(occ_map_path=occ_map_path,ori_img=t_img,save_dir=os.path.join(occ_dir,'heatmap.png'))
            '''


        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            #save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            #np.savetxt(checkpoints_dir + "/Uy" + FLAGS.UC_name + '.txt', Uy, fmt="%.20f", delimiter=",")
            #np.savetxt(checkpoints_dir + "/Cy" + FLAGS.UC_name + '.txt', Cy, fmt="%.20f", delimiter=",")
            #np.savetxt(checkpoints_dir + "/Ux" + FLAGS.UC_name + '.txt', Ux, fmt="%.20f", delimiter=",")
            #np.savetxt(checkpoints_dir + "/Cx" + FLAGS.UC_name + '.txt', Cx, fmt="%.20f", delimiter=",")
            logging.info("stopped")
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
