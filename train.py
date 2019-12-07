import tensorflow as tf
from model import CycleGAN
from data_reader import get_source_batch, get_target_batch
from datetime import datetime
import os
import logging
from utils import ImagePool
import fuzzy
import numpy as np
from compute_accuracy import computeAccuracy

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
tf.flags.DEFINE_float('learning_rate', 2e-4,  #2e-4
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('learning_rate2', 2e-6, #2e-6
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
tf.flags.DEFINE_string('load_model', None, #From None or without Max
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('UC_name', "toxo40X",
                       'name of the source data, default: None')



Ux=Uy= Cx= Cy=None
def train():
    if FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

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
        G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x, Disperse_loss, Fuzzy_loss,feature_x,feature_y = cycle_gan.model()
        optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss, Disperse_loss)
        optimizers2 = cycle_gan.optimize2(Fuzzy_loss)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            sess.run(tf.global_variables_initializer())
            step = 1

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            x_path = FLAGS.X + FLAGS.UC_name
            print('now is in FCM initializing!')
            if FLAGS.load_model is None:
                
                x_images, x_id_list, x_len, x_labels,_ ,_= get_source_batch(0, 256, 256, source_dir=x_path)
                y_images, y_id_list, y_len, y_labels,_,_ = get_target_batch(0, 256, 256, target_dir=FLAGS.Y)
                print('x_len',len(x_images))
                print('y_len',len(y_images))
                x_data=[]
                y_data=[]
                for x in x_images:
                    feature_x_eval = ( sess.run(
                        feature_x, feed_dict={cycle_gan.x: [x]}
                    ))
                    x_data.append(feature_x_eval[0])
                for y in y_images:
                    feature_y_eval = (sess.run(
                        feature_y, feed_dict={cycle_gan.y: [y]}
                    ))
                    y_data.append(feature_y_eval[0])
                Ux, Uy, Cx, Cy= fuzzy.initialize_UC_test(x_len,x_data,y_len,y_data, FLAGS.UC_name,checkpoints_dir)
                np.savetxt(checkpoints_dir + "/Ux" + FLAGS.UC_name + '.txt', Ux, fmt="%.20f", delimiter=",")
                np.savetxt(checkpoints_dir + "/Uy" + FLAGS.UC_name + '.txt', Uy, fmt="%.20f", delimiter=",")
                np.savetxt(checkpoints_dir + "/Cx" + FLAGS.UC_name + '.txt', Cx, fmt="%.20f", delimiter=",")
                np.savetxt(checkpoints_dir + "/Cy" + FLAGS.UC_name + '.txt', Cy, fmt="%.20f", delimiter=",")

            else:
                Ux = np.loadtxt(checkpoints_dir + "/Ux" + FLAGS.UC_name + '.txt', delimiter=",")
                Ux = [[x] for x in Ux]
                Uy = np.loadtxt(checkpoints_dir + "/Uy" + FLAGS.UC_name + '.txt', delimiter=",")
                Cx = np.loadtxt(checkpoints_dir + "/Cx" + FLAGS.UC_name + '.txt', delimiter=",")
                Cx = [Cx]
                Cy = np.loadtxt(checkpoints_dir + "/Cy" + FLAGS.UC_name + '.txt', delimiter=",")
            print('FCM initialization is ended! Go to train')
            max_accuracy = 0
            while not coord.should_stop():
                images_x, idx_list, len_x, labels_x,_ ,_= get_source_batch(FLAGS.batch_size, FLAGS.image_size,
                                                                       FLAGS.image_size, source_dir=x_path)
                subUx = fuzzy.getSubU(Ux, idx_list)
                label_x = [x[0] for x in subUx]
                images_y, idy_list, len_y, labels_y,_,_ = get_target_batch(FLAGS.batch_size, FLAGS.image_size,
                                                                       FLAGS.image_size, target_dir=FLAGS.Y)
                subUy = fuzzy.getSubU(Uy, idy_list)
                label_y = [x[0] for x in subUy]
                _,_, Fuzzy_loss_val, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary, Disperse_loss_val,feature_x_eval,feature_y_eval = (
                    sess.run(
                        [optimizers,optimizers2,Fuzzy_loss, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op,
                         Disperse_loss, feature_x, feature_y],
                        feed_dict={cycle_gan.x: images_x, cycle_gan.y: images_y,
                                   cycle_gan.Uy2x: subUy, cycle_gan.Ux2y: subUx,
                                   cycle_gan.x_label: label_x, cycle_gan.y_label: label_y,
                                   cycle_gan.ClusterX: Cx, cycle_gan.ClusterY: Cy}
                    )
                )
                train_writer.add_summary(summary, step)
                train_writer.flush()
                '''
                Optimize Networks
                
                
                if step % 10 == 0:
                    print('-----------Step %d:-------------' % step)
                    logging.info('  G_loss   : {}'.format(G_loss_val))
                    logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
                    logging.info('  F_loss   : {}'.format(F_loss_val))
                    logging.info('  D_X_loss : {}'.format(D_X_loss_val))
                    logging.info('  Disperse_loss : {}'.format(Disperse_loss_val))
                    logging.info('  Fuzzy_loss : {}'.format(Fuzzy_loss_val))
                
                Optimize FCM algorithm
                '''
                if step % 100== 0:
                    print('Now is in FCM training!')
                    y_images, y_id_list, y_len, y_labels,_ ,_= get_target_batch(0, 256, 256, target_dir=FLAGS.Y)
                    print('y_len', len(y_images))
                    #x_data = []
                    y_data = []
                    for y in y_images:
                        feature_y_eval = (sess.run(
                            feature_y, feed_dict={cycle_gan.y: [y]}
                        ))
                        y_data.append(feature_y_eval[0])

                    #print('y_data:',np.sum(y_data,1))
                    Uy, Cy = fuzzy.updata_U(checkpoints_dir, y_data, Uy, FLAGS.UC_name)
                    accuracy, tp, tn, fp, fn, f1_score, recall, precision, specificity=computeAccuracy(Uy, y_labels)

                    print("accuracy:%.4f\ttp:%.4f\ttn:%.4f\tfp %d\tfn:%d" %
                          (accuracy, tp, tn, fp, fn))
                    if accuracy >= max_accuracy:
                        max_accuracy = accuracy
                        if not os.path.exists(checkpoints_dir + "/max"):
                            os.makedirs(checkpoints_dir + "/max")
                        f = open(checkpoints_dir + "/max/step.txt", 'w')
                        f.seek(0)
                        f.truncate()
                        f.write(str(step) + '\n')
                        f.write(str(accuracy) + '\taccuracy\n')
                        f.close()
                        np.save(checkpoints_dir + "/max/feature_fcgan.npy",y_data)
                        np.savetxt(checkpoints_dir + "/max/"+ "/Uy" + FLAGS.UC_name + '.txt', Uy, fmt="%.20f", delimiter=",")
                        np.savetxt(checkpoints_dir + "/max/"+ "/Cy" + FLAGS.UC_name + '.txt', Cy, fmt="%.20f", delimiter=",")
                        np.savetxt(checkpoints_dir + "/max/"+ "/Ux" + FLAGS.UC_name + '.txt', Ux, fmt="%.20f", delimiter=",")
                        np.savetxt(checkpoints_dir + "/max/"+ "/Cx" + FLAGS.UC_name + '.txt', Cx, fmt="%.20f", delimiter=",")
                        save_path = saver.save(sess, checkpoints_dir + "/max/model.ckpt",global_step=step)
                        print("Max model saved in file: %s" % save_path)
                    print('max_accuracy:', max_accuracy)
                    print('mean_U',np.min(Uy,0))
                step += 1
                if step > 10000:
                    break
                if step>10000:
                    logging.info('train stop!')
                    break

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            np.savetxt(checkpoints_dir + "/Uy" + FLAGS.UC_name + '.txt', Uy, fmt="%.20f", delimiter=",")
            np.savetxt(checkpoints_dir + "/Cy" + FLAGS.UC_name + '.txt', Cy, fmt="%.20f", delimiter=",")
            np.savetxt(checkpoints_dir + "/Ux" + FLAGS.UC_name + '.txt', Ux, fmt="%.20f", delimiter=",")
            np.savetxt(checkpoints_dir + "/Cx" + FLAGS.UC_name + '.txt', Cx, fmt="%.20f", delimiter=",")
            logging.info("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
