import tensorflow as tf
import ops
import utils
from discriminator import Discriminator
from generator import Generator
import numpy as np
from classifier import Classifier


class CycleGAN:
    def __init__(self,
                 batch_size=1,
                 image_size=256,
                 use_lsgan=True,
                 norm='instance',
                 lambda1=10,
                 lambda2=10,
                 learning_rate=2e-4,
                 learning_rate2=2e-6,
                 beta1=0.5,
                 ngf=64
                 ):
        """
        Args:
          batch_size: integer, batch size
          image_size: integer, image size
          lambda1: integer, weight for forward cycle loss (X->Y->X)
          lambda2: integer, weight for backward cycle loss (Y->X->Y)
          use_lsgan: boolean
          norm: 'instance' or 'batch'
          learning_rate: float, initial learning rate for Adam
          beta1: float, momentum term of Adam
          ngf: number of gen filters in first conv layer
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_lsgan = use_lsgan
        use_sigmoid = not use_lsgan
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.learning_rate2 = learning_rate2
        self.beta1 = beta1

        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

        self.G = Generator('G', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
        self.D_Y = Discriminator('D_Y',
                                 self.is_training, norm=norm, use_sigmoid=use_sigmoid)
        self.F = Generator('F', self.is_training, norm=norm, image_size=image_size)
        self.D_X = Discriminator('D_X',
                                 self.is_training, norm=norm, use_sigmoid=use_sigmoid)
        self.C = Classifier("C", self.is_training, norm=norm, use_sigmoid=use_sigmoid)

        self.Uy2x = tf.placeholder(tf.float32, [None, 2], name="Uy2x")
        self.Ux2y = tf.placeholder(tf.float32, [None, 1], name="Ux2y")
        self.x_label = tf.placeholder(tf.float32, [None], name="x_label")
        self.y_label = tf.placeholder(tf.float32, [None], name="y_label")
        self.ClusterX = tf.placeholder(tf.float32, [1, 100], name="ClusterX")
        self.ClusterY = tf.placeholder(tf.float32, [2, 100], name="ClusterY")

        self.x = tf.placeholder(tf.float32,
                                shape=[batch_size, image_size, image_size, 3], name="x")
        self.y = tf.placeholder(tf.float32,
                                shape=[batch_size, image_size, image_size, 3], name="y")
        #self.fake_x=self.F(self.y)
        #self.fake_y=self.G(self.x)
        #self.fake_x = tf.placeholder(tf.float32,
        #                             shape=[batch_size, image_size, image_size, 3], name="fake_x")
        #self.fake_y = tf.placeholder(tf.float32,
        #                             shape=[batch_size, image_size, image_size, 3], name="fake_y")

    def model(self):
        x = self.x
        y = self.y

        #self.fake_x = self.F(self.y)
        #self.fake_y = self.G(self.x)
        cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)

        # X -> Y
        fake_y = self.G(x)
        G_gan_loss = self.generator_loss(self.D_Y, fake_y, self.y_label, use_lsgan=self.use_lsgan)
        G_loss = G_gan_loss + cycle_loss
        D_Y_loss = self.discriminator_loss(self.D_Y, y, fake_y, self.y_label, use_lsgan=self.use_lsgan)

        # Y -> X
        fake_x = self.F(y)
        F_gan_loss = self.generator_loss(self.D_X, fake_x, self.x_label, use_lsgan=self.use_lsgan)
        F_loss = F_gan_loss + cycle_loss
        D_X_loss = self.discriminator_loss(self.D_X, x, fake_x, self.x_label, use_lsgan=self.use_lsgan)

        # fuzzy
        Fuzzy_x_loss, feature_x = self.fuzzy_loss(self.C, x, self.Ux2y, self.ClusterX)
        Fuzzy_y_loss,feature_y = self.fuzzy_loss(self.C, fake_x, self.Uy2x, self.ClusterY)
        Disperse_loss = -self.disperse_loss(feature_y, self.Uy2x)
        Fuzzy_loss = Fuzzy_x_loss + Fuzzy_y_loss#+Disperse_loss

        #feature_x = self.C(x)
        #feature_y = self.C(fake_x)
        # summary
        tf.summary.histogram('D_Y/true', self.D_Y(y))
        tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x)))
        tf.summary.histogram('D_X/true', self.D_X(x))
        tf.summary.histogram('D_X/fake', self.D_X(self.F(y)))

        tf.summary.scalar('loss/G', G_gan_loss)
        tf.summary.scalar('loss/D_Y', D_Y_loss)
        tf.summary.scalar('loss/F', F_gan_loss)
        tf.summary.scalar('loss/D_X', D_X_loss)
        tf.summary.scalar('loss/cycle', cycle_loss)
        tf.summary.scalar('loss/Disperse', Disperse_loss)
        tf.summary.scalar('loss/Fuzzy', Fuzzy_loss)

        tf.summary.image('X/generated', utils.batch_convert2int(self.G(x)))
        tf.summary.image('X/reconstruction', utils.batch_convert2int(self.F(self.G(x))))
        tf.summary.image('Y/generated', utils.batch_convert2int(self.F(y)))
        tf.summary.image('Y/reconstruction', utils.batch_convert2int(self.G(self.F(y))))

        return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x, Disperse_loss, Fuzzy_loss,feature_x,feature_y

    def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss, Disperse_loss):
        def make_optimizer(loss, variables, name='Adam'):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000
            beta1 = self.beta1
            learning_rate = (
                tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                              decay_steps, end_learning_rate,
                                              power=1.0),
                    starter_learning_rate
                )
            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                    .minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step

        G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
        D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
        F_optimizer = make_optimizer(F_loss, self.F.variables, name='Adam_F')
        D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')
        Disperse_optimizer = make_optimizer(Disperse_loss, self.C.variables, name='Adam_D_X')

        #with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer, Disperse_optimizer]):
        #    return tf.no_op(name='optimizers')
        with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
            return tf.no_op(name='optimizers')
    def optimize2(self, Fuzzy_loss):
        def make_optimizer2(loss, variables, name='Adam2'):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate2
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000
            beta1 = self.beta1
            learning_rate = (
                tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                              decay_steps, end_learning_rate,
                                              power=1.0),
                    starter_learning_rate
                )
            )
            tf.summary.scalar('learning_rate2/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                    .minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step

        Fuzzy_optimizer = make_optimizer2(Fuzzy_loss, self.C.variables, name='Adam_Fuzzy')

        with tf.control_dependencies([Fuzzy_optimizer]):
            return tf.no_op(name='optimizers2')

    def discriminator_loss(self, D, y, fake_y, label, use_lsgan=True):
        """ Note: default: D(y).shape == (batch_size,5,5,1),
                           fake_buffer_size=50, batch_size=1
        Args:
          G: generator object
          D: discriminator object
          y: 4D tensor (batch_size, image_size, image_size, 3)
        Returns:
          loss: scalar
        """
        if use_lsgan:
            # use mean squared error
            error_real = tf.reduce_mean(tf.squared_difference(D(y), label))
            error_fake = tf.reduce_mean(tf.square(D(fake_y)))
        else:
            # use cross entropy
            error_real = -tf.reduce_mean(ops.safe_log(D(y)))
            error_fake = -tf.reduce_mean(ops.safe_log(1 - D(fake_y)))
        loss = (error_real + error_fake) / 2
        return loss

    def generator_loss(self, D, fake_y, label, use_lsgan=True):
        """  fool discriminator into believing that G(x) is real
        """
        if use_lsgan:
            # use mean squared error
            loss = tf.reduce_mean(tf.squared_difference(D(fake_y), label))
        else:
            # heuristic, non-saturating loss
            loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
        return loss

    def cycle_consistency_loss(self, G, F, x, y):
        """ cycle consistency loss (L1 norm)
        """
        forward_loss = tf.reduce_mean(tf.abs(F(G(x)) - x))
        backward_loss = tf.reduce_mean(tf.abs(G(F(y)) - y))
        loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return loss

    def disperse_loss(self, data, U, DIM=100, m=2):
        data_n = self.batch_size
        cluster_num = 2
        tensor_m = tf.constant(np.ones([data_n, cluster_num], dtype=np.float32) * m)
        UM = tf.pow(U, tensor_m)
        dumpy_sum_num = tf.matmul(UM, data, transpose_a=True)
        dum = tf.expand_dims(tf.reduce_sum(UM, 0), 1)
        g = []
        for i in range(DIM):
            g.append(dum)
        dumpy_sum_dum = tf.concat(g, axis=1)
        clusters = tf.divide(dumpy_sum_num, dumpy_sum_dum)
        c1 = []
        c2 = []
        for i in range(data_n):
            c1.append(tf.expand_dims(clusters[0], 0))
            c2.append(tf.expand_dims(clusters[1], 0))
        cluster_1 = tf.concat(c1, axis=0)
        cluster_2 = tf.concat(c2, axis=0)
        distance = tf.reduce_mean(tf.sqrt(tf.pow(tf.subtract(cluster_1, cluster_2), 2)))
        return distance
    def disperse_loss_original(self, data, C, U, DIM=100, m=2):
        data_n = self.batch_size
        cluster_num = 2
        tensor_m = tf.constant(np.ones([data_n, cluster_num], dtype=np.float32) * m)
        UM = tf.pow(U, tensor_m)
        dumpy_sum_num = tf.matmul(UM, C(data), transpose_a=True)
        dum = tf.expand_dims(tf.reduce_sum(UM, 0), 1)
        g = []
        for i in range(DIM):
            g.append(dum)
        dumpy_sum_dum = tf.concat(g, axis=1)
        clusters = tf.divide(dumpy_sum_num, dumpy_sum_dum)
        c1 = []
        c2 = []
        for i in range(data_n):
            c1.append(tf.expand_dims(clusters[0], 0))
            c2.append(tf.expand_dims(clusters[1], 0))
        cluster_1 = tf.concat(c1, axis=0)
        cluster_2 = tf.concat(c2, axis=0)
        distance = tf.reduce_mean(tf.sqrt(tf.pow(tf.subtract(cluster_1, cluster_2), 2)))
        return distance

    def fuzzy_loss(self, C, x, U, clusters):
        data = C(x)
        data_n = data.get_shape()[0]
        c1 = []
        c2 = []
        for i in range(data_n):
            c1.append(tf.expand_dims(clusters[0], 0))
            if clusters.get_shape()[0] == 2:
                c2.append(tf.expand_dims(clusters[1], 0))
        cluster_1 = tf.concat(c1, axis=0)
        if clusters.get_shape()[0] == 2:
            cluster_2 = tf.concat(c2, axis=0)

        distance_1 = tf.reduce_mean(tf.sqrt(tf.pow(tf.subtract(data, cluster_1), 2)), axis=1)
        if clusters.get_shape()[0] == 2:
            distance_2 = tf.reduce_mean(tf.sqrt(tf.pow(tf.subtract(data, cluster_2), 2)), axis=1)
            distance = tf.concat([tf.expand_dims(distance_1, 1), tf.expand_dims(distance_2, 1)], axis=1)
        else:
            distance = distance_1
        fuzzyLoss = tf.reduce_mean(tf.multiply(distance, U))

        return fuzzyLoss,data
