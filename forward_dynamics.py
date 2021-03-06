import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from mpi4py import MPI
import tensorflow as tf
from models import linear, normalized_columns_initializer
from utils import get_placeholder
from embedding import CnnEmbedding

class ForwardDynamics(object):
    def __init__(self, name, emb_size, ac_space):
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            self._init(emb_size, ac_space)

    def _init(self, emb_size, ac_space):

        self.phi1 = tf.placeholder(dtype=tf.float32, shape=[None, emb_size], name='phi1')
        self.phi2 = tf.placeholder(dtype=tf.float32, shape=[None, emb_size], name='phi2')

        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space.n])
        # self.learning_rate = tf.placeholder(tf.float32, ())

        size = 256
        # forward model: f(phi1,asample) -> phi2
        # Note: no backprop to asample of policy: it is treated as fixed for predictor training
        f = tf.concat([self.phi1, asample], 1)
        f1 = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
        f2 = linear(f1, self.phi1.get_shape()[1].value, "f2", normalized_columns_initializer(0.01))
        self.forwardloss = 0.5 * tf.reduce_sum(tf.square(tf.subtract(f2, self.phi2)), name='forwardloss')
        self.forwardloss = self.forwardloss / 288  # lenFeatures=288. Factored out to make hyperparams not depend on it.

        # self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.forwardloss, var_list=self.get_trainable_variables())

        var_list = self.get_trainable_variables()
        self.lossandgrad = U.function([self.phi1, self.phi2, self.asample], [self.forwardloss] + [U.flatgrad(self.forwardloss, var_list)])
        self.adam = MpiAdam(var_list, epsilon=1e-5)
        
        U.initialize()
        self.adam.sync()

    def get_loss(self, phi1, phi2, asample):
        sess = tf.get_default_session()
        error = sess.run(self.forwardloss,
                         {self.phi1: phi1, self.phi2: phi2, self.asample: asample})
        return error
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def train(self, phi1, phi2, asample, learning_rate):
        *newlosses, g = self.lossandgrad(phi1, phi2, asample)
        self.adam.update(g, learning_rate)
        # print('Forward Dynamics loss')
        # print(newlosses)
        # print(g)
    # def train(self, phi1, phi2, asample, learning_rate):
    #     sess = tf.get_default_session()
    #     sess.run(self.train_step, {self.phi1: phi1,
    #                                self.phi2: phi2,
    #                                self.asample: asample,
    #                                self.learning_rate: learning_rate})
