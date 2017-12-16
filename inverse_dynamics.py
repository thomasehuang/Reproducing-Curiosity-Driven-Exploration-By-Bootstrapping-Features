import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from mpi4py import MPI
import tensorflow as tf
import gym
from models import linear, normalized_columns_initializer
from utils import get_placeholder
import baselines.common.tf_util as U
from baselines.common.distributions import make_pdtype

from embedding import CnnEmbedding

class InverseDynamics(object):
    def __init__(self, name, ob_space, ac_space, emb_size=None, emb_network=None):
        self.emb2 = CnnEmbedding("embedding", ob_space, ac_space, emb_size, reuse=True)
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            self._init(ac_space, emb_size, emb_network)

    def _init(self, ac_space, emb_size=None, emb_network=None):
        self.emb_network = emb_network
        
        self.s1, self.phi1 = self.emb_network.get_input_and_last_layer()
        self.s2, self.phi2 = self.emb2.get_input_and_last_layer()
        # self.phi1 = tf.placeholder(dtype=tf.float32, shape=[None, emb_size], name='phi1')
        # self.phi2 = tf.placeholder(dtype=tf.float32, shape=[None, emb_size], name='phi2')

        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space.n])

        size = 256
        # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
        g = tf.concat([self.phi1, self.phi2], 1)
        g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))
        aindex = tf.argmax(self.asample, axis=1)  # aindex: [batch_size,]
        logits = linear(g, ac_space.n, "glast", normalized_columns_initializer(0.01))
        self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits=logits, labels=aindex), name="invloss")
        self.ainvprobs = tf.nn.softmax(logits, dim=-1)

        var_list = self.get_trainable_variables()
        self.lossandgrad = U.function([self.s1, self.s2, self.asample], [self.invloss] + [U.flatgrad(self.invloss, var_list)])
        self.adam = MpiAdam(var_list, epsilon=1e-5)
        
        U.initialize()
        self.adam.sync()

    def act(self, ob):
        return self._act(ob)
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope) + \
               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.emb_network.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope) + \
               tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.emb_network.scope)
    def train(self, s1, s2, asample, learning_rate):
        *newlosses, g = self.lossandgrad(s1, s2, asample)
        self.adam.update(g, learning_rate)
        # print('Inverse Dynamics loss')
        # print(newlosses)
        # print(g)
        