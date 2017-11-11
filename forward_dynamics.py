#!/bin/sh python3

import tensorflow as tf
from models import linear, normalized_columns_initializer
from utils import get_placeholder
from embedding import CnnEmbedding

class ForwardDynamics(object):
    def __init__(self, name, ob_space, ac_space):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space):

        input_shape = [None] + list(ob_space.shape)
        self.s1, phi1 = CnnEmbedding("phi1", ob_space, ac_space).get_input_and_last_layer()
        self.s2, phi2 = CnnEmbedding("phi2", ob_space, ac_space).get_input_and_last_layer()
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space.n])

        # forward model: f(phi1,asample) -> phi2
        # Note: no backprop to asample of policy: it is treated as fixed for predictor training
        size = 256
        f = tf.concat([phi1, asample], 1)
        f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
        f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))
        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
        self.forwardloss = self.forwardloss * 512.0  # lenFeatures=288. Factored out to make hyperparams not depend on it.

    def get_loss(self, s1, s2, asample):
        sess = tf.get_default_session()
        error = sess.run(self.forwardloss,
                         {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        #error = error * constants['PREDICTION_BETA']
        return error
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
