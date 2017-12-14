#!/bin/sh python3

import baselines.common.tf_util as U
import tensorflow as tf
import gym
from models import linear, normalized_columns_initializer
from utils import get_placeholder
import baselines.common.tf_util as U
from baselines.common.distributions import make_pdtype

from embedding import CnnEmbedding

class Policy(object):
    recurrent = False
    def __init__(self, name, ac_space, joint_training, emb_size=None, emb_network=None):
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            self._init(ac_space, joint_training, emb_size, emb_network)

    def _init(self, ac_space, joint_training, emb_size=None, emb_network=None):
        self.pdtype = pdtype = make_pdtype(ac_space)

        self.emb_network = emb_network
        self.joint_training = joint_training
        size = 128
        if self.joint_training:
            self.input, output = emb_network.get_input_and_last_layer()
            x = tf.nn.relu(linear(output, size, 'lin1', normalized_columns_initializer(1.0)))
        else:
            self.input = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, emb_size])
            x = tf.nn.relu(linear(self.input, size, 'lin1', normalized_columns_initializer(1.0)))

        # x = tf.nn.relu(linear(x, 32, 'lin2', normalized_columns_initializer(1.0)))
        logits = linear(x, pdtype.param_shape()[0], "logits", normalized_columns_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.ac = self.pd.sample()
        # self.probs = tf.nn.softmax(logits, dim=-1)[0, :]
        self.vpred = linear(x, 1, "value", normalized_columns_initializer(1.0))

        self._act = U.function([self.input], [self.ac, self.vpred])

    def act(self, ob):
        return self._act(ob)
    def get_variables(self):
        if self.joint_training:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope) + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.emb_network.scope)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        if self.joint_training:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope) + \
                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.emb_network.scope)
        else:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
