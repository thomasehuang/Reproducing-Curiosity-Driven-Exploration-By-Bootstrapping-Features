#!/bin/sh python3

import tensorflow as tf
import gym
from models import linear, normalized_columns_initializer
from utils import get_placeholder
import baselines.common.tf_util as U
from baselines.common.distributions import make_pdtype

from embedding import CnnEmbedding

class Policy(object):
    recurrent = False
    def __init__(self, name, ac_space, is_backprop_to_embedding, emb_space=None, emb=None):
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            self._init(ac_space, is_backprop_to_embedding, emb_space, emb)

    def _init(self, ac_space, is_backprop_to_embedding, emb_space=None, emb=None):
        self.pdtype = pdtype = make_pdtype(ac_space)

        # self.input = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length, 512])
        if is_backprop_to_embedding:
            self.input, output = emb.get_input_and_last_layer()
            x = tf.nn.relu(linear(output, 128, 'lin1', normalized_columns_initializer(1.0)))
        else:
            # self.input = tf.placeholder(dtype=tf.float32, shape=[None, emb_space], name='input')
            self.input = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, emb_space])
            x = tf.nn.relu(linear(self.input, 128, 'lin1', normalized_columns_initializer(1.0)))

        x = tf.nn.relu(linear(x, 32, 'lin2', normalized_columns_initializer(1.0)))
        logits = linear(x, ac_space.n, "logits", normalized_columns_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.ac = self.pd.sample()
        # self.probs = tf.nn.softmax(logits, dim=-1)[0, :]
        self.vpred = linear(x, 1, "value", normalized_columns_initializer(1.0))

    def act(self, input):
        sess = tf.get_default_session()
        return sess.run([self.ac, self.vpred], {self.input: input})
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class CnnPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, kind='large'):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
    
        x = ob / 255.0
        if kind == 'small': # from A3C paper
            x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 256, 'lin', U.normc_initializer(1.0)))
        elif kind == 'large': # Nature DQN
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 512, 'lin', U.normc_initializer(1.0)))
        else:
            raise NotImplementedError

        logits = U.dense(x, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = U.dense(x, 1, "value", U.normc_initializer(1.0))[:,0]

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample() # XXX
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []


