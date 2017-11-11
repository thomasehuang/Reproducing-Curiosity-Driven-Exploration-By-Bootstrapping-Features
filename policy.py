#!/bin/sh python3

import tensorflow as tf
import gym
from models import linear, normalized_columns_initializer
from utils import get_placeholder

class Policy(object):
    recurrent = False
    def __init__(self, name, ac_space):
        with tf.variable_scope(name):
            self._init(ac_space)
            self.scope = tf.get_variable_scope().name

    def _init(self, ac_space):

        sequence_length = None

        self.input = get_placeholder(name="emb", dtype=tf.float32, shape=[sequence_length, 512])

        x = tf.nn.relu(linear(self.input, 128, 'lin1', normalized_columns_initializer(1.0)))
        x = tf.nn.relu(linear(x, 32, 'lin2', normalized_columns_initializer(1.0)))

        logits = linear(x, ac_space.n, "logits", normalized_columns_initializer(0.01))
        self.probs = tf.nn.softmax(logits, dim=-1)[0, :]
        self.vpred = linear(x, 1, "value", normalized_columns_initializer(1.0))

    def act(self, embedding):
        sess = tf.get_default_session()
        return sess.run([self.probs, self.vpred], {self.input: embedding})
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

