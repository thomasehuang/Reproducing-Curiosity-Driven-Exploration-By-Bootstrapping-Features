import tensorflow as tf
import gym
import numpy as np
import baselines.common.tf_util as U
from models import conv2d, linear, normalized_columns_initializer, flatten
from utils import get_placeholder

class CnnEmbedding(object):
    def __init__(self, name, ob_space, ac_space, embedding_space_size, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            self._init(ob_space, ac_space, embedding_space_size)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, embedding_space_size):
        assert isinstance(ob_space, gym.spaces.Box)

        sequence_length = None

        # self.input = tf.placeholder(dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        self.input = U.get_placeholder(name="ob_f", dtype=tf.float32, shape=[None] + list(ob_space.shape))
        self.embedding_space = embedding_space_size

        x = self.input / 255.0
        x = tf.nn.relu(conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
        x = tf.nn.relu(conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
        x = tf.nn.relu(conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
        x = flatten(x)
        self.x = tf.nn.relu(linear(x, self.embedding_space, 'lin', normalized_columns_initializer(1.0)))

    def embed(self, state):
        sess = tf.get_default_session()
        return sess.run(self.x, {self.input: state})
    def get_input_and_last_layer(self):
        return [self.input, self.x]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_embedding_space(self):
        return self.embedding_space

