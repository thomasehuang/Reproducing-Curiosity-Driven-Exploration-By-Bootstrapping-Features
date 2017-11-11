import tensorflow as tf
import gym
import numpy as np
from models import conv2d, linear, normalized_columns_initializer, flatten
from utils import get_placeholder

class CnnEmbedding(object):
    def __init__(self, name, ob_space, ac_space):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space):
        assert isinstance(ob_space, gym.spaces.Box)

        sequence_length = None

        self.input = get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        x = self.input / 255.0
        x = tf.nn.relu(conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
        x = tf.nn.relu(conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
        x = tf.nn.relu(conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
        x = flatten(x)
        self.x = tf.nn.relu(linear(x, 512, 'lin', normalized_columns_initializer(1.0)))

    def embed(self, state):
        sess = tf.get_default_session()
        return sess.run([self.x], {self.input: np.expand_dims(state, axis=0)})
    def get_input_and_last_layer(self):
        return [self.input, self.x]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

