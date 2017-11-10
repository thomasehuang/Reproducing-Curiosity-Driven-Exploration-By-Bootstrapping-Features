#!/bin/sh python3
import tensorflow as tf  # pylint: ignore-module

_PLACEHOLDER_CACHE = {}  # name -> (placeholder, dtype, shape)
def get_placeholder(name, dtype, shape):
    if name in _PLACEHOLDER_CACHE:
        out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
        assert dtype1 == dtype and shape1 == shape
        return out
    else:
        out = tf.placeholder(dtype=dtype, shape=shape, name=name)
        _PLACEHOLDER_CACHE[name] = (out, dtype, shape)
        return out

def get_placeholder_cached(name):
    return _PLACEHOLDER_CACHE[name][0]

def reset():
    global _PLACEHOLDER_CACHE
    _PLACEHOLDER_CACHE = {}
    tf.reset_default_graph()
