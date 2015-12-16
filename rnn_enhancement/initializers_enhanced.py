"""Library of Initializers"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

print('Initializer Library -- Will add to this if necessary.')


def identity_initializer(scale = 1.0):

    print('Warning -- You have opted to use the identity_initializer for your identity matrix!')
    def _initializer(shape, dtype=tf.float32):
        if len(shape) == 1:
            return tf.constant(0., dtype=dtype, shape=shape)
        elif len(shape) == 2 and shape[0] == shape[1]:
            return tf.constant(scale*np.identity(shape[0], dtype))
        elif len(shape) == 4 and shape[2] == shape[3]:
            array = np.zeros(shape, dtype=float)
            cx, cy = shape[0]/2, shape[1]/2
            for i in range(shape[2]):
                array[cx, cy, i, i] = 1
            return tf.constant(scale*array, dtype=dtype)
        else:
            raise 
    return _initializer




def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    print('Warning -- You have opted to use the orthogonal_initializer function')
    def _initializer(shape, dtype=tf.float32):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      print('you have initialized one orthogonal matrix.')
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer