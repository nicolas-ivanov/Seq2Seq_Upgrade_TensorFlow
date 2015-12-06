"""Linear Algebraic Functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


def euclidean_norm(tensor, reduction_indicies = None, name = None):
	with tf.op_scope(tensor + reduction_indicies, name, "euclidean_norm"): #need to have this for tf to work
		squareroot_tensor = tf.sqrt(tensor)
		euclidean_norm = tf.sum(squareroot_tensor, reduction_indicies =  reduction_indicies)
		return euclidean_norm

def frobenius_norm(tensor, reduction_indicies = None, name = None):
	with tf.op_scope(tensor + reduction_indicies, name, "frobenius_norm"): #need to have this for tf to work
		squareroot_tensor = tf.sqrt(tensor)
		tensor_sum = tf.sum(squareroot_tensor, reduction_indicies =  reduction_indicies)
		frobenius_norm = tf.sqrt(tensor_sum)
		return frobenius_norm

