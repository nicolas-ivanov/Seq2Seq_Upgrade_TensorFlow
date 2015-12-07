"""Library of Linear Algebraic Functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  assert args
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"): #in this linear scope, the library that you're retriving is Linear
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size]) #i think this is retrieving the weight matrix
    if len(args) == 1:
      res = tf.matmul(args[0], matrix) #this is just one matrix to multiply by 
    else:
      res = tf.matmul(tf.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable("Bias", [output_size],
                                initializer=tf.constant_initializer(bias_start)) #this is retrieving the bias term that you would use
  return res + bias_term


  '''Nick, the matrix variable tf I think is your weight matrix'''

def euclidean_norm(tensor, reduction_indicies = None, name = None):
	with tf.op_scope(tensor + reduction_indicies, name, "euclidean_norm"): #need to have this for tf to work
		squareroot_tensor = tf.square(tensor)
		euclidean_norm = tf.sum(squareroot_tensor, reduction_indicies =  reduction_indicies)
		return euclidean_norm

def frobenius_norm(tensor, reduction_indicies = None, name = None):
	with tf.op_scope(tensor + reduction_indicies, name, "frobenius_norm"): #need to have this for tf to work
		squareroot_tensor = tf.square(tensor)
		tensor_sum = tf.sum(squareroot_tensor, reduction_indicies =  reduction_indicies)
		frobenius_norm = tf.sqrt(tensor_sum)
		return frobenius_norm

