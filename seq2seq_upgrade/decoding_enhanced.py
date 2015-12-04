
"""Library for decoding functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.python.platform

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


def sample_with_temperature(a, temperature=1.0):
	'''this function takes logits input, and produces a specific number from the array.
	As you increase the temperature, you will get more diversified output but with more errors

	args: 
	Logits -- this must be a 1d array
	Temperature -- how much variance you want in output

	returns:
	Selected number from distribution
	'''

	'''
	Equation can be found here: https://en.wikipedia.org/wiki/Softmax_function (under reinforcement learning)

        Karpathy did it here as well: https://github.com/karpathy/char-rnn/blob/4297a9bf69726823d944ad971555e91204f12ca8/sample.lua
        '''
	
	a = np.squeeze(a)/temperature #start by reduction of temperature
	
	exponent_raised = np.exp(a) #this makes the temperature much more effective and gets rid of negative numbers. 

	probs = exponent_raised / np.sum(exponent_raised) #this will make everything add up to 100% 

	#get rid of any negative numbers in the probabilities -- they shouldn't be in here anyways
	probs = probs.clip(0)

	#reduce the sum for rounding errors
	subtracting_factor = 0.002/probs.shape[0]

	probs = probs - subtracting_factor

	multinomial_part = np.random.multinomial(1, probs, 1)

	final_number = int(np.argmax(multinomial_part))

	return final_number

