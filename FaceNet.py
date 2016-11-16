from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops

def conv2d(
		input_layer,
		input_depth,
		output_depth,
		kernel_size,
		stride,
		padding_type,
		name
		):
	with tf.variable_scope(name):
		kernel = tf.get_variable("weights",
								 [kernel_size[0], 
								  kernel_size[1], 
								  input_depth, 
								  output_depth],
								  initializer=tf.truncated_normal_initializer(stddev=1e-1))
		conv = tf.nn.conv2d(input_layer, 
							kernel, 
							[1, stride[0], stride[1], 1], 
							padding=padding_type)
		biases = tf.get_variable("biases", [output_depth], initializer=tf.constant_initializer(), dtype = input_layer.dtype)
		bias = tf.nn.bias_add(conv, biases)
		activation = tf.nn.relu(bias)
	return activation

def max_pool(
		input_layer,
		kernel,
		stride,
		padding_type,
		name
		):
	with tf.variable_scope(name):
		maxpool = tf.nn.max_pool(
							input_layer,
							ksize=[1, kernel[0], kernel[1], 1],
							strides=[1, stride[0], stride[1], 1],
							padding=padding_type)
	return maxpool

def average_pool(
		input_layer,
		kernel,
		stride,
		padding_type,
		name
		):
	with tf.variable_scope(name):
		averagepool = tf.nn.avg_pool(
							input_layer,
							ksize=[1, kernel[0], kernel[1], 1],
							strides=[1, stride[0], stride[1], 1],
							padding=padding_type)
	return averagepool

def l2_pool(
		input_layer,
		kernel,
		stride,
		padding_type,
		name):
	with tf.variable_scope(name):
		power = tf.square(input_layer)
		subsample = tf.nn.avg_pool(
							power,
							ksize=[1, kernel[0], kernel[1], 1],
							strides=[1, stride[0], stride[1], 1],
							padding=padding_type)
		#subsample_sum = tf.matmul(subsample, float(kernel[0] * kernel[1]))
		output = tf.sqrt(subsample)
	return output

def inception(
		input_layer,
		input_depth,
		kernel_stride,
		op1_size,
		op2_reduce,
		op2_output,
		op3_reduce,
		op3_output,
		pool_kernel,
		pool_size,
		pool_stride,
		pool_type,
		name
		):
	# Info
	print("Name=", name)
	print("Input Depth=", input_depth)
	print("Kernel Sizes= {3, 5}")
	print("Kernel Stride= {%d, %d}" % (kernel_stride, kernel_stride))
	print("Output Size= {%d, %d}" % (op2_output, op3_output))
	print("Reduce Size= {%d, %d, %d, %d}" % (op2_reduce, op3_reduce, pool_size, op1_size))
	print("Pooling= {%s, %d, %d, %d, %d}" % (pool_type, 
											 pool_kernel, 
											 pool_kernel, 
											 pool_stride, 
											 pool_stride))
	if pool_size > 0:
		pool_output = pool_size
	else:
		pool_output = input_depth
	print("Output Size =", op1_size+op2_output+op3_output+pool_output)
	print()

	network = []

	with tf.variable_scope(name):
		with tf.variable_scope("1x1"):
			if op1_size > 0:
				conv1 = conv2d(
							input_layer, 
							input_depth, 
							op1_size, 
							[1, 1], 
							[1, 1], 
							'SAME', 
							'conv1x1')
				network.append(conv1)
		with tf.variable_scope("3x3"):
			if op2_reduce > 0:
				conv3a = conv2d(
							input_layer,
							input_depth,
							op2_reduce,
							[1, 1],
							[1, 1],
							'SAME',
							'conv1x1')
				conv3 = conv2d(
							conv3a,
							op2_reduce,
							op2_output,
							[3, 3],
							[kernel_stride, kernel_stride],
							'SAME',
							'conv3x3')
				network.append(conv3)
		with tf.variable_scope("5x5"):
			if op3_reduce > 0:
				conv5a = conv2d(
							input_layer,
							input_depth,
							op3_reduce,
							[1, 1],
							[1, 1],
							'SAME',
							'conv1x1')
				conv5 = conv2d(
							conv5a,
							op3_reduce,
							op3_output,
							[5, 5],
							[kernel_stride, kernel_stride],
							'SAME',
							'conv5x5')
				network.append(conv5)
		with tf.variable_scope("Pooling"):
			if pool_type == "MAX":
				pool = max_pool(
							input_layer, 
							[pool_kernel, pool_kernel], 
							[pool_stride, pool_stride], 
							'SAME', 
							'pool')
			elif pool_type == "L2":
				pool = l2_pool(
							input_layer,
							[pool_kernel, pool_kernel],
							[pool_stride, pool_stride],
							'SAME',
							'pool')

			if pool_size > 0:
				pool_conv = conv2d(
								pool,
								input_depth,
								pool_size,
								[1, 1],
								[1, 1],
								'SAME',
								'conv1x1',
								)
			else:
				pool_conv = pool
			network.append(pool_conv)
		incept = array_ops.concat(3, network, name=name)
	return incept

def inference(images, keep_prob):
	endpoints = {}
	net = conv2d(images, 3, 64, [7, 7], [2, 2], 'SAME', 'conv1_7x7')
	endpoints['conv1'] = net
	net = max_pool(net, [3, 3], [2, 2], 'SAME', 'pool1')
	endpoints['pool1'] = net
	# Normilize

	# "Inception" 2 only includes the 1x2 convolution and 3x3 convolution
	# that are part of a normal inception. Instead of using the inception
	# fuction I just did it here with convolutions 
	net = conv2d(net, 64, 64, [1, 1], [1, 1], 'SAME', 'conv2_1x1')
	endpoints['conv2_1x1'] = net
	net = conv2d(net, 64, 192, [3, 3], [1, 1], 'SAME', 'conv3_3v3')
	endpoints['conv3_3x3'] = net
	# Normalize
	net = max_pool(net, [3, 3], [2, 2], 'SAME', 'pool2')
	endpoints['pool2'] = net

	# Inception 3
	net = inception(net, 192, 1, 64, 96, 128, 16, 32, 3, 32, 1, 'MAX', 'inception3a')
	endpoints['inception3a'] = net
	net = inception(net, 256, 1, 64, 96, 128, 32, 64, 3, 64, 1, 'L2', 'inception3b')
	endpoints['inception3b'] = net
	net = inception(net, 320, 2, 0, 128, 256, 32, 64, 3, 0, 2, 'MAX', 'inception3c')
	endpoints['inception3c'] = net

	# Inception 4
	net = inception(net, 640, 1, 256, 96, 192, 32, 64, 3, 128, 1, 'L2', 'inception4a')
	endpoints['inception4a'] = net
	net = inception(net, 640, 1, 224, 112, 224, 32, 64, 3, 128, 1, 'L2', 'inception4b')
	endpoints['inception4b'] = net
	net = inception(net, 640, 1, 192, 128, 256, 32, 64, 3, 128, 1, 'L2', 'inception4c')
	endpoints['inception4c'] = net
	net = inception(net, 640, 1, 160, 144, 288, 32, 64, 3, 128, 1, 'L2', 'inception4d')
	endpoints['inception4d'] = net
	net = inception(net, 640, 2, 0, 160, 256, 64, 128, 3, 0, 2, 'MAX', 'inception4e')
	endpoints['inception4e'] = net

	# Inception 5
	net = inception(net, 1024, 1, 384, 192, 384, 48, 128, 3, 128, 1, 'L2', 'inception5a')
	endpoints['inception5a'] = net
	net = inception(net, 1024, 1, 384, 192, 384, 48, 128, 3, 128, 1,'MAX', 'inception5b')
	endpoints['inception5b'] = net

	net = average_pool(net, [7,7], [1,1], 'VALID', 'pool3')
	endpoints['pool3'] = net
	net = tf.reshape(net, [-1, 1024])
	net = tf.nn.dropout(net, keep_prob)
	endpoints['dropout1'] = net
	weights = tf.get_variable("FC_Weights", [1024, 128], initializer=tf.truncated_normal_initializer(stddev=1e-1))
	net = tf.matmul(net, weights)
	biases = tf.get_variable("FC_Biases", [128], initializer=tf.constant_initializer())
	logits = tf.nn.bias_add(net, biases)
	endpoints['logits'] = logits

	return logits, endpoints


def triplet_loss(anchor, positive, negative, alpha):
	with tf.variable_scope('triplet_loss'):
		pos_distance = tf.reduce_sum(tf.square(tf.sub(anchor, positive)), 1)
		neg_distance = tf.reduce_sum(tf.square(tf.sub(anchor, negative)), 1)
		basic_loss = tf.add(tf.sub(pos_distance, neg_distance), alpha)
		loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

if __name__ == '__main__':
	images = tf.zeros([2, 224, 224, 3])
	logits, endpoints = inference(images, 0.5)
	a = tf.zeros([1, 128])
	p = tf.zeros([1, 128])
	n = tf.zeros([1, 128]) + 1
	loss = triplet_loss(a, p, n, 0.2)