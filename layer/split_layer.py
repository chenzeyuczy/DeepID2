#! /usr/bin/python
#-*- coding:utf-8 -*-

import caffe
import numpy as np

class SplitLayer(caffe.Layer):

	def setup(self, bottom, top):
		data, labels = bottom[0].data, bottom[1].data
		(batch_size, C) = data.shape
		N = batch_size / 2
		top[0].reshape(N, C)
		top[1].reshape(N, C)
		top[2].reshape(N, 1)
		pass

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		data, labels = bottom[0].data, bottom[1].data

		# Declare variables.
		(batch_size, C) = data.shape
		N = batch_size / 2
		data1 = np.empty((N, C), dtype = np.float32)
		data2 = np.empty((N, C), dtype = np.float32)
		# Get similarity of each pair.
		sim = labels[::2] == labels[1::2]

		data1 = data[::2]
		data2 = data[1::2]

		top[0].reshape(N, C)
		top[0].data[...] = data1

		top[1].reshape(N, C)
		top[1].data[...] = data2

		top[2].reshape(N, 1)
		top[2].data[...] = sim
		pass

	def backward(self, top, propagate_down, bottom):
		if not propagate_down[0]:
			return

		data = bottom[0].data

		# Declare variables.
		(batch_size, C) = data.shape
		N = batch_size ／ 2
		
		diff = np.zeros((batch_size, C), dtype = np.float32)

		for idx in xrange(N):
			diff[idx * 2] = top[0].diff[idx]
			diff[idx * 2 + 1] = top[1].diff[idx]

		bottom[0].diff[...] = diff
		pass

