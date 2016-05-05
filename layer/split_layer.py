#! /usr/bin/python
#-*- coding:utf-8 -*-

import caffe
import numpy as np, json
from container import Container

class SplitLayer(caffe.Layer):

	def setup(self, bottom, top):
		data, pair_info = bottom[0].data, bottom[1].data
		(batch_size, C) = data.shape
		N = len(pair_info)
		top[0].reshape(N, C)
		top[1].reshape(N, C)
		top[2].reshape(N, 1)
		pass

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		data, pair_info = bottom[0].data, bottom[1].data

		# Declare variables.
		(batch_size, C) = data.shape
		N = len(pair_info)
		data1 = np.empty((N, C), dtype = np.float32)
		data2 = np.empty((N, C), dtype = np.float32)
		sim = np.reshape(pair_info[:, 2], (N, 1))

		for idx in xrange(N):
			data1[idx] = data[pair_info[idx, 0]]
			data2[idx] = data[pair_info[idx, 1]]

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

		data, pair_info = bottom[0].data, bottom[1].data

		# Declare variables.
		(batch_size, C) = data.shape
		N = len(pair_info)
		
		diff = np.zeros((batch_size, C), dtype = np.float32)

		for idx in xrange(N):
			idx1, idx2 = pair_info[idx, :2]
			diff[idx1] += top[0].diff[idx]
			diff[idx2] += top[1].diff[idx]

		bottom[0].diff[...] = diff
		bottom[1].diff[...] = 0
		pass

