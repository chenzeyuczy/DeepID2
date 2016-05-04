#! /usr/bin/python
#-*- coding:utf-8 -*-

import caffe
import numpy as np, json
from container import Container

class SplitLayer(caffe.Layer):

	def setup(self, bottom, top):
		data, pair_info = labels bottom[0].data, bottom[1].data
		(N, C) = data.shape
		top[0].reshape(N, C)
		top[1].reshape(N, C)
		top[2].reshape(N, 1)
		pass

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		data, pair_info = labels bottom[0].data, bottom[1].data

		# Declare variables.
		(N, C) = data.shape
		pair_num = len(pair_info)
		data1 = np.empty((pair_num, C), dtype = np.float32)
		data2 = np.empty((pair_num, C), dtype = np.float32)
		sim = pair_info[:, 2]

		for idx in xrange(pair_num):
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

		data, pair_info = labels bottom[0].data, bottom[1].data

		# Declare variables.
		(N, C) = data.shape
		pair_num = len(pair_info)
		
		diff = np.zeros((N, C), dtype = np.float32)

		for idx = xrange(pair_num):
			idx1, idx2 = pair_info[idx, :2]
			diff[idx1] += self.diff[idx]
			diff[idx2] += self.diff[idx]

		bottom[0].diff[...] = diff
		bottom[1].diff[...] = 0
		pass

