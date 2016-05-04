#! /usr/bin/python
#-*- coding:utf-8 -*-

import caffe
import numpy as np, json
from container import Container

class SplitLayer(caffe.Layer):

	def setup(self, bottom, top):
		data, labels = bottom
		sample_num = len(labels)
		pass

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		data, labels = bottom

		(N, C, W, H) = data.shape
		(N, K) = label.shape

		for idx1 in xrange(sample_num):
			data1[idx1] = data[idx1]
			idx2 = np.random.randInt(sample_num)
			data2[idx1] = data[idx2]
			label_out[idx1] = 1 if labels[idx1] == labels[idx2] else 0

		top[0].reshape(N, C, W, H)
		top[0].data[...] = data1

		top[1].reshape(N, C, W, H)
		top[1].data[...] = data2

		top[2].reshape(N, K)
		top[2].data[...] = label_out
		pass

	def backward(self, top, propagate_down, bottom):
		pass

