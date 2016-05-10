#! /usr/bin/python
#-*- coding:utf-8 -*-

import caffe
import numpy as np, json
from container import Container

class DataLayer(caffe.Layer):

	def setup(self, bottom, top):
		# Parse param_str from prototxt.
		param = json.loads(self.param_str)
		data_file = param["data_file"]
		batch_size = param["batch_size"]

		# Custom parameters.
		container = Container(data_file)
		container.setBatchSize(batch_size)
		self.__data = container

		top[0].reshape(batch_size, 3, 150, 150)
		top[1].reshape(batch_size, 1)
		top[2].reshape(batch_size, 3)
		pass

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		(data, label) = self.__data.next()

		(N, C, W, H) = data.shape
		(N, K) = label.shape

		pair_num = N / 2
		pair_info = np.empty((pair_num, 3), dtype = np.float32)
		idx_pair = np.arange(N).reshape(pair_num, 2)
		pair_info[:, :2] = idx_pair
		for i in xrange(pair_num):
			pair_info[i, 2] = label[idx_pair[i, 0]] == label[idx_pair[i, 1]]

		top[0].reshape(N, C, W, H)
		top[0].data[...] = data

		top[1].reshape(N, K)
		top[1].data[...] = label

		top[2].reshape(pair_num, 3)
		top[2].data[...] = pair_info
		pass

	def backward(self, top, propagate_down, bottom):
		pass

