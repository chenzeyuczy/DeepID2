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

		top[0].reshape(1, 3, 150, 150)
		top[1].reshape(1, 1)
		pass

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		(data, label) = self.__data.next()

		(N, C, W, H) = data.shape
		(N, K) = label.shape

		top[0].reshape(N, C, W, H)
		top[0].data[...] = data

		top[1].reshape(N, K)
		top[1].data[...] = label
		pass

	def backward(self, top, propagate_down, bottom):
		pass

