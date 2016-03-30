#! /usr/bin/python
#-*- coding:utf-8 -*-

import caffe
import numpy as np

class DataLayer(caffe.Layer):

	def setData(self, dataset):
		self.__data = dataset

	def setup(self, bottom, top):
		top[0].reshape(1, 3, 255, 255)
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

