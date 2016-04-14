#! /usr/bin/python
#-*- coding:utf-8 -*-

import caffe
import numpy as np, json
from container import Container
from dataset import Dataset

class DataLayer(caffe.Layer):

	def setup(self, bottom, top):
		# Parse param_str from prototxt.
		param = json.loads(self.param_str)
		dataset_path = param["dataset"]
		split_ratio = param["split_ratio"]
		selected_partition = param["selected_partition"]
		batch_size = param["batch_size"]

		# Custom parameters.
		dataset = Dataset(dataset_path)
		dataset.split(split_ratio)
		dataset.setPartition(selected_partition)
		container = Container(dataset)
		container.setBatchSize(batch_size)
		self.__data = container

		top[0].reshape(1, 3, 250, 250)
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

