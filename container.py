#! /usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
from PIL import Image
from dataset import Dataset

class Container():

	def __init__(self, dataset):
		self.__batch_size = 10
		self.__dataset = dataset
		self.__num_sample = len(dataset)
		self.__per_idx = None
		self.__cur_idx = 0
		pass

	def setBatchSize(self, batch_size):
		self.__batch_size = batch_size
		self.__cur_idx = 0

	def __iter__(self):
		return self

	def next(self):
		data_list = []
		labels = []
		for i in xrange(self.__batch_size):
			idx = self.getNextIndex()
			(img_path, label) = self.__dataset[idx]
			data_list.append(self.loadImage(img_path))
			labels.append(label)
		data_info = np.array(data_list)
		label_info = self.convertLabel(labels)
		return (data_info, label_info)

	def getNextIndex(self):
		# Check whether it comes to the begining.
		if self.__cur_idx == 0:
			self.__per_idx = np.random.permutation(self.__num_sample)
		idx = self.__per_idx[self.__cur_idx]
		self.__cur_idx = (self.__cur_idx + 1) % self.__num_sample
		return idx

	def loadImage(self, img_path, size = None):
		img = Image.open(img_path)
		if size:
			img = img.resize(size)
		img = np.array(img, dtype = np.float64)
		img = img.transpose(2, 0, 1)
		return img

	def convertLabel(self, labels):
		label_num = len(labels)
		class_num = self.__dataset.getClassNum()
		#label_info = np.zeros((label_num, class_num), dtype = np.float32)
		#for i in xrange(label_num):
		#	label = labels[i]
		#	label_info[i, label] = 1
		label_info = np.array(labels)
		label_info = label_info.reshape(label_num, 1)
		return label_info

if __name__ == '__main__':
	dataset_path = '/home/chenzeyu/dataset/lfw-deepfunneled'
	dataset = Dataset(dataset_path)
	split_ratio = 0.1
	dataset.split(split_ratio)
	dataset.setPartition(1)

	container = Container(dataset)
	container.setBatchSize(100)
	batch = container.next()

