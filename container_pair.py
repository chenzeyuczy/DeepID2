#! /usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
from PIL import Image
from container import Container

class ContainerPair(Container):

	def __init__(self, data_file):
		self.parseFile(data_file)
		self.__batch_size = 10
		self.__num_sample = len(self.__info)
		self.__per_idx = None
		self.__cur_idx = 0
		pass

	def setBatchSize(self, batch_size):
		self.__batch_size = batch_size
		self.__cur_idx = 0

	def next(self):
		data_list1 = []
		data_list2 = []
		labels = []
		for i in xrange(self.__batch_size):
			img_data1 = None
			while img_data1 == None:
				idx = np.random.randint(self.__num_sample)
				(img_path1, label1) = self.__info[idx]
				img_data1 = self.loadImage(img_path1)
			data_list1.append(img_data1)
			img_data2 = None
			while img_data2 == None:
				idx = np.random.randint(self.__num_sample)
				(img_path2, label2) = self.__info[idx]
				img_data2 = self.loadImage(img_path2)
			data_list2.append(img_data2)
			labels.append(1 if (label1 == label2) else 0)
		data_info1 = np.array(data_list1)
		data_info2 = np.array(data_list2)
		label_info = self.convertLabel(labels)
		return (data_info1, data_info2, label_info)

	def parseFile(self, datafile):
		# Store info as a format of [src_path, label].
		self.__info = []
		with open(datafile, 'r') as f:
			for line in f:
				item = line.strip().split()
				assert len(item) == 2
				self.__info.append(item)

if __name__ == '__main__':
	data_file = './data/train_set.txt'

	container = ContainerPair(data_file)
	container.setBatchSize(100)
	batch = container.next()
	
	print batch

