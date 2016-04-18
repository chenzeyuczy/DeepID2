#! /usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
from PIL import Image

class Container():

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

	def __iter__(self):
		return self

	def next(self):
		data_list = []
		labels = []
		for i in xrange(self.__batch_size):
			img_data = None
			while img_data == None:
				idx = self.getNextIndex()
				(img_path, label) = self.__info[idx]
				img_data = self.loadImage(img_path)
			data_list.append(img_data)
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
		try:
			img = Image.open(img_path)
			img = np.array(img, dtype = np.float32)
		except:
			print "Error occur"
			return None
		if size:
			img = img.resize(size)
		# Extend image dimension in case of gray mode.
		if len(img.shape) == 2:
			w, h = img.shape
			img_tem = np.empty((w, h, 3))
			img_tem[:,:,0] = img
			img_tem[:,:,1] = img
			img_tem[:,:,2] = img
			img = img_tem
		img = img.transpose(2, 0, 1)
		return img

	def convertLabel(self, labels):
		label_num = len(labels)
		label_info = np.array(labels)
		label_info = label_info.reshape(label_num, 1)
		return label_info

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

	container = Container(data_file)
	container.setBatchSize(100)
	batch = container.next()

