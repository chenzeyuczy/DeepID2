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
		for i in xrange(self.__batch_size / 2):
			idx = self.getNextIndex()
			(path1, label1, path2, label2) = self.__info[idx]
			data1 = self.loadImage(path1)
			data2 = self.loadImage(path2)
			data_list.append(data1)
			labels.append(label1)
			data_list.append(data2)
			labels.append(label2)
		# Format transformation.
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
		except:
			print "Error occur when trying to load", img_path
			return None
		# Resize image.
		if size:
			img = img.resize(size)
		else:
			size = img.size
		img = np.array(img, dtype = np.float32)
		# Convert gray mode into RGB mode.
		if len(img.shape) == 2:
			w, h = img.shape
			img_tem = np.empty((w, h, 3))
			img_tem[:,:,0] = img
			img_tem[:,:,1] = img
			img_tem[:,:,2] = img
			img = img_tem

#		patch_size = (100, 100)
#		patch_w, patch_h = patch_size
#		img = img[-patch_w:, 0:patch_h, :]
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
				assert len(item) == 4
				self.__info.append(item)
			f.close()

if __name__ == '__main__':
	data_file = './data/train_set.txt'
	data_file = './preprocess/sample_mix.txt'

	container = Container(data_file)
	container.setBatchSize(100)
	batch = container.next()

	for item in batch:
		print item
		print item.shape

