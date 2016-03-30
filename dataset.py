#! /usr/bin/python
#-*- coding:utf-8 -*-

import os, random
from PIL import Image
import numpy as np

class Dataset:

	def __init__(self, data_root = '.'):
		# Remove separator at the end of data_root.
		data_root = data_root.rstrip(os.sep)
		self.__data_root = data_root
		self.loadInfo()
		self.__train_set = []
		self.__test_set = []
		# Config type of samples, 0 for thw whole, 1 for train and 2 for test.
		self.__partition = 0
		self.loadData()
	
	def __len__(self):
		return len(self.__data)

	def __getitem__(self, key):
		return self.__data[key]

	# Load info from file system.
	def loadInfo(self):
		self.__info = []
		self.__data = []
		if self.__data_root:
			folders = sorted(os.listdir(self.__data_root))
			for folder in folders:
				folder_path = os.sep.join([self.__data_root, folder])
				files = sorted(os.listdir(folder_path))
				data_src = []
				for f in files:
					file_path = os.sep.join([self.__data_root, folder, f])
					data_src.append(file_path)
				self.__info.append((folder, data_src))
		return
	
	def split(self, ratio = 0.7):
		split_num = int(len(self) * ratio)
		class_num = len(self.__info)
		permutation = np.random.permutation(class_num)
		self.__train_set = permutation[:split_num]
		self.__test_set = permutation[split_num:]
		print "Split into two parts with " + str(split_num) + " in train set and " + str(len(self) - split_num) + " in test set."

	def setPartition(self, partition):
		self.__partition = partition
		self.loadData()

	def loadData(self):
		if self.__partition == 1:
			select_seq = self.__train_set
		elif self.__partition == 2:
			select_seq = self.__test_set
		else:
			select_seq = range(len(self.__info))
		self.__data = []
		for label in select_seq:
			data_list = self.__info[label][1]
			for item in data_list:
				self.__data.append((item, label))
		return

	def getClassNum(self):
		return len(self.__info)

	def show(self):
		if self.__partition == 0:
			class_num = len(self.__info)
		elif self.__partition == 1:
			class_num = len(self.__train_set)
		else:
			class_num = len(self.__test_set)
		id_num = len(self.__data)
		print "Total sample: " + str(id_num) + " identities with " + str(class_num) + " classes."

if __name__ == '__main__':
	data_root = '/home/chenzeyu/dataset/lfw-deepfunneled'
	dataset = Dataset(data_root)
	dataset.split(0.01)
	print str(len(dataset)) + " identities found."

#	train_data, train_label = dataset.getData(1)
#	print str(len(train_data)) + " loaded, with " + str(max(train_label) + 1) + " classes."
	dataset.setPartition(1)
	dataset.show()

