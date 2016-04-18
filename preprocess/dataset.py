#! /usr/bin/python
#-*- coding:utf-8 -*-

import os, numpy as np

class Dataset:
	def __init__(self, data_root):
		self.__data_root = data_root
		self.loadInfo()
		pass

	def __len__(self):
		return len(self.__info)

	def loadInfo(self):
		self.__info = []
		if self.__data_root:
			# List folders in root directory.
			folders = sorted(os.listdir(self.__data_root))
			for folder in folders:
				folder_path = os.sep.join([self.__data_root, folder])
				files = sorted(os.listdir(folder_path))
				# Add each file into list.
				data_src = []
				for f in files:
					file_path = os.sep.join([self.__data_root, folder, f])
					data_src.append(file_path)
				self.__info.append((folder, data_src))
		return

	def split(self, ratio):
		split_num = int(len(self) * ratio)
		class_num = len(self.__info)
		permutation = np.random.permutation(class_num)
		self.__train_set = permutation[:split_num]
		self.__test_set = permutation[split_num:]
		print "Split into two parts with " + str(split_num) + " in train set and " + str(len(self) - split_num) + " in test set."
		pass

	def export(self, train_file = None, test_file = None):
		# Assume that the dataset has been divided into seperate part.
		with open(train_file, 'w') as f:
			contents = []
			for index in xrange(len(self.__train_set)):
				items = self.__info[self.__train_set[index]][1]
				for item in items:
					contents.append(item + ' ' + str(index) + '\n')
			f.writelines(contents)
			print "Export train set to", train_file
		with open(test_file, 'w') as f:
			contents = []
			for index in xrange(len(self.__test_set)):
				items = self.__info[self.__test_set[index]][1]
				for item in items:
					contents.append(item + ' ' + str(index) + '\n')
			f.writelines(contents)
			print "Export test set to", test_file
	pass

if __name__ == '__main__':
	dataset_root = '/home/chenzeyu/dataset/CASIA/CASIA-WebFace'
	dataset = Dataset(dataset_root)
	split_ratio = 0.7
	dataset.split(split_ratio)
	train_file = './data/train_set.txt'
	test_file = './data/test_set.txt'
	dataset.export(train_file, test_file)

