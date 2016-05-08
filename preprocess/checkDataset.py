#! /use/bin/python
#-*- coding:utf-8 -*-

import os

dataset_root = '/home/chenzeyu/dataset/CASIA/CASIA-cropped'

folders = sorted(os.listdir(dataset_root))

class_num = len(folders)
min_num = 100
max_num = 0
sum_num = 0

for folder in folders:
	folder_path = dataset_root + '/' + folder
	files = sorted(os.listdir(folder_path))
	file_num = len(files)
	print 'Class', folder, 'has', file_num, 'images.'
	if min_num > file_num:
		min_num = file_num
	if max_num < file_num:
		max_num = file_num
	sum_num += file_num

avg_num = sum_num * 1.0 / class_num
print 'Min:', min_num, 'Max:', max_num, 'Sum:', sum_num, 'Avg:', avg_num

