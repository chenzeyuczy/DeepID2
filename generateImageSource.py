#! /usr/bin/python
#-*- coding:utf-8 -*-

import os

dataset_path = '/home/chenzeyu/dataset/lfw-deepfunneled'
folders = sorted(os.listdir(dataset_path))

lines = []
counter = 0
for folder in folders:
	imgs = sorted(os.listdir(dataset_path + '/' + folder))
	for img in imgs:
		file_path = os.sep.join([dataset_path, folder, img])
		lines.append(file_path + ' ' +  str(counter) + '\n')
	counter += 1

file_name = 'ImageSource.txt'
f = open(file_name, 'w')
f.writelines(lines)
f.close()

