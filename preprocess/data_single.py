#! /usr/bin/env python
#-*- coding:utf-8 -*-

import os

# Iterate file system.
def loadInfo(dataset_root):
	print "Searching files in", dataset_root
	folders = sorted(os.listdir(dataset_root))
	file_info = []
	for folder in folders:
		folder_path = dataset_root + '/' + folder
		if not os.path.isdir(folder_path):
			continue
		files = sorted(os.listdir(folder_path))
		file_info.append(map(lambda x: folder_path + '/' + x, files))
	return file_info

def saveFileInfo(file_list, output_path):
	print "Writing file info to", output_path
	with open(output_path, 'w') as f:
		label = 0
		for filenames in file_list:
			for item in filenames:
				line = ' '.join([item, str(label)]) + '\n'
				f.write(line)
			label += 1
		f.close()

if __name__ == '__main__':
	root_path = '/home/chenzeyu/dataset/CASIA/CASIA-cropped'
	root_path = '/home/chenzeyu/dataset/lfw/lfw-cropped'
	file_info = loadInfo(root_path)
	output_path = 'test/CASIA.txt'
	output_path = 'test/lfw.txt'
	saveFileInfo(file_info, output_path)

