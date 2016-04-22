#! /usr/bin/python

import cv2, numpy as np
import os

if __name__ == '__main__':
	dataset_root = '/home/chenzeyu/dataset/CASIA/CASIA-aligned/'
	output_root = '/home/chenzeyu/dataset/CASIA/CASIA-cropped/'

	if not os.path.exists(output_root):
		os.mkdir(output_root)

	target_size = (150, 150)

	folders = os.listdir(dataset_root)
	for folder in folders:
		folder_path = dataset_root + folder
		output_folder = output_root + folder

		if not os.path.exists(output_folder):
			os.mkdir(output_folder)

		img_names = os.listdir(folder_path)
		for img_name in img_names:
			img_path = folder_path + '/' + img_name
			output_path = output_folder + '/' + img_name
			img = cv2.imread(img_path)
			img = cv2.resize(img, target_size)
			cv2.imwrite(output_path, img)

