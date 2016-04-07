#! /usr/bin/python

import cv2, numpy as np
import os

if __name__ == '__main__':
	dataset = '/home/chenzeyu/dataset/lfw-cropped/'

	target_size = (250, 250)

	folders = os.listdir(dataset)
	for folder in folders:
		folder_path = dataset + folder
		img_names = os.listdir(folder_path)
		for img_name in img_names:
			img_path = folder_path + '/' + img_name
			img = cv2.imread(img_path)
			img = cv2.resize(img, target_size)
			cv2.imwrite(img_path, img)

