#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os
from PIL import Image

def retrive_folder(root_path):
	file_list = []
	folders = os.listdir(root_path)
	for folder in folders:
		folder_path = root_path + '/' + folder
		file_names = os.listdir(folder_path)
		for file_name in file_names:
			file_path = folder + '/' + file_name
			file_list.append(file_path)
	print("%d files found in %s." % (len(file_list), root_path))
	return file_list

def check_image(root_path, file_list, img_size, img_mode):
	print("Checking original image format.")
	print("Default size: %s, default mode: %s" % (img_size, img_mode))
	counter_size = 0
	counter_mode = 0
	for file_path in file_list:
		file_path = root_path + '/' + file_path
		img = Image.open(file_path)
		if img.size != img_size:
			print("Size of %s: %s" % (img, img.size))
			counter_size += 1
		if img.mode != img_mode:
			print("Mode of %s: %s" % (img, img.mode))
			counter_mode += 1
	print("%d img don't match the size, %d img don't match the mode." % (counter_size, counter_mode))
	return counter_size | counter_mode == 0

def transform_image(src_root, dst_root, file_list, config):
	img_box, img_mode, img_flip = config
	print "Transforming images from %s to %s." % (src_root, dst_root)

	for file_path in file_list:
		src_path = src_root + '/' + file_path
		dst_path = dst_root + '/' + file_path

		dst_dir = os.path.dirname(dst_path)
		if not os.path.exists(dst_dir):
			os.makedirs(dst_dir)

		img = Image.open(src_path)
		if img_mode:
			img = img.convert(img_mode)
		if img_box:
			img = img.crop(img_box)
		if img_flip == True:
			img = img.flip(Image.FLIP_LEFT_RIGHT)
		img.save(dst_path)
	return

def load_config():
	configs = [
		((0, 0, 90, 90), 'RGB', False),
		((0, 60, 90, 150), 'RGB', False),
		((60, 0, 150, 90), 'RGB', False),
		((60, 60, 150, 150), 'RGB', False),
		((30, 30, 120, 120), 'RGB', False),
	]
	return configs

if __name__ == '__main__':
	root_path = '/home/chenzeyu/dataset/CASIA/CASIA-cropped'

	file_list = retrive_folder(root_path)

	init_size = (150, 150)
	init_mode = 'RGB'
	if not check_image(root_path, file_list, init_size, init_mode):
		assert False

	configs = load_config()

	for config in configs:
		box, mode, flip = config
		config_str = '_'.join(map(str, box)) + '_' + mode
		if flip:
			config_str += "FLIP"
		dst_root = '_'.join(['/home/chenzeyu/dataset/CASIA/CASIA-cropped', config_str])
		print("Saving files to %s." % (dst_root))
		transform_image(root_path, dst_root, file_list, config)
	
