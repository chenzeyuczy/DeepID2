#! /usr/bin/python

import numpy as np, cv2
import os

def face_align(img, face_cascade, target_size = None):
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	print len(faces), "faces found."
	for (x,y,w,h) in faces:
		img_crop = img[y: y + h, x: x + w]
		if target_size:
			img_crop = cv2.resize(img_crop, target_size)
		return img_crop
	if target_size:
		img = cv2.resize(img, target_size)
	return img

if __name__ == "__main__":
	src_dataset = "/home/chenzeyu/dataset/CASIA/CASIA-WebFace/"
	des_dataset = "/home/chenzeyu/dataset/CASIA/CASIA-cropped/"

	if not os.path.exists(des_dataset):
		os.makedirs(des_dataset)

	class_names = sorted(os.listdir(src_dataset))
	counter = 0
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	target_size = (150, 150)

	for class_name in class_names:
		folder = src_dataset + class_name + '/'
		file_names = os.listdir(folder)

		counter += 1
		print "Processing class", class_name, "with",  len(file_names), "images,", len(class_names) - counter, "class(es) left."

		des_dir = des_dataset + class_name
		if not os.path.exists(des_dir):
			os.makedirs(des_dir)

		for file_name in file_names:
			img_path = src_dataset + class_name + '/' + file_name
			img = cv2.imread(img_path)
			img_crop = face_align(img, face_cascade, target_size)
			img_path = des_dataset + class_name + '/' + file_name
			cv2.imwrite(img_path, img_crop)

