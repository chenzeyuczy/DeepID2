#! /usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
from PIL import Image

def preprocessImage(img, size):
	img = Image.fromarray(img)
	img = img.resize(size)
	img = np.array(img, dtype = np.float64)
	img = img.transpose(2, 0, 1)
	return img

