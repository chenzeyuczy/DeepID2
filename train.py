#! /usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
import setup, caffe

from container import Container
from dataset import Dataset

caffe.set_mode_gpu()

dataset = Dataset("/home/chenzeyu/dataset/lfw-deepfunneled")
split_ratio = 0.01
dataset.split(split_ratio)
dataset.setPartition(1)

container = Container(dataset)
container.setBatchSize(100)
batch = container.next()

solver = caffe.SGDSolver("model/solver.prototxt")

#solver.net.copy_from("bvlc_reference_caffenet.caffemodel")

print solver.net.layers[0]
solver.net.layers[0].setData(container)
print solver.net.layers[0]

for i in xrange(10000):
	solver.step(5)
	if i % 100 == 0:
		solver.net.save("temp.caffemodel")

