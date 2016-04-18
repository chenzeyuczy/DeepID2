#! /usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
import setup, caffe

from container import Container

caffe.set_mode_gpu()

# dataset = Dataset("/home/chenzeyu/dataset/lfw-deepfunneled")
# split_ratio = 0.01
# dataset.split(split_ratio)
# dataset.setPartition(1)
#
# container = Container(dataset)
# container.setBatchSize(100)

model = "./model/solver.prototxt"
solver = caffe.SGDSolver(model)

weights = "./result/deepid2_iter_10000.caffemodel"
#solver.net.copy_from("bvlc_reference_caffenet.caffemodel")

for i in xrange(100):
	solver.step(100)

