#! /usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
import setup, caffe

from container import Container
from dataset import Dataset

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

weights = "./model/bvlc_reference_caffenet.caffemodel"
# solver.net.copy_from("bvlc_reference_caffenet.caffemodel")

for i in xrange(10):
	solver.step(50)

