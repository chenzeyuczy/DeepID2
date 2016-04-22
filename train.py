#! /usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
import setup, caffe

from container import Container

caffe.set_mode_gpu()

model = "./model/solver.prototxt"
solver = caffe.SGDSolver(model)

weights = "./result/deepid2_iter_1000.caffemodel"
state = "./result/deepid2_iter_1000.solversate"
#solver.net.copy_from(weights)
#solver.net.restore(state)

#solver.forward()
for i in xrange(500):
	solver.step(1000)

