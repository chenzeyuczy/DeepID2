#! /usr/bin/python
#-*- coding:utf-8 -*-

import setup

import numpy as np
import caffe

caffe.set_mode_gpu()

dataset_path = '/home/chenzeyu/dataset/lfw-funneled'
dataset = Dataset(dataset_path)
dataset.split()
train_data = dataset.getData(1)

solver_config = 'model/solver.prototxt'
solver = caffe.SGDSolver(solver_config)

copy_model = None
if copy_model:
	solver.net.copy_from(copy_model)

solver.net.layers[0].setDataConfig(data_config)

max_iter_num = 100
solver.step(max_iter_num)

