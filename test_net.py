#! /usr/bin/python
#-*- coding:utf-8 -*-

import setup, caffe
import layer.data_layer

deploy_file = "./model/deploy.prototxt"

weights = "./result/deepid2_iter_1000.caffemodel"
net = caffe.Net(deploy_file, 0)
net.copy_from(weights)

net.forward()
ids = np.argmax(net.blobs['id'].data, axis = 1)
labels = net.blobs['label']
