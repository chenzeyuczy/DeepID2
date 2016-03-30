#! /usr/bin/python
#-*- coding:utf-8 -*-

import setup, caffe
import layer.data_layer

deploy_file = "./model/deploy.prototxt"
net = caffe.Net(deploy_file, 0)

