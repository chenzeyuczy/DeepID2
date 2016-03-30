#! /usr/bin/python
#-*- coding:utf-8 -*-

# Script to setup environment.

import sys

caffe_path = "/home/chenzeyu/software/caffe/python"

if caffe_path not in sys.path:
	sys.path.append(caffe_path)

