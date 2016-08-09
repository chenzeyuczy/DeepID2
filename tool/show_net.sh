#! /bin/bash

DRAW_NET_FILE="/home/chenzeyu/software/caffe/python/draw_net.py"
DEPLOY_FILE="../model/deploy.prototxt"
NET_IMG_PATH="../net.jpg"

python $DRAW_NET_FILE $DEPLOY_FILE $NET_IMG_PATH

