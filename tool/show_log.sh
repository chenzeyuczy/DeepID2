#! /bin/bash

PARSE_TOOL=/home/chenzeyu/software/caffe/tools/extra/parse_log.sh
DRAW_TOOL=/home/chenzeyu/software/caffe/tools/extra/plot_training_log.py.example
LOG_FILE=./log/deepid2_55_47_dropout_1000000.log
SUFFIX=.log
BASENAME=$(basename -s $SUFFIX ${LOG_FILE})
CLEAN_FLAG=true

MODE=6
OUTPUT_FILE=./test/image/$BASENAME.png

# Parse log.
#$PARSE_TOOL $LOG_FILE

# Draw picture from log.
$DRAW_TOOL $MODE $OUTPUT_FILE $LOG_FILE

# Remove temp files if required.
if [ $CLEAN_FLAG ] ; then
	TEST_DATA=$BASENAME$SUFFIX.test
	TRAIN_DATA=$BASENAME$SUFFIX.train
	rm $TEST_DATA $TRAIN_DATA
fi

