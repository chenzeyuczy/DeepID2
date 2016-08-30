#! /bin/bash

CAFFE_PATH=~/software/caffe/build/tools/caffe

MODE=train

SOLVER=model/solver.prototxt
ITERATION=1000000
#WEIGHTS=result/deepid2_dropout_iter_500000.caffemodel
#SNAPSHOT=result/deepid2_dropout_iter_500000.solverstate

OPTION="-solver ${SOLVER} -iterations ${ITERATION}"
INFO=_dropout_fix

if [[ -n "${SNAPSHOT}" ]]; then
	OPTION="${OPTION} -snapshot ${SNAPSHOT}"
elif [[ -n "${WEIGHTS}" ]]; then
	OPTION="${OPTION} -weights ${WEIGHTS}"
fi

LOG_PATH=log/deepid2${INFO}_${ITERATION}.log

$CAFFE_PATH $MODE $OPTION 2>&1 | tee $LOG_PATH

