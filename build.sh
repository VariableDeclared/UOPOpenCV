#!/bin/bash

echo "Which week is it?"
read WEEK
cd ${WEEK}/build
cmake ../ -DCMAKE_BINARY_DIR:STATIC=$(pwd)/bin
make
cd ../../
