#!/bin/bash

python -i run_train.py \
    --gpu '0,1,2,3' \
    -p 0.3 \
    -w 10000 \
    --lr 0.00005 \
    --name ResNet50_p0.3_2layers_w10000_lr0.00005_DeepFool \