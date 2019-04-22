#!/bin/bash

python run_deepfool.py \
    --gpu '0,1,2,3' \
    --imagenet_dir '/DATA4_DB3/data/kydu/data/' \
    --save_dir '/DATA5_DB8/data/yfli/datasets/tmp_deepfool/' \
    -p 'train' \
    -b 256
