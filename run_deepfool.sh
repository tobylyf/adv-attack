#!/bin/bash

python run_deepfool.py \
    --gpu '1,3,6,9' \
    --imagenet_dir '/DATA4_DB3/data/kydu/data/' \
    --save_dir '/DATA5_DB8/data/yfli/datasets/tmp_deepfool/' \
    -p 'val' \
    -b 16
