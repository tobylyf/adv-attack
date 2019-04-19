#!/bin/bash

python run_cw.py \
    --gpu '1,2,6,9' \
    --imagenet_dir '/DATA4_DB3/data/kydu/data/' \
    --save_dir '/DATA5_DB8/data/yfli/datasets/tmp_cw/' \
    -p 'val' \
    -b 16