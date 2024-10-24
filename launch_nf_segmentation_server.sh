#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
monailabel start_server --app nf_segmentation_app --studies data_folder --conf models all --conf batch_size 8 --conf resample_only_in_2d True
