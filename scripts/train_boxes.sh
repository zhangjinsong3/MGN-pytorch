#!/usr/bin/env bash
# Note: 200 server has no enough shared memory, set nThread and batch smaller.
CUDA_VISIBLE_DEVICES=4,5 python main.py  \
--datadir ../../Opensource_datasets/Boxes  \
--data_train Boxes \
--data_test Boxes \
--num_classes 3000 \
--height 256 \
--width 256 \
--batchid 16  \
--batchtest 32  \
--test_every 10  \
--epochs 200  \
--decay_type step_100_140  \
--loss 2*CrossEntropy+1*Triplet  \
--margin 0.6  \
--random_erasing  \
--random_crop \
--color_jitter \
--save Boxes_MGN_adam_margin_0.6_resize_keep_aspect_ratio_new_dataset_augmentations \
--nThread 0 \
--nGPU 2   \
--lr 1e-4  \
--optimizer ADAM \
--save_models \
--resize_keep_aspect_ratio \
--resume 0
# --re_rank \  # It seems that re_rank does not works well on Boxes dataset