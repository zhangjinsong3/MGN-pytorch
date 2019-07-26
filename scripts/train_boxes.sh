#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2,6 python main.py  \
--datadir ../../Opensource_datasets/Boxes  \
--data_train Boxes \
--data_test Boxes \
--num_classes 3719 \
--height 384 \
--width 384 \
--batchid 16  \
--batchtest 32  \
--test_every 10  \
--epochs 200  \
--decay_type step_100_140  \
--loss 2*CrossEntropy+1*Triplet  \
--margin 0.6  \
--random_erasing  \
--random_crop \
--save Boxes_MGN_adam_margin_0.6_resize_keep_aspect_ratio_new_dataset_augmentations_384_num_classes_3719  \
--nThread 8 \
--nGPU 2   \
--lr 2e-4  \
--optimizer ADAM \
--save_models \
--resize_keep_aspect_ratio \
--resume 180  \
--test_only
#--multi_gallery \
#--retrieval_only
#--test_only
#--use_mask  \
#--re_rank   # It seems that re_rank does not works well on Boxes dataset
#--color_jitter \
