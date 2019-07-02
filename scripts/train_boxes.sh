#!/usr/bin/env bash
# Note: 200 server has no enough shared memory, set nThread and batch smaller.
CUDA_VISIBLE_DEVICES=6,7 python main.py  \
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
--decay_type step_120_180  \
--loss 5*CrossEntropy+1*Triplet  \
--margin 0.6  \
--random_erasing  \
--save Boxes_MGN_adam_margin_1.2_resize_keep_aspect_ratio_re_rank  \
--nThread 0 \
--nGPU 2   \
--lr 1e-4  \
--optimizer ADAM \
--save_models \
--resize_keep_aspect_ratio \
--re_rank \  # It seems that re_rank does not works well on Boxes dataset
--pre_train ./experiment/Boxes_MGN_adam_margin_1.2/model/model_160.pt