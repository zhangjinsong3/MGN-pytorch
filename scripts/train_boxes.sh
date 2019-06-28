#!/usr/bin/env bash
# Note: 200 server has no enough shared memory, set nThread and batch smaller.
CUDA_VISIBLE_DEVICES=6,7 python main.py  \
--datadir ../../Opensource_datasets/Boxes  \
--data_train Boxes \
--data_test Boxes \
--num_classes 3000 \
--height 256 \
--width 256 \
--batchid 8  \
--batchtest 16  \
--test_every 10  \
--epochs 160  \
--decay_type step_120_140  \
--loss 1*CrossEntropy+2*Triplet  \
--margin 1.2  \
--re_rank  \
--random_erasing  \
--save Boxes_MGN_adam_margin_1.2  \
--nThread 0 \
--nGPU 2   \
--lr 2e-4  \
--optimizer ADAM \
--save_models \
--reset