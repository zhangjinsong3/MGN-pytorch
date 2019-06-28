#!/usr/bin/env bash
# Note: 200 server has no enough shared memory, set nThread and batch small.
CUDA_VISIBLE_DEVICES=6,7 python main.py  \
--datadir ../../Opensource_datasets/Market-1501-v15.09.15  \
--batchid 8  \
--batchtest 16  \
--test_every 20  \
--epochs 160  \
--decay_type step_120_140  \
--loss 1*CrossEntropy+2*Triplet  \
--margin 1.2  \
--re_rank  \
--random_erasing  \
--save MGN_adam_margin_1.2  \
--nThread 1 \
--nGPU 2   \
--lr 2e-4  \
--optimizer ADAM \
--save_models \
--reset