#!/bin/bash
python main.py --dataset four-square-$1 --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --save_flag 1 --pretrain 0 --batch_size 256 --epoch 30 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_id 1 --weights_path models/model_$1




