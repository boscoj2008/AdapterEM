#!/bin/bash
python src/train_snli_adapter.py --model_name_or_path bert-base-uncased --dataset_name snli --do_train  --do_predict --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 1e-4 --num_train_epochs 15 --output_dir tmp/snli/ --lr_scheduler_type constant  --overwrite_output_dir 
