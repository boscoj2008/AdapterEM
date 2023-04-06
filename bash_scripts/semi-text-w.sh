#!/bin/bash
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 1e-4 --adapter_size 2
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 1e-4 --adapter_size 4
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 1e-4 --adapter_size 8
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 1e-4 --adapter_size 16
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 1e-4 --adapter_size 32

python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 1e-4 --adapter_size 2 -ts
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 1e-4 --adapter_size 4 -ts 
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 1e-4 --adapter_size 8 -ts
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 1e-4 --adapter_size 16 -ts
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 1e-4 --adapter_size 32 -ts


python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 2e-4 --adapter_size 2
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 2e-4 --adapter_size 4
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 2e-4 --adapter_size 8
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 2e-4 --adapter_size 16
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 2e-4 --adapter_size 32

python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 2e-4 --adapter_size 2 -ts
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 2e-4 --adapter_size 4 -ts 
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 2e-4 --adapter_size 8 -ts
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 2e-4 --adapter_size 16 -ts
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 2e-4 --adapter_size 32 -ts

python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 3e-4 --adapter_size 2
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 3e-4 --adapter_size 4
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 3e-4 --adapter_size 8
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 3e-4 --adapter_size 16
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 3e-4 --adapter_size 32

python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 3e-4 --adapter_size 2 -ts
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 3e-4 --adapter_size 4 -ts 
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 3e-4 --adapter_size 8 -ts
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 3e-4 --adapter_size 16 -ts
python src/main.py -d semi-text-w -n 30 -a_setup task_only -k 0.1 --lr 3e-4 --adapter_size 32 -ts
