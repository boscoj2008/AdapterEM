#!/bin/bash
python src/mlm_train.py -d rel-heter --mlm_prob 0.40 --epochs 3
python src/mlm_train.py -d rel-text --mlm_prob 0.40 --epochs 3
python src/mlm_train.py -d semi-heter --mlm_prob 0.40 --epochs 3
python src/mlm_train.py -d semi-homo --mlm_prob 0.40 --epochs 3
python src/mlm_train.py -d semi-rel --mlm_prob 0.40 --epochs 3
python src/mlm_train.py -d semi-text-c --mlm_prob 0.40 --epochs 3
python src/mlm_train.py -d semi-text-w --mlm_prob 0.40 --epochs 3
python src/mlm_train.py -d geo-heter --mlm_prob 0.40 --epochs 3
