#! /bin/bash

python mf.py --loss bce --sampler uni --lr 0.01
python mf.py --loss bce --sampler pop --lr 0.01
python mf.py --loss bpr --sampler uni --lr 0.01
python mf.py --loss bpr --sampler pop --lr 0.01