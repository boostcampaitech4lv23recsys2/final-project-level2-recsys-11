#! /bin/bash
python mf.py --exp_name SGD-MF --loss bce --sampler uni --embedding_size 8
python mf.py --exp_name SGD-MF --loss bce --sampler uni --embedding_size 16

python mf.py --exp_name SGD-MF --loss bce --sampler pop --embedding_size 8
python mf.py --exp_name SGD-MF --loss bce --sampler pop --embedding_size 16

python mf.py --exp_name BPR-MF --loss bpr --sampler uni --embedding_size 8
python mf.py --exp_name BPR-MF --loss bpr --sampler uni --embedding_size 16

python mf.py --exp_name BPR-MF --loss bpr --sampler pop --embedding_size 8
python mf.py --exp_name BPR-MF --loss bpr --sampler pop --embedding_size 16
