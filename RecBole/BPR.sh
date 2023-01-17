#! /bin/bash
find saved/ -name "BPR*" -delete

neg_distribution_list="uniform popularity"
neg_sample_num_list="1 3"
embedding_size_list="16 32"
weight_decay_list="0.0 0.2"


for neg_distribution_var in $neg_distribution_list
do
    for neg_sample_num_var in $neg_sample_num_list
    do
        for embedding_size_var in $embedding_size_list
        do
            for weight_decay_var in $weight_decay_list
            do
            python BPR.py \
            --neg_distribution $neg_distribution_var \
            --neg_sample_num $neg_sample_num_var \
            --embedding_size $embedding_size_var \
            --weight_decay $weight_decay_var
            done
        done
    done
done

