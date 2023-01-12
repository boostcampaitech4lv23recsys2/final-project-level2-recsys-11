#! /bin/bash
train_batch_size_list="2048 4096"
epochs_list="30 50"
neg_distribution_list="uniform popularity"
neg_sample_num_list="1 3 5"
neg_min_list="0 1"

embedding_size_list="8 16 32 64"
weight_decay_list="0.0 0.1 0.2"

for train_batch_size_var in $train_batch_size_list
do
    for epochs_var in $epochs_list
    do
        for neg_distribution_var in $neg_distribution_list
        do
            for neg_sample_num_var in $neg_sample_num_list
            do
                for neg_min_var in $neg_min_list
                do
                    for embedding_size_var in $embedding_size_list
                    do
                        for weight_decay_var in $weight_decay_list
                        do
                        python BPR.py \
                        --train_batch_size $train_batch_size_var \
                        --epochs $epochs_var \
                        --neg_distribution $neg_distribution_var \
                        --neg_sample_num $neg_sample_num_var \
                        --neg_min $neg_min_var \
                        --embedding_size $embedding_size_var \
                        --weight_decay $weight_decay_var
                        done
                    done
                done
            done
        done
    done    
done
