#! /bin/bash
reg_weight_list="0.0 100.0 200.0 300.0 400.0"

for reg_weight_var in $reg_weight_list
do
    python EASE.py \
    --reg_weight $reg_weight_var
done

