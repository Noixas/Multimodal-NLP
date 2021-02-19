#!/bin/bash

lrs="3e-5 1e-5 3e-6" # lower learning rate = slower training
dropouts="0.2 0.3 0.5" # higher dropout = stronger regularization
pos_weights="1 2 5" # loss weight of the positive class - useful to tackle class imbalance
# 3 * 3 * 3 = 27 combinations, so if each takes 30 mins, that makes 14 hours

for learning_rate in $lrs; do
    for dropout in $dropouts; do
        for pos_weight in $pos_weights; do
            # display arguments in multiline for better readability
            args=(
                # Paths
                --config config/uniter-base.json
                --data_path ./dataset --model_path ./model_checkpoints 
                --pretrained_model_file uniter-base.pt 
                --feature_path ./dataset/own_features
                --model_save_name meme.pt
                # Constant hyperparams
                --scheduler warmup_cosine 
                --warmup_steps 500 
                --max_epoch 30 
                --batch_size 16 
                --patience 5 
                --gradient_accumulation 2
                --seed 43

                # Changing hyperparams
                --lr ${learning_rate} 
                --dropout ${dropout}
                --pos_wt %{pos_weight}
                )

            python -u train_uniter.py "${args[@]}"
        done
    done
done
   