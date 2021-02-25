#!/bin/bash

seeds="18 43 99 " 
# echo "Start base model random seed run"
# for seed in $seeds; do
#             echo "Seed:"
#             echo ${seed}
#             args=(
#                 # Paths
#                 --config config/uniter-base.json
#                 --data_path ./dataset --model_path ./model_checkpoints 
#                 --pretrained_model_file uniter-base.pt 
#                 --feature_path ./dataset/own_features
#                 --model_save_name meme.pt
#                 # Constant hyperparams
#                 --scheduler warmup_cosine 
#                 --warmup_steps 500 
#                 --max_epoch 30 
#                 --batch_size 16 
#                 --patience 5 
#                 --gradient_accumulation 2
#                 --lr 3e-5 
#                 --pos_wt 1
#                 --normalize_img
#                 --seed ${seed}

#                 --linear_layers  1
#                 --gender_race_hidden_size 0
#                 --upsample_multiplier 0
#                 --note "Baseline (with normalization, 1 linear layer) "
#                 )

#             python -u train_uniter.py "${args[@]}"
# done
# echo "Start 2 linear layer random seed run" 
# for seed in $seeds; do
#             echo "Seed:"
#             echo ${seed}
#             args=(
#                 # Paths
#                 --config config/uniter-base.json
#                 --data_path ./dataset --model_path ./model_checkpoints 
#                 --pretrained_model_file uniter-base.pt 
#                 --feature_path ./dataset/own_features
#                 --model_save_name meme.pt
#                 # Constant hyperparams
#                 --scheduler warmup_cosine 
#                 --warmup_steps 500 
#                 --max_epoch 30 
#                 --batch_size 16 
#                 --patience 5 
#                 --gradient_accumulation 2
#                 --lr 3e-5 
#                 --pos_wt 1
#                 --normalize_img
#                 --seed ${seed}

#                 --linear_layers  2
#                 --gender_race_hidden_size 0
#                 --upsample_multiplier 0
#                 --note "Baseline + 2 linear layers"
#                 )

#             python -u train_uniter.py "${args[@]}"
# done
# seeds_crash = 99 
# echo "Start text filtering random seed run" 
# # for seed in $seeds_crash; do
# echo "Seed:"
# echo 99 
# args=(
#     # Paths
#     --config config/uniter-base.json
#     --data_path ./dataset --model_path ./model_checkpoints 
#     --pretrained_model_file uniter-base.pt 
#     --feature_path ./dataset/own_features
#     --model_save_name meme.pt
#     # Constant hyperparams
#     --scheduler warmup_cosine 
#     --warmup_steps 500 
#     --max_epoch 30 
#     --batch_size 16 
#     --patience 5 
#     --gradient_accumulation 2
#     --lr 3e-5 
#     --pos_wt 1
#     --normalize_img
#     --seed 99 

#     --linear_layers  1
#     --gender_race_hidden_size 0
#     --upsample_multiplier 0
#     --note "Baseline + Text Filtering"
#     --filter_text
#     )

# python -u train_uniter.py "${args[@]}"
# done
# echo "Start psampling random seed run" 
# for seed in $seeds; do
#             echo "Seed:"
#             echo ${seed}
#             args=(
#                 # Paths
#                 --config config/uniter-base.json
#                 --data_path ./dataset --model_path ./model_checkpoints 
#                 --pretrained_model_file uniter-base.pt 
#                 --feature_path ./dataset/own_features
#                 --model_save_name meme.pt
#                 # Constant hyperparams
#                 --scheduler warmup_cosine 
#                 --warmup_steps 500 
#                 --max_epoch 30 
#                 --batch_size 16 
#                 --patience 5 
#                 --gradient_accumulation 2
#                 --lr 3e-5 
#                 --pos_wt 1
#                 --normalize_img
#                 --seed ${seed}

#                 --linear_layers  1
#                 --gender_race_hidden_size 0
#                 --upsample_multiplier 3
#                 --note "Baseline + 3x Upsampled"
#                 )

#             python -u train_uniter.py "${args[@]}"
# done
# echo "Start Gender & Race probs random seed run" 
# for seed in $seeds; do
#             echo "Seed:"
#             echo ${seed}
#             args=(
#                 # Paths
#                 --config config/uniter-base.json
#                 --data_path ./dataset --model_path ./model_checkpoints 
#                 --pretrained_model_file uniter-base.pt 
#                 --feature_path ./dataset/own_features
#                 --model_save_name meme.pt
#                 # Constant hyperparams
#                 --scheduler warmup_cosine 
#                 --warmup_steps 500 
#                 --max_epoch 30 
#                 --batch_size 16 
#                 --patience 5 
#                 --gradient_accumulation 2
#                 --lr 3e-5 
#                 --pos_wt 1
#                 --normalize_img
#                 --seed ${seed}

#                 --linear_layers  1
#                 --gender_race_hidden_size 8
#                 --upsample_multiplier 0
#                 --note "Baseline + Gender & Race probs"
#                 )

#             python -u train_uniter.py "${args[@]}"
# done
# echo "Start 2 linear layer + Text filtering random seed run" 
# for seed in $seeds; do
#             echo "Seed:"
#             echo ${seed}
#             args=(
#                 # Paths
#                 --config config/uniter-base.json
#                 --data_path ./dataset --model_path ./model_checkpoints 
#                 --pretrained_model_file uniter-base.pt 
#                 --feature_path ./dataset/own_features
#                 --model_save_name meme.pt
#                 # Constant hyperparams
#                 --scheduler warmup_cosine 
#                 --warmup_steps 500 
#                 --max_epoch 30 
#                 --batch_size 16 
#                 --patience 5 
#                 --gradient_accumulation 2
#                 --lr 3e-5 
#                 --pos_wt 1
#                 --normalize_img
#                 --seed ${seed}

#                 --linear_layers  2
#                 --gender_race_hidden_size 0
#                 --upsample_multiplier 0
#                 --note "Baseline + 2 linear layers + Text Filtering"
#                 --filter_text

#                 )

#             python -u train_uniter.py "${args[@]}"
# done
echo "Start  2 linear layers + Text Filtering + 3x Upsampled random seed run" 
for seed in $seeds; do
            echo "Seed:"
            echo ${seed}
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
                --lr 3e-5 
                --pos_wt 1
                --normalize_img
                --seed ${seed}

                --linear_layers  2
                --gender_race_hidden_size 0
                --upsample_multiplier 3
                --note "Baseline + 2 linear layers + Text Filtering + 3x Upsampled"
                --filter_text

                )

            python -u train_uniter.py "${args[@]}"
done