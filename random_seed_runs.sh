#!/bin/bash

seeds="18 43 99"

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
# echo "Start  2 linear layers + Text Filtering + 3x Upsampled random seed run" 
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
#                 --upsample_multiplier 3
#                 --note "Baseline + 2 linear layers + Text Filtering + 3x Upsampled"
#                 --filter_text

#                 )

#             python -u train_uniter.py "${args[@]}"
## done

# echo "Start  2 linear layers + Text Filtering +  Gender & Race probs" 
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
#                 --gender_race_hidden_size 8
#                 --upsample_multiplier 0
#                 --note "Baseline + 2 linear layers + Text Filtering +  Gender & Race probs"
#                 --filter_text

#                 )

#             python -u train_uniter.py "${args[@]}"
# done
# echo "Start  Base + Image upsampling" 
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
#                 # --filter_text
#                 --note "Baseline + Image upsampling"
                
#                 --upsample_options "I"

#                 )

#             python -u train_uniter.py "${args[@]}"
# done
# echo "Print only one seed for test purposes"
# seeds = "43"
# echo "Start  Base + 2 linear layers + text_filtering + Image upsampling" 
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

#                 --linear_layers 2
#                 --gender_race_hidden_size 0
#                 --upsample_multiplier 3
#                 --filter_text
#                 --note "Baseline + 2 linear layers + text_filtering +   Image upsampling"
                
#                 --upsample_options "I" #Option image [I] and [2] from text upsampling

#                 )

#             python -u train_uniter.py "${args[@]}"
# done
# echo "Start  Base + 2 linear layers + text_filtering + text_upsampling+  Image upsampling" 
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

#                 --linear_layers 2
#                 --gender_race_hidden_size 0
#                 --upsample_multiplier 3
#                 --filter_text
#                 --note "Baseline + 2 linear layers + text_filtering +   Image upsampling"
                
#                 --upsample_options "I 2" #Option image [I] and [2] from text upsampling

#                 )

#             python -u train_uniter.py "${args[@]}"
# done

# echo "Start  Base + 2 linear layers + Image upsampling" 
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

#                 --linear_layers 2
#                 --gender_race_hidden_size 0
#                 --upsample_multiplier 3
#                 # --filter_text
#                 --note "Baseline + 2 linear layers + Image upsampling"
                
#                 --upsample_options "I"

#                 )

#             python -u train_uniter.py "${args[@]}"
# done


# NOT DONE
# echo "Start  Base + text filtering + Image upsampling" 
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

#                 --linear_layers 1
#                 --gender_race_hidden_size 0
#                 --upsample_multiplier 3
#                 --filter_text
#                 --note "Baseline + 2 linear layers + Image upsampling"
                
#                 --upsample_options "I"

#                 )

#             python -u train_uniter.py "${args[@]}"
# done
# echo "Start  Base + Gender & Race probs + Image upsampling" 
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

#                 --linear_layers 1
#                 --gender_race_hidden_size 8
#                 --upsample_multiplier 3
#                 # --filter_text
#                 --note "Baseline + Gender & Race probs+ Image upsampling"
                
#                 --upsample_options "I"

#                 )

#             python -u train_uniter.py "${args[@]}"
# done
# echo "Start  Base + text_upsampling + Image upsampling" 
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

#                 --linear_layers 1
#                 --gender_race_hidden_size 0
#                 --upsample_multiplier 3
#                 # --filter_text
#                 --note "Baseline + text_upsampling+ Image upsampling"
                
#                 --upsample_options "I 2" #Option image [I] and [2] from text upsampling

#                 )

#             python -u train_uniter.py "${args[@]}"
# done
# echo "Start  Base + Augmenting data" 
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

#                 --linear_layers 1
#                 --gender_race_hidden_size 0
#                 --upsample_multiplier 3
#                 # --filter_text
#                 --note "Baseline + Augmenting data"
                
#                 --upsample_options "A" #Option image [I] and [2] from text upsampling

#                 )

#             python -u train_uniter.py "${args[@]}"
# done
seeds="18 43 99"
# echo "Start  Base + text_upsampling + Image upsampling" 
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

#                 --linear_layers 1
#                 --gender_race_hidden_size 0
#                 --upsample_multiplier 1.5
#                 # --filter_text
#                 --note "Baseline + text_upsampling+ Image upsampling"
                
#                 --upsample_options "I 2" #Option image [I] and [2] from text upsampling

#                 )

#             python -u train_uniter.py "${args[@]}"
# done






####################################
##TODO 
# multipliers="1 2 4 5 3"
# echo "Start  Base + Augmenting data" 
# for seed in $seeds; do
#     for multipl in $multipliers; do
#             echo "Seed:"
#             echo ${seed}
#             echo "Multiplier:"
#             echo ${multipl}
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

#                 --linear_layers 1
#                 --gender_race_hidden_size 0
#                 --upsample_multiplier ${multipl}
#                 # --filter_text
#                 --note "Baseline + Augmenting data"
                
#                 --upsample_options "A" #Option image [I] and [2] from text upsampling

#                 )

#             python -u train_uniter.py "${args[@]}"
#     done
# done
###################################################################################

echo "Start  Base + 2 linear layers + text_filtering + 1.5 text_upsampling +  1.5 Image upsampling" 
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

                --linear_layers 2
                --gender_race_hidden_size 0
                --upsample_multiplier 1.5
                --filter_text
                --note "Baseline + 2 linear layers + text_filtering + 1.5 text_upsampling +  1.5 Image upsampling"
                
                --upsample_options "I 2" #Option image [I] and [2] from text upsampling

                )

            python -u train_uniter.py "${args[@]}"
done

echo "Start  Base + 2 linear layers  + 1.5 text_upsampling +  1.5 Image upsampling" 
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

                --linear_layers 2
                --gender_race_hidden_size 0
                --upsample_multiplier 1.5
                # --filter_text
                --note "Baseline + 2 linear layers + 1.5 text_upsampling +  1.5 Image upsampling"
                
                --upsample_options "I 2" #Option image [I] and [2] from text upsampling

                )

            python -u train_uniter.py "${args[@]}"
done

multipliers="1 2 4 5 3"
echo "Start  Base + Augmenting data" 
for seed in $seeds; do
    for multipl in $multipliers; do
            echo "Seed:"
            echo ${seed}
            echo "Multiplier:"
            echo ${multipl}
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

                --linear_layers 1
                --gender_race_hidden_size 0
                --upsample_multiplier ${multipl}
                # --filter_text
                --note "Baseline + Augmenting data"
                
                --upsample_options "A" #Option image [I] and [2] from text upsampling

                )

            python -u train_uniter.py "${args[@]}"
    done
done