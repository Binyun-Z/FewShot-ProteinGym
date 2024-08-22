#!/bin/bash

# 指定包含CSV文件的文件夹路径
folder_path="./data/ProteinNPT_data/fitness/substitutions_singles"

# 获取文件夹中的所有CSV文件
csv_files=$(ls "$folder_path"/*.csv)

# 循环所有CSV文件并执行命令
for csv_file in $csv_files
do
    # 生成随机种子
    seed=1

    # 打印或记录当前正在处理的CSV文件

    echo "Processing file: $(basename "$csv_file") with seed: $seed"
    # 构建并执行命令
    accelerate launch --config_file config/parallel_config.yaml scripts/train.py \
        --config config/training_config.yaml \
        --dataset $(basename "$csv_file") \
        --sample_seed 0 \
        --model_seed $seed
done

# csv_files=(
#     "BLAT_ECOLX_Jacquier_2013.csv"
#     "CALM1_HUMAN_Weile_2017.csv"
#     "DYR_ECOLI_Thompson_2019.csv"
#     "DLG4_RAT_McLaughlin_2012.csv"
#     "REV_HV1H2_Fernandes_2016.csv"
#     "TAT_HV1BR_Fernandes_2016.csv"
#     "RL40A_YEAST_Roscoe_2013.csv"
#     "P53_HUMAN_Giacomelli_2018_WT_Nutlin.csv"
# )

# # 循环所有CSV文件并执行命令
# for csv_file in "${csv_files[@]}"
# do
#     # 生成随机种子
#     seed=1

#     # 打印或记录当前正在处理的CSV文件
#     echo "Processing file: $(basename "$csv_file") with seed: $seed"
#     accelerate launch --config_file config/parallel_config.yaml scripts/train.py \
#         --config config/training_config.yaml \
#         --dataset $(basename "$csv_file") \
#         --sample_seed 0 \
#         --model_seed $seed
# done