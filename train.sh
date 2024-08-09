#!/bin/bash

# 指定包含CSV文件的文件夹路径
folder_path="./data/ProteinNPT_data/fitness/substitutions_singles"

# 获取文件夹中的所有CSV文件
csv_files=$(ls "$folder_path"/*.csv)
# 要排除的文件列表
exclude_file1="A0A140D2T1_ZIKV_Sourisseau_2019.csv"
exclude_file2="A0A192B1T2_9HIV1_Haddox_2018.csv"
# 循环所有CSV文件并执行命令
for csv_file in $csv_files
do
    # 生成随机种子
    seed=1

    # 打印或记录当前正在处理的CSV文件
    # echo $csv_file
    # echo "Processing file: $(basename "$csv_file") with seed: $seed"

    if [ "$(basename "$csv_file")" = $exclude_file1 ] || [ "$(basename "$csv_file")" = $exclude_file2 ]; then
        echo "Skipping file: $(basename "$csv_file")"
        continue
    fi
    echo "Processing file: $(basename "$csv_file") with seed: $seed"
    # 构建并执行命令
    accelerate launch --config_file config/parallel_config.yaml scripts/train.py \
        --config config/training_config.yaml \
        --dataset $(basename "$csv_file") \
        --sample_seed 0 \
        --model_seed $seed
done