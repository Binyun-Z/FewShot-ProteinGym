import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy
from transformers import EsmForMaskedLM, EsmTokenizer, EsmConfig
import os
import argparse
from pathlib import Path
import accelerate
from accelerate import Accelerator
from datetime import datetime
from data_utils import Mutation_Set,split_data
from stat_utils import spearman, compute_score, BT_loss, KLloss
import gc
import warnings
import time
import yaml
warnings.filterwarnings("ignore")
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard

def train(model, model_reg, trainloader, optimizer, tokenizer, lambda_reg, pbar):

    model.train()

    total_loss = 0.

    for step, data in enumerate(trainloader):
        seq, mask = data[0], data[1]
        wt, wt_mask = data[2], data[3]
        pos = data[4]
        golden_score = data[5]
        score, logits = compute_score(model, seq, mask, wt, pos, tokenizer)
        score = score.cuda()

        l_BT = BT_loss(score, golden_score)

        out_reg = model_reg(wt, wt_mask)
        logits_reg = out_reg.logits
        l_reg = KLloss(logits, logits_reg, seq, mask)

        loss = l_BT + lambda_reg*l_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # 更新进度条
        pbar.update(1)
        # 记录训练损失到 TensorBoard
        # writer.add_scalar('BTloss', l_BT.item(), epoch * len(trainloader) + step)
        # writer.add_scalar('KL_loss', l_reg.item(), epoch * len(trainloader) + step)
        # writer.add_scalar('Train/Loss', total_loss, epoch * len(trainloader) + step)
    return total_loss


def evaluate(model, testloader, tokenizer, accelerator, istest=False):
    model.eval()
    seq_list = []
    score_list = []
    gscore_list = []
    with torch.no_grad():
        for step, data in enumerate(testloader):
            seq, mask = data[0], data[1]
            wt, wt_mask = data[2], data[3]
            pos = data[4]
            golden_score = data[5]
            pid = data[6]
            if istest:
                pid = pid.cuda()
                pid = accelerator.gather(pid)
                for s in pid:
                    seq_list.append(s.cpu())

            score, logits = compute_score(model, seq, mask, wt, pos, tokenizer)

            score = score.cuda()
            score = accelerator.gather(score)
            golden_score = accelerator.gather(golden_score)
            score = np.asarray(score.cpu())
            golden_score = np.asarray(golden_score.cpu())
            score_list.extend(score)
            gscore_list.extend(golden_score)
    score_list = np.asarray(score_list)
    gscore_list = np.asarray(gscore_list)
    sr = spearman(score_list, gscore_list)

    if istest:
        seq_list = np.asarray(seq_list)

        return sr, score_list, seq_list
    else:
        return sr


def main():
    parser = argparse.ArgumentParser(description='ConFit train, set hyperparameters')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                        help='the config file name')
    parser.add_argument('--dataset', type=str,default='A0A140D2T1_ZIKV_Sourisseau_2019.csv',help='the dataset name')
    parser.add_argument('--sample_seed', type=int, default=0, help='the sample seed for dataset')
    parser.add_argument('--model_seed', type=int, default=1, help='the random seed for the pretrained model initiate')
    parser.add_argument('--fold_spilt_type', type=str, default='fold_modulo_5', help='method for split data')
    args = parser.parse_args()
    dataset = args.dataset

    #read in config
    with open(f'{args.config}', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    batch_size = int(int(config['batch_size'])/int(config['gpu_number']))

    sc_list = []

    # for fold_i in range(1):# 五折交叉验证
    accelerator = Accelerator()

    val_i = 4




    ### creat model
    if config['model'] == 'ESM-1v':
        basemodel = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')
        model_reg = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')
        tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')

    elif config['model'] == 'ESM-2':
        basemodel = EsmForMaskedLM.from_pretrained('facebook/esm2_t48_15B_UR50D')
        model_reg = EsmForMaskedLM.from_pretrained('facebook/esm2_t48_15B_UR50D')
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')

    elif config['model'] == 'ESM-1b':
        basemodel = EsmForMaskedLM.from_pretrained('facebook/esm1b_t33_650M_UR50S')
        model_reg = EsmForMaskedLM.from_pretrained('facebook/esm1b_t33_650M_UR50S')
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm1b_t33_650M_UR50S')

    for pm in model_reg.parameters():
        pm.requires_grad = False
    model_reg.eval()    #regularization model


    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config['lora_r']),
        lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']),
        target_modules=["query", "value"]
    )

    model = get_peft_model(basemodel, peft_config)

    # create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['ini_lr']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2*int(config['max_epochs']), eta_min=float(config['min_lr']))
    if os.environ.get("ACCELERATE_USE_FSDP", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    model_reg = accelerator.prepare(model_reg)

    accelerator.print(f'===================dataset:{dataset}, preparing data=============')

    # sample data
    data_path = 'data/ProteinNPT_data/fitness/substitutions_singles/'
    if accelerator.is_main_process:
        split_data(dataset,data_path)


    # with accelerator.main_process_first():
    #     train_csv = pd.DataFrame(None)
        
    #     for i in range(5):
    #         if i == fold_i: #后面改为交叉验证
    #             val_csv = pd.read_csv(os.path.join(data_path,dataset.split('.')[0],f'{dataset.split('.')[0]}_{args.fold_spilt_type}_data_{i}.csv'))   #using 1/5 train data as validation set
    #         temp_csv = pd.read_csv(os.path.join(data_path,dataset.split('.')[0],f'{dataset.split('.')[0]}_{args.fold_spilt_type}_data_{i}.csv')) 
    #         train_csv = pd.concat([train_csv, temp_csv], axis=0)
    #     test_csv = val_csv 
    

    with accelerator.main_process_first():
        train_csv = pd.DataFrame(None)
        for i in range(5):
            if i == val_i: #后面改为交叉验证
                val_csv = pd.read_csv(os.path.join(data_path,dataset.split('.')[0],f'{dataset.split('.')[0]}_{args.fold_spilt_type}_data_{i}.csv'))   #using 1/5 train data as validation set
            if i == val_i-1: #后面改为交叉验证
                test_csv = pd.read_csv(os.path.join(data_path,dataset.split('.')[0],f'{dataset.split('.')[0]}_{args.fold_spilt_type}_data_{i}.csv'))   #using 1/5 train data as validation set
            temp_csv = pd.read_csv(os.path.join(data_path,dataset.split('.')[0],f'{dataset.split('.')[0]}_{args.fold_spilt_type}_data_{i}.csv')) 
            train_csv = pd.concat([train_csv, temp_csv], axis=0)



    #creat dataset and dataloader
    trainset = Mutation_Set(data=train_csv, tokenizer=tokenizer)
    testset = Mutation_Set(data=test_csv,  tokenizer=tokenizer)
    valset = Mutation_Set(data=val_csv,  tokenizer=tokenizer)
    with accelerator.main_process_first():
        trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=trainset.collate_fn, shuffle=True)
        testloader = DataLoader(testset, batch_size=2, collate_fn=testset.collate_fn)
        valloader = DataLoader(valset, batch_size=2, collate_fn=valset.collate_fn)

    trainloader = accelerator.prepare(trainloader)
    testloader = accelerator.prepare(testloader)
    valloader = accelerator.prepare(valloader)
    accelerator.print('==============data preparing done!================')
    accelerator.print("Current allocated memory:", torch.cuda.memory_allocated())
    accelerator.print("cached:", torch.cuda.memory_reserved())


    best_sr = -np.inf
    best_epoch = 0
    tolerance = 0.005  # 收敛容忍度
    prev_loss = float('inf')  # 上一个损失初始化为无穷大
    endure = 0  # 记录耐心值

    log_dir = os.path.join('scripts/runs', f"{dataset}_{args.fold_spilt_type}_{args.model_seed}_{datetime.now().strftime("%Y-%m-%d %H-%M")}_val_{val_i}")
    writer = SummaryWriter(log_dir)
    accelerator.print(f'日志文件的目录为：-----------------------> {log_dir}')


    checkpoint_dir = os.path.join('checkpoint', f'{dataset}',f'{args.fold_spilt_type}',
                                    f'model_seed_{args.model_seed}_{datetime.now().strftime("%Y-%m-%d %H-%M")}_val_{val_i}')
    if not os.path.isdir(checkpoint_dir):
        if accelerator.is_main_process:
            os.makedirs(checkpoint_dir)
    accelerator.print(f'模型权重文件的目录为：-----------------------> {checkpoint_dir}')

    for epoch in range(int(config['max_epochs'])):
        with tqdm(total=len(trainloader), desc=f"Epoch {epoch + 1}/{config['max_epochs']}", unit="batch") as pbar:
            loss = train(model, model_reg, trainloader, optimizer, tokenizer, float(config['lambda_reg']),pbar)
        accelerator.print(f'========epoch{epoch}; training loss :{loss}=================')

        writer.add_scalar('Train/Loss', loss, epoch)
        sr = evaluate(model, valloader, tokenizer, accelerator)
        writer.add_scalar('val spearman correlation', sr, epoch)
        accelerator.print(f'========epoch{epoch}; val spearman correlation :{sr}=================')
        scheduler.step()
        # if best_sr > sr:
        #     endure += 1
        # else:
        #     endure = 0
        #     best_sr = sr
        #     best_epoch = epoch
        #     accelerator.wait_for_everyone()
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     unwrapped_model.save_pretrained(checkpoint_dir)
        # if sr == 1.0:
        #     accelerator.print(f'========early stop at epoch{epoch}!============')
        #     break
        # if endure > int(config['endure_time']):
        #     accelerator.print(f'========early stop at epoch{epoch}!============')
        #     break
        # 检查损失是否收敛
        if abs(prev_loss - loss) < tolerance:
            endure += 1
        else:
            endure = 0
            
        prev_loss = loss  # 更新上一个损失
        
        if sr == 1.0:
            accelerator.print(f'========early stop at epoch{epoch}!============')
            sc_list.append(best_sr)
            break
        
        if endure > int(config['endure_time']):
            sc_list.append(best_sr)
            accelerator.print(f'========early stop at epoch{epoch}!============')
            break

        # 保存最佳模型
        if best_sr < sr:
            best_sr = sr
            best_epoch = epoch
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(checkpoint_dir)
        if epoch==int(config['max_epochs']):
            sc_list.append(best_sr)
        
        # 清理内存和删除模型
        # del model
        # del basemodel
        # del model_reg
        # del tokenizer
        # del optimizer
        # del scheduler
        # accelerator.free_memory()
        # accelerator.print(f'========Fold {fold_i} completed. Memory cleared.============')

        # # inference on the test sest
    accelerator.print('=======training done!, test the performance!========')

    del basemodel
    del model
    accelerator.free_memory()

    if config['model'] == 'ESM-1v':
        basemodel = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')
        tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')

    if config['model'] == 'ESM-2':
        basemodel = EsmForMaskedLM.from_pretrained('facebook/esm2_t48_15B_UR50D')
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')

    if config['model'] == 'ESM-1b':
        basemodel = EsmForMaskedLM.from_pretrained('facebook/esm1b_t33_650M_UR50S')
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm1b_t33_650M_UR50S')

    model = PeftModel.from_pretrained(basemodel, checkpoint_dir)
    model = accelerator.prepare(model)
    sr, score, pid = evaluate(model, testloader, tokenizer, accelerator, istest=True)
    pred_csv = pd.DataFrame({f'{args.model_seed}': score, 'PID': pid})
    if accelerator.is_main_process:
        if not os.path.isdir(f'predicted/{dataset}'):
            os.makedirs(f'predicted/{dataset}')
        if os.path.exists(f'predicted/{dataset}/pred.csv'):
            pred = pd.read_csv(f'predicted/{dataset}/pred.csv', index_col=0)
            pred = pd.merge(pred, pred_csv, on='PID')
        else:
            pred = pred_csv
        pred.to_csv(f'predicted/{dataset}/pred.csv')
    accelerator.print(f'=============the test spearman correlation for early stop: {sr}==================')

    # print(f'训练结束！-------->  {dataset}数据集中五折交叉验证的平均相关系数：{np.mean(sc_list)}')


if __name__ == "__main__":
    main()