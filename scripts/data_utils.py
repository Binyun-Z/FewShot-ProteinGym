import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import bootstrap
import numpy as np
import re
import os
import shutil
class Mutation_Set(Dataset):
    def __init__(self, data, tokenizer, sep_len=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = sep_len
        self.seq, self.attention_mask = tokenizer(list(self.data['mutated_sequence']), padding='max_length',
                                                  truncation=True,
                                                  max_length=self.seq_len).values()

        target = list(data['target_seq'])
        self.target, self.tgt_mask = tokenizer(target, padding='max_length', truncation=True,
                                               max_length=self.seq_len).values()
        self.score = torch.tensor(np.array(self.data['DMS_score']))
        self.pid = np.asarray(data['PID'])

        if type(list(self.data['mut_pos'])[0]) != str:
            self.position = [[u] for u in self.data['mut_pos']]

        else:
            self.position = []
            for u in self.data['mut_pos']:
                p = re.findall(r'\d+', u)
                pos = [int(v) for v in p]
                self.position.append(pos)



    def __getitem__(self, idx):
        return [self.seq[idx], self.attention_mask[idx], self.target[idx],self.tgt_mask[idx] ,self.position[idx], self.score[idx], self.pid[idx]]

    def __len__(self):
        return len(self.score)

    def collate_fn(self, data):
        seq = torch.tensor(np.array([u[0] for u in data]))
        att_mask = torch.tensor(np.array([u[1] for u in data]))
        tgt = torch.tensor(np.array([u[2] for u in data]))
        tgt_mask = torch.tensor(np.array([u[3] for u in data]))
        pos = [torch.tensor(u[4]) for u in data]
        score = torch.tensor(np.array([u[5] for u in data]), dtype=torch.float32)
        pid = torch.tensor(np.array([u[6] for u in data]))
        return seq, att_mask, tgt, tgt_mask, pos, score, pid
        
def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true)[0]

def compute_stat(sr):
    sr = np.asarray(sr)
    mean = np.mean(sr)
    std = np.std(sr)
    sr = (sr,)
    ci = list(bootstrap(sr, np.mean).confidence_interval)
    return mean, std, ci


def return_position_single(mutation):
    """Note: Only works for single mutations"""
    position = mutation.split(":")[0][1:-1]
    return int(position)

def keep_singles(DMS, mutant_column='mutant'):
    DMS = DMS[~DMS[mutant_column].str.contains(":")]
    return DMS


def create_folds_random(DMS, n_folds=5, mutant_column='mutant'):
    column_name = 'fold_random_{}'.format(n_folds)
    try:
        mutated_region_list = DMS[mutant_column].apply(lambda x: return_position_single(x)).unique()
    except:
        print("Mutated region not found from 'mutant' variable -- assuming the full protein sequence is mutated")
        mutated_region_list = range(len(DMS['mutated_sequence'].values[0]))
    len_mutated_region = len(mutated_region_list)
    if len_mutated_region < n_folds:
        raise Exception("Error, there are fewer mutated regions than requested folds")
    DMS[column_name] = np.random.randint(0, n_folds, DMS.shape[0])
    print(DMS[column_name].value_counts())
    return DMS

def create_folds_by_position_modulo(DMS, n_folds=5, mutant_column='mutant'):
    column_name = 'fold_modulo_{}'.format(n_folds)
    mutated_region_list = sorted(DMS[mutant_column].apply(return_position_single).unique())
    len_mutated_region = len(mutated_region_list)
    if len_mutated_region < n_folds:
        raise Exception("Error, there are fewer mutated regions than requested folds")
    position_to_fold = {pos: i % n_folds for i, pos in enumerate(mutated_region_list)}
    DMS[column_name] = DMS[mutant_column].apply(lambda x: position_to_fold[return_position_single(x)])
    print(DMS[column_name].value_counts())
    return DMS

def create_folds_by_contiguous_position_discontiguous(DMS, n_folds=5, mutant_column='mutant'):
    column_name = 'fold_contiguous_{}'.format(n_folds)
    mutated_region_list = sorted(DMS[mutant_column].apply(lambda x: return_position_single(x)).unique())
    len_mutated_region = len(mutated_region_list)
    k, m = divmod(len_mutated_region, n_folds)
    folds = [[i] * k + [i] * (i < m) for i in range(n_folds)]
    folds = [item for sublist in folds for item in sublist]
    folds_indices = dict(zip(mutated_region_list, folds))
    if len_mutated_region < n_folds:
        raise Exception("Error, there are fewer mutated regions than requested folds")
    DMS[column_name] = DMS[mutant_column].apply(lambda x: folds_indices[return_position_single(x)])
    print(DMS[column_name].value_counts())
    return DMS
def get_pos(row):
    pos = []
    for mut in row['mutant'].split(':'):
        result = int(re.findall(r'\d+', mut)[0])
        pos.append(result)
    if len(pos)<=1:return pos[0]
    else:
        return pos
def get_wt(seq, mut):
    # mut的输入为A2D, or A2D:B3C
    pos = []
    chars = []
    
    for mutation in mut.split(':'):
        original_char = mutation[0]  # 获取原始字符
        position = int(re.findall(r'\d+', mutation)[0])  # 获取位置
        pos.append(position)
        chars.append(original_char)  # 保存原始字符
    
    seq = list(seq)
    for i, p in enumerate(pos):
        seq[p - 1] = chars[i]  # 替换为原始字符
    
    return ''.join(seq)
def get_data(data_path):
    #ESM能处理的序列的最大长度1024，除去开始的字符和终止的字符最大氨基酸序列的长度为1022，
    # 但是某些序列的突变位点超过1022，因此需要处理

    df = pd.read_csv(data_path)
    df['mut_pos'] = df.apply(get_pos,axis = 1)

    positions = list(df['mut_pos'])
    # print(positions)
    max_pos, min_pos = max(positions), min(positions)
    gap = max_pos - min_pos + 1
    wt_seq = get_wt(df['mutated_sequence'][0],df['mutant'][0])
    L= len(wt_seq)
    # print(df.head(5))
    # print(f'pos-------------{min_pos,max_pos}')


    max_len = 1022
    if max_pos < max_len:
        wt_seq = wt_seq[:max_len]
        left, right = 0,max_len


    elif gap <= max_len:
        window_l = max(min_pos - (max_len - gap) // 2, 0)
        window_r = min(max_pos + (max_len - gap) // 2, L - 1)
        seq_lr = wt_seq[window_l: window_l + max_len]
        seq_rl = wt_seq[window_r - max_len + 1: window_r + 1]
        
        if len(seq_lr) > len(seq_rl):
            wt_seq = seq_lr
            left, right = window_l, window_l + max_len
        else:
            wt_seq = seq_rl
            left, right = window_r - max_len + 1, window_r + 1
    else:
        n = 0
        left, right = 0, max_len
        for w in range(max_len,L,10):
            window_n = sum((df['mut_pos']>=w-max_len) & (df['mut_pos']<w))
            # print(f"Window: {w}, Count: {window_n}")
            # print(window_n)
            if window_n > n:
                left= w-max_len
                n = window_n
        
        right  = left+max_len
        wt_seq = wt_seq[left:right]

    # print(left,right)

    df = df[(df['mut_pos']>=left) & (df['mut_pos']<right)]
    df['mutated_sequence'] =df['mutated_sequence'].apply(lambda seq: seq[left:right])
    df['mut_pos'] = df['mut_pos']-left
    df['target_seq'] = wt_seq
    df['PID'] = df.index
    # print(df.head(5))

    return df

def split_data(dataset,data_path):
    if isinstance(dataset, list):
        for csv_file in dataset:
            df = pd.read_csv(os.path.join(data_path,csv_file))
            df['mut_pos'] = df.apply(get_pos,axis = 1)
            wt_seq = get_wt(df['mutated_sequence'][0],df['mutant'][0])
            df['target_seq'] = wt_seq
            df['PID'] = df.index
            df = df[df['mut_pos']<1023]
            split_data_path = os.path.join(data_path,csv_file.split('.')[0])
            if os.path.exists(split_data_path):
                print(f'----------------->{csv_file}{"-" * (60 - len(csv_file))} 数据集已经划分！删除重新划分')
                shutil.rmtree(split_data_path)
            os.makedirs(split_data_path)
            fold_types = ['fold_random_5','fold_modulo_5','fold_contiguous_5']
            for fold_type in fold_types:
                for i in range(5):
                    data = df[df[fold_type]==i]
                    data.to_csv(os.path.join(split_data_path,f'{csv_file.split('.')[0]}_{fold_type}_data_{i}.csv'),index = False)
            print(f'----------------->{csv_file}{"-" * (60 - len(csv_file))} 数据集已经划分！')
    else:
        df = pd.read_csv(os.path.join(data_path,dataset))
        df['mut_pos'] = df.apply(get_pos,axis = 1)
        wt_seq = get_wt(df['mutated_sequence'][0],df['mutant'][0])
        df['target_seq'] = wt_seq
        df['PID'] = df.index
        df = df[df['mut_pos']<1023]
        split_data_path = os.path.join(data_path,dataset.split('.')[0])
        if os.path.exists(split_data_path):
            print(f'----------------->{dataset}{"-" * (60 - len(dataset))} 数据集已经划分！删除重新划分')
            shutil.rmtree(split_data_path)

        os.makedirs(split_data_path)
        fold_types = ['fold_random_5','fold_modulo_5','fold_contiguous_5']
        for fold_type in fold_types:
            for i in range(5):
                data = df[df[fold_type]==i]
                data.to_csv(os.path.join(split_data_path,f'{dataset.split('.')[0]}_{fold_type}_data_{i}.csv'),index = False)
        print(f'----------------->{dataset}{"-" * (60 - len(dataset))} 数据集已经划分！')


def split_sample_data(df_path, sample_size,fold_type,val_frac = 0.2,shuffle = True):
    # 获取符合模式的所有 CSV 文件
    #fold_types : 'fold_random_5','fold_modulo_5','fold_contiguous_5'
    # sample_size = 20,40,60,80,100
    
    df = get_data(df_path)
    if shuffle:
        df = df.sample(frac=1)

    if len(df)/5>sample_size:

        # data = df[df[fold_type]==0]
        # train_data = data.sample(n=sample_size)
        # 从 DataFrame 中随机采样
        train_data  = df[:sample_size]
    elif len(df)/5<sample_size<len(df):
        # train_data = df.sample(n=sample_size) 
        train_data  = df[:sample_size]

    else:
        print('------数据集中的样本数小于采样数,默认使用当前数据集中一半的样本用于训练------!')
        train_data = df.sample(frac = 0.5)
    test_data=df[~df.index.isin(train_data.index)]
    val_data = train_data.sample(frac=val_frac)
    train_data = train_data[~train_data.index.isin(val_data.index)]

    # 合并所有样本
    return train_data,val_data,test_data