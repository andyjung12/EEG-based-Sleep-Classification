import os
import shutil
import numpy as np
import time
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from sklearn.model_selection import KFold
from torch.functional import F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
from model import Model

subjects = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26], [27, 28], [29, 30], [31, 32], [33, 34], [35, 36], [37, 38]]

h5py_dir = Path('/mnt/data/koohong/sleep_classification/data_preprocess/multi_chan_20')
PATH = 'model_save/'

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=200, verbose=False, delta=0, path='es_saved/checkpoint'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, val_step):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, val_step)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, val_step)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, val_step):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path+'_'+str(val_step)+'.pt')
        self.val_loss_min = val_loss


def make_save_directory(name):
    save_dir = f"es_saved/fold_{name}"
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass
    print(f"'{save_dir}' is created!")
    
def make_results_directory(name, module):
    results_dir = f"results/{name}"
    try:
        os.makedirs(results_dir)
        os.makedirs(results_dir+"/models")
        os.makedirs(results_dir+"/log")
        os.makedirs(results_dir+"/tb")
    except FileExistsError:
        pass  
    shutil.copy("main_f1.py", results_dir)  
    print(f"'{results_dir}' is created!")

def from_fold_to_idx(fold_list):
    output = []
    for fold in fold_list:
        output += subjects[fold]
    output.sort()
    return output

def generate_folds(train_fold, val_fold, test_fold):
    files = list(h5py_dir.iterdir())
    train_fold = from_fold_to_idx(train_fold)
    val_fold = from_fold_to_idx(val_fold)
    test_fold = from_fold_to_idx(test_fold)
    train_files = []
    val_files = []
    test_files = []
    for file in files:
        fname = file.name
        file_fold = fname.split('_')[0][3:]
        file_fold = int(file_fold)
        if file_fold in train_fold:
            train_files.append(file)
        elif file_fold in val_fold:
            val_files.append(file)
        elif file_fold in test_fold:
            test_files.append(file)
        else:
            raise ValueError('Invalid Fold index!')
    train_files.sort()
    val_files.sort()
    test_files.sort()
    return train_files, val_files, test_files

def load_folds():
    fname = '/mnt/data/koohong/sleep_classification/data_preprocess/20folds_info_xsleepnet.pkl'
    with open(fname, 'rb') as f:
        datadict = pickle.load(f)
    train_folds = datadict['train_folds']
    val_folds = datadict['val_folds']
    test_folds = datadict['test_folds']
    for fold in range(20):
        yield train_folds[fold], val_folds[fold], test_folds[fold]


def load_no_seq(fold, mode='val'):
    assert mode in ['val', 'test']
    if mode == 'val':
        data_dir = '/mnt/data/koohong/sleep_classification/data_preprocess/20_no_seq_val_xsleepnet'
    else:
        data_dir = '/mnt/data/koohong/sleep_classification/data_preprocess/20_no_seq_test_xsleepnet'
    with open(data_dir + '/' + f'{fold}-th.pkl', 'rb') as f:
        datadict = pickle.load(f)
    y_ens = datadict['y_ens']
    len_test = datadict['num_test']

    return y_ens, len_test


"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import pathlib
import pickle
import random

import h5py
import numpy as np
from torch.utils.data import Dataset


class EDFXDataset(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.

    Args:
        root (pathlib.Path): Path to the dataset.
        dataset_cache_file (pathlib.Path). A file in which to cache dataset
            information for faster load times. Default: dataset_cache.pkl.
        num_cols (tuple(int), optional): if provided, only slices with the desired
            number of columns will be considered.
    """

    def __init__(
        self,
        files,
        dataset_cache_file
        
    ):
        self.dataset_cache_file = dataset_cache_file
        self.L = 29
        self.examples = []

        if self.dataset_cache_file.exists():
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
                
        else:
            dataset_cache = {}

        if dataset_cache.get(dataset_cache_file) is None:
            #files = list(pathlib.Path(root).iterdir())
            for fname in files:
                with h5py.File(fname, "r") as hf:
                    num_slices = hf['data']['volume'][()]#hf["volume"].item()

                self.examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices-28)
                ]


            dataset_cache[dataset_cache_file] = self.examples
            logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
            with open(self.dataset_cache_file, "wb") as f:
                pickle.dump(dataset_cache, f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[dataset_cache_file]


    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        fname, dataslice = self.examples[i]

        with h5py.File(fname, "r") as hf:
            stft = hf['data']['stft'][dataslice:dataslice+self.L]
            raw = hf['data']['time'][dataslice:dataslice+self.L]
            y = hf['data']['label'][dataslice:dataslice+self.L].astype(np.int16)
            stft = stft[:, 0, :, :]
            raw = raw[:, 0, :]
        return raw, stft, y, fname.name, dataslice
        

        
def make_train_test_loader(train_files, val_files, test_files, dataset_cache_files
                        ,batch_size
                        ):

    train_dataset_cache_file = dataset_cache_files[0]
    val_dataset_cache_file = dataset_cache_files[1]
    test_dataset_cache_file = dataset_cache_files[2]

    edfx_dataset_train = EDFXDataset(train_files, train_dataset_cache_file)
    edfx_dataset_val = EDFXDataset(val_files, val_dataset_cache_file)
    edfx_dataset_test  = EDFXDataset(test_files, test_dataset_cache_file)

    train_loader = DataLoader(edfx_dataset_train
     ,batch_size=batch_size 
     ,shuffle=True, drop_last=True, num_workers=12)
    val_loader = DataLoader(edfx_dataset_val
     ,batch_size=batch_size 
    ,shuffle=False, drop_last=False, num_workers=12)
    test_loader = DataLoader(edfx_dataset_test
     ,batch_size=batch_size 
    ,shuffle=False, drop_last=False, num_workers=12)
    return train_loader, val_loader, test_loader
    
def ensemble(outputs):
    B, L, five = outputs.shape
    assert five == 5
    pred = torch.zeros((B+L-1, five))
    for i in range(B):
        pred[i:i+L, :] += torch.log(outputs[i,:,:]+1.0e-12)
    # pred = torch.argmax(pred, dim=-1)
    # EDFX-78
    # return pred+1
    # SHHS
    return pred

def val_aggregate(target):
    B, L = target.shape
    output = torch.cat([target[0, :L//2], target[:, L//2], target[-1, L//2+1:]], dim=0)
    return output
    
def main_train_no_telegram(name, tag, model, 
          train_loader, val_loader, test_loader, val_len, labels_val, test_len, labels_test, epochs, device, 
           batch_size,
          criterion, optimizer, scheduler, clip_norm, fold_num):
    
    # Log
    print(name, tag)
    cur_time = "%d%02d%02d%02d"%(time.gmtime(time.time())[0:4])
    
    # Setup
    len_test  = len(test_loader.dataset)
    model_path  = Path("es_saved/fold_" + str(tag))

    
    outputs = []
    time_outputs = []
    tf_outputs = []
    t_labels_list = []
    t_labels_debug = []
    no_ens_outputs = []

    with torch.no_grad():
        ckpt_path_list = list((model_path.iterdir()))
        ckpt_path_list.sort()
        ckpt_path = ckpt_path_list[-1]
        print(ckpt_path)
        testing_model = Model(device=device).to(device)
        testing_model.load_state_dict(torch.load(ckpt_path))
        testing_model.eval()
        Trunning_loss = 0.0
        Trunning_corrects = 0
        Trunning_answers = 0

        Trunning_time_corrects = 0
        Trunning_tf_corrects = 0
        Trunning_se_corrects = 0
        
        
        Trunning_corrects_debug = 0
        Trunning_answers_debug = 0
        
        for i, (ttime, tstft, ty, tname, t_dataslice) in enumerate(tqdm(test_loader)):
            ttime = ttime.to(torch.float32).to(device)
            tstft = tstft.to(torch.float32).to(device)
            ttime = ttime.unsqueeze(2)
            tstft = tstft.unsqueeze(2)
            t_labels = ty.to(torch.long)

            B, L, C, _, _ = tstft.shape
            t_labels = t_labels.to(device)
            
            t_labels_list.append(t_labels.detach().cpu())

            time_output, tf_output, toutput = testing_model(ttime, tstft)  # feed forward

            toutput = toutput.reshape(B*L, 5)
            t_labels = t_labels.reshape(B*L, )
                        
            t_labels_debug.append(t_labels.reshape(-1).detach().cpu().numpy())

            t_loss = criterion(toutput, t_labels)

            toutput = toutput.reshape(B, L, 5)
            t_labels = t_labels.reshape(B, L)

            t_probs = nn.functional.softmax(toutput, dim=-1)
            time_probs = nn.functional.softmax(time_output, dim=-1)
            tf_probs = nn.functional.softmax(tf_output, dim=-1)

            t_preds = torch.argmax(t_probs, dim=-1)
            no_ens_outputs.append(t_preds.reshape(-1).detach().cpu())
            time_preds = torch.argmax(time_probs, dim=-1)
            tf_preds = torch.argmax(tf_probs, dim=-1)
            
            outputs.append(t_probs.reshape(B, L, 5).detach().cpu())
            time_outputs.append(time_probs.reshape(B, L, 5).detach().cpu())
            tf_outputs.append(tf_probs.reshape(B, L, 5).detach().cpu())

            Trunning_loss += t_loss.item()
                
            # History
            Trunning_corrects_debug += torch.sum((t_preds==t_labels))
            Trunning_answers_debug += tstft.shape[0] * tstft.shape[1]

        t_pred_list = []
        time_pred_list = []
        tf_pred_list = []
        labels_t_list = []

        outputs = torch.cat(outputs)
        time_outputs = torch.cat(time_outputs)
        tf_outputs = torch.cat(tf_outputs)
        t_labels_list = torch.cat(t_labels_list)

        #Self Ensemble
        t_pred = ensemble(outputs)
        t_pred_list.append(torch.argmax(t_pred, dim=-1))
        time_pred = ensemble(time_outputs)
        time_pred_list.append(time_pred)
        tf_pred = ensemble(tf_outputs)
        tf_pred_list.append(tf_pred)

        self_tpred = torch.argmax(t_pred, dim=-1)

        # Multimodal Ensemble
        final_tpred = t_pred + time_pred + tf_pred
        final_tpred = torch.argmax(final_tpred, dim=-1)
        
        labels_t = val_aggregate(t_labels_list)
        labels_t_list.append(labels_t)

        time_pred = torch.argmax(time_pred, dim=-1)
        tf_pred = torch.argmax(tf_pred, dim=-1)

 
        t_correct = (final_tpred == labels_t)
        time_correct = (time_pred == labels_t)
        tf_correct = (tf_pred == labels_t)
        se_correct = (self_tpred == labels_t)

        t_correct = torch.sum(t_correct)
        time_correct = torch.sum(time_correct)
        tf_correct = torch.sum(tf_correct)
        se_correct = torch.sum(se_correct)

        Trunning_corrects += t_correct
        Trunning_se_corrects += se_correct
        Trunning_time_corrects += time_correct
        Trunning_tf_corrects += tf_correct

        Trunning_answers += len(labels_t)
        test_acc  = Trunning_corrects / (Trunning_answers) * 100
        test_se_acc  = Trunning_se_corrects / (Trunning_answers) * 100
        test_time_acc = Trunning_time_corrects / (Trunning_answers) * 100
        test_tf_acc = Trunning_tf_corrects / (Trunning_answers) * 100

        test_acc_debug = Trunning_corrects_debug / Trunning_answers_debug * 100

        print("Test Acc debug : ", test_acc_debug)
        print("Test SE Acc : ", test_se_acc)
        print("Test Acc : ", test_acc)
        print("Test Time Acc : ", test_time_acc)
        print("Test TF Acc : ", test_tf_acc)
        
        ################### Kappa & F1###############
        t_pred_list = np.concatenate(t_pred_list)
        labels_t_list = np.concatenate(labels_t_list)
        se_f1 = f1_score(labels_t_list, t_pred_list, average=None)
        se_kappa_score = cohen_kappa_score(labels_t_list, t_pred_list)
        with open('pred.npy', 'wb') as f:
            np.save(f, np.array(t_pred_list))
        with open('label.npy', 'wb') as f:
            np.save(f, np.array(labels_t_list))

        f1 = f1_score(labels_t_list, final_tpred, average=None)
        kappa_score = cohen_kappa_score(labels_t_list, final_tpred)

        no_ens_pred_list = np.concatenate(no_ens_outputs)
        no_ens_label_list = np.concatenate(t_labels_debug)
        no_ens_f1 =  f1_score(no_ens_label_list, no_ens_pred_list, average=None)
        no_ens_kappa = cohen_kappa_score(no_ens_label_list, no_ens_pred_list)

        print("F1 score : ", f1)
        print("Cohen's Kappa: ", kappa_score)
        print("No Ens F1 score : ", no_ens_f1)
        print("No Ens Cohen's Kappa: ", no_ens_kappa)
        print("SE F1 score : ", se_f1)
        print("SE Cohen's Kappa: ", se_kappa_score)
        return f1, kappa_score, se_f1, se_kappa_score, no_ens_f1, no_ens_kappa, t_pred_list, labels_t_list
        ##############################################
