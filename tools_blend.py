import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from sklearn.model_selection import KFold
from torch.functional import F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import Model

subjects = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26], [27, 28], [29, 30], [31, 32], [33, 34], [35, 36], [37, 38]]

h5py_dir = Path('/mnt/data/koohong/sleep_classification/data_preprocess/multi_chan_20')
PATH = 'model_save/'

class EarlyStopping:
    def __init__(self, patience=200, verbose=False, delta=0, path=None):
        assert path is not None
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        
    def __call__(self, val_loss, model, val_step):
        # Loss
        # score = -val_loss
        
        # Accuracy
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, val_step)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter > self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, val_step)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, val_step):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        if val_step < 10:
            val_step = '00000' + str(val_step)
        elif val_step < 100:
            val_step = '0000' + str(val_step)
        elif val_step < 1000:
            val_step = '000' + str(val_step)
        elif val_step < 10000:
            val_step = '00' + str(val_step)
        elif val_step < 100000:
            val_step = '0' + str(val_step)
        else:
            val_step = str(val_step)
        torch.save(model.state_dict(), self.path+'_'+val_step+'.pt')
        self.val_loss_min = val_loss


def make_save_directory(fold):
    save_dir = f"es_saved/fold_{fold}"
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
    shutil.copy("main.py", results_dir)  
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

def train_check_generate_folds(train_check_fold):
    files = list(h5py_dir.iterdir())
    
    train_check_fold = from_fold_to_idx(train_check_fold)
    train_check_files = []

    for file in files:
        fname = file.name
        file_fold = fname.split('_')[0][3:]
        file_fold = int(file_fold)
        if file_fold in train_check_fold:
            train_check_files.append(file)
        # else:
        #     raise ValueError('Invalid Fold index!')
    train_check_files.sort()
    return train_check_files

def load_folds():
    fname = '/mnt/data/koohong/sleep_classification/data_preprocess/20folds_info_xsleepnet.pkl'
    with open(fname, 'rb') as f:
        datadict = pickle.load(f)
    train_folds = datadict['train_folds']
    train_check_folds = datadict['train_check_folds']
    val_folds = datadict['val_folds']
    test_folds = datadict['test_folds']
    for fold in range(20):
        yield train_folds[fold], train_check_folds[fold], val_folds[fold], test_folds[fold]


def train_check_generate_folds(train_check_fold):
    files = list(h5py_dir.iterdir())
    
    train_check_fold = from_fold_to_idx(train_check_fold)
    train_check_files = []

    for file in files:
        fname = file.name
        file_fold = fname.split('_')[0][3:]
        file_fold = int(file_fold)
        if file_fold in train_check_fold:
            train_check_files.append(file)
        # else:
        #     raise ValueError('Invalid Fold index!')
    train_check_files.sort()
    return train_check_files

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

def train_check_load_no_seq(fold, mode='val'):
    assert mode in ['train_check']
    if mode == 'train_check':
        data_dir = '/mnt/data/koohong/sleep_classification/data_preprocess/20_no_seq_train_check_xsleepnet'
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
        

        
def make_train_test_loader(train_files, train_check_files, val_files, test_files, dataset_cache_files
                        ,batch_size
                        ):

    train_dataset_cache_file = dataset_cache_files[0]
    train_check_dataset_cache_file = dataset_cache_files[1]
    val_dataset_cache_file = dataset_cache_files[2]
    test_dataset_cache_file = dataset_cache_files[3]

    edfx_dataset_train = EDFXDataset(train_files, train_dataset_cache_file)
    edfx_dataset_train_check = EDFXDataset(train_check_files, train_check_dataset_cache_file)
    edfx_dataset_val = EDFXDataset(val_files, val_dataset_cache_file)
    edfx_dataset_test  = EDFXDataset(test_files, test_dataset_cache_file)

    train_loader = DataLoader(edfx_dataset_train
     ,batch_size=batch_size 
     ,shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    train_check_loader = DataLoader(edfx_dataset_train_check
     ,batch_size=batch_size 
     ,shuffle=False, drop_last=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(edfx_dataset_val
     ,batch_size=batch_size 
    ,shuffle=False, drop_last=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(edfx_dataset_test
     ,batch_size=batch_size 
    ,shuffle=False, drop_last=False, num_workers=8, pin_memory=True)

    return train_loader, train_check_loader, val_loader, test_loader
########### Validation Ensemble ##################    
def ensemble(outputs, device):
    B, L, five = outputs.shape
    assert five == 5
    pred = torch.zeros((B+L-1, five)).to(device)
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
####################################################

def main_train_no_telegram(name, tag, model, 
          train_loader, train_check_loader, val_loader, test_loader, train_check_len, labels_train_check, val_len, labels_val, test_len, labels_test, epochs, device, 
           batch_size,
          criterion, optimizer, scheduler, clip_norm, fold_num):
    
    # Log
    print(name, tag)
    
    cur_time = "%d%02d%02d%02d"%(time.gmtime(time.time())[0:4])
    logger = open(f"results/{name}/log/{name}-{tag}-{cur_time}.log", "w")
    #tbwriter = SummaryWriter(f"results/{name}/tb/{tag}")
    tbwriter = SummaryWriter(f'tfb_events/{name}/fold-{tag}')
    
    # Setup
    train_hist = []
    train_check_hist = []
    val_hist = []
    test_hist = []
    best_val_loss = np.inf
    best_val_acc = -1
    len_train = len(train_loader.dataset)
    len_train_check = len(train_check_loader.dataset)
    len_val   = len(val_loader.dataset)
    len_test  = len(test_loader.dataset)

    train_step = 0
    val_step = 0
    test_step = 0
    early_stopping = EarlyStopping(path='es_saved/fold_'+str(tag)+'/checkpoint')
    is_break = False
    model_path  = Path("es_saved/fold_" + str(tag))

    t_weight = 0
    tf_weight = 0
    j_weight = 0

    t_val_acc = []
    tf_val_acc = []
    j_val_acc = []
    val_acc_list = []
    smoothing_size = 20
    ##########################

    for epoch in range(0, epochs):
        time_epoch_start = time.time()
        print(f"{epoch} epoch")
        
        ### TRAIN ###
        running_loss = 0.0
        running_corrects = 0
        running_answers = 0
        i = 0
        step = 0
        for i, (raw, stft, y, name, dataslice) in enumerate(tqdm(train_loader)):           

            # Train Start
            model.train()
            labels = y.to(torch.long)
    
            time_step_start = time.time()
            
            raw = raw.to(torch.float32).to(device)
            stft = stft.to(torch.float32).to(device)
            raw = raw.unsqueeze(2)
            stft= stft.unsqueeze(2)

            labels = labels.to(device)
            B, L, C, _, _ = stft.shape
            raw_out, psd_out, out = model(raw, stft)
            raw_out = raw_out.reshape(B*L, 5)
            psd_out = psd_out.reshape(B*L, 5)
            out = out.reshape(B*L, 5)
            labels = labels.reshape(B*L, )
        
            # Loss
            raw_loss = criterion(raw_out, labels)
            psd_loss = criterion(psd_out, labels)
            out_loss = criterion(out, labels)
            
            if val_step == 0:
                loss = (1/3) * raw_loss + (1/3) * psd_loss + (1/3) * out_loss
            else:
                loss = t_weight * raw_loss + tf_weight * psd_loss + j_weight * out_loss
                # print(t_weight, tf_weight, j_weight)

            
            out = out.reshape(B, L, 5)
            labels = labels.reshape(B, L)

            preds = nn.functional.softmax(out, dim=-1)
            preds = torch.argmax(preds, dim=-1)

            # Accuracy
            corrects = (preds == labels)        
            accuracy = (torch.sum(corrects) / (stft.shape[0] * stft.shape[1]) * 100)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            # History
            running_corrects += torch.sum(corrects).item()
            running_answers += stft.shape[0] * stft.shape[1]
            running_loss     += loss.item()
            time_step = time.time() - time_step_start
            tbwriter.add_scalar(tag='Train Loss Step', scalar_value=loss.item(), global_step=train_step)
            logger.write(f"{epoch}epoch train {step}step: accuracy {accuracy:.2f}% || loss {loss:.6f} || {time_step//60}min {time_step%60:.2f}sec\n")
            logger.flush()
            step += 1
            train_step += 1 
            # scheduler.step()
       
            if train_step % 100 == 99: 
                Vrunning_out_loss = 0.0
                Vrunning_raw_loss = 0.0
                Vrunning_psd_loss = 0.0
                
                time_val_epoch_start = time.time()
                with torch.no_grad():
                    model.eval()

                    Vrunning_loss = 0.0
                    Vrunning_corrects = 0
                    Vrunning_answers = 0
                    Vrunning_corrects_debug = 0
                    Vrunning_answers_debug = 0
                    Vrunning_se_corrects = 0
                    Vrunning_raw_corrects = 0
                    Vrunning_psd_corrects = 0

                    vali_step = 0

                    v_outputs = []
                    v_raw_outputs = []
                    v_psd_outputs = []
                    v_labels_list = []
                    v_labels_debug = []

                    for i, (vtime, vstft, vy, name, dataslice) in enumerate(tqdm(val_loader)):
                        vtime = vtime.to(torch.float32).to(device)
                        vstft = vstft.to(torch.float32).to(device)
                        vtime = vtime.unsqueeze(2)
                        vstft = vstft.unsqueeze(2)
                        v_labels = vy.to(torch.long)
                        
                        time_step_start = time.time()

                        B, L, C, _, _ = vstft.shape
                        v_labels = v_labels.to(device)
                        
                        v_labels_list.append(v_labels.detach())

                        vraw_out, vpsd_out, voutput = model(vtime, vstft)
                        vraw_out = vraw_out.reshape(B*L, 5)
                        vpsd_out = vpsd_out.reshape(B*L, 5)
                        voutput = voutput.reshape(B*L, 5)
                        v_labels = v_labels.reshape(B*L, )
                        
                        v_labels_debug.append(v_labels.reshape(B, L).detach())
                        
                        vraw_loss = criterion(vraw_out, v_labels)
                        vout_loss = criterion(voutput, v_labels)
                        vpsd_loss = criterion(vpsd_out, v_labels)

                        v_loss = vout_loss

                        Vrunning_out_loss += vout_loss.item()
                        Vrunning_raw_loss += vraw_loss.item()
                        Vrunning_psd_loss += vpsd_loss.item()

                        # Accuracy
                        vraw_out = vraw_out.reshape(B, L, 5)
                        vpsd_out = vpsd_out.reshape(B, L, 5)
                        voutput = voutput.reshape(B, L, 5)
                        v_labels = v_labels.reshape(B, L)

                        v_raw_probs = nn.functional.softmax(vraw_out, dim=-1)
                        v_psd_probs = nn.functional.softmax(vpsd_out, dim=-1)
                        v_probs = nn.functional.softmax(voutput, dim=-1)

                        v_raw_preds = torch.argmax(v_raw_probs, dim=-1)
                        v_psd_preds = torch.argmax(v_psd_probs, dim=-1)

                        v_preds = torch.argmax(v_probs, dim=-1)
                        
                        v_outputs.append(v_probs.reshape(B, L, 5).detach())
                        v_raw_outputs.append(v_raw_probs.reshape(B, L, 5).detach())
                        v_psd_outputs.append(v_psd_probs.reshape(B, L, 5).detach())
                    
                        Vrunning_loss += v_loss.item()
                        time_step = time.time() - time_step_start

                        vali_step += 1
      
                        # History
                        Vrunning_corrects_debug += torch.sum((v_preds==v_labels))
                        Vrunning_answers_debug += vstft.shape[0] * vstft.shape[1]
                    
                    
                    v_outputs = torch.cat(v_outputs)
                    v_raw_outputs = torch.cat(v_raw_outputs)
                    v_psd_outputs = torch.cat(v_psd_outputs)

                    v_labels_list = torch.cat(v_labels_list)
                    
                    # Self Ensemble
                    v_pred = ensemble(v_outputs, device=device)
                    v_raw_pred = ensemble(v_raw_outputs, device=device)
                    v_psd_pred = ensemble(v_psd_outputs, device=device)

                    self_vpred = torch.argmax(v_pred, dim=-1)
                    self_vraw_pred = torch.argmax(v_raw_pred, dim=-1)
                    self_vpsd_pred = torch.argmax(v_psd_pred, dim=-1)

                    ###### multimodal ensemble ######
                    final_pred = v_pred + v_raw_pred + v_psd_pred
                    final_pred = torch.argmax(final_pred, dim=-1)
                    ##################################

                    labels_v = val_aggregate(v_labels_list)

                    v_raw_correct = (self_vraw_pred == labels_v)
                    v_psd_correct = (self_vpsd_pred == labels_v)
                    v_raw_correct = torch.sum(v_raw_correct)
                    v_psd_correct = torch.sum(v_psd_correct)

                    v_correct = (final_pred == labels_v)
                    se_correct = (self_vpred == labels_v)
                    v_correct = torch.sum(v_correct)
                    se_correct = torch.sum(se_correct)

                    Vrunning_raw_corrects += v_raw_correct
                    Vrunning_psd_corrects += v_psd_correct
                    Vrunning_corrects += v_correct
                    Vrunning_se_corrects += se_correct
                    Vrunning_answers += len(labels_v)

                    val_acc  = Vrunning_corrects / (Vrunning_answers) * 100
                    val_se_acc = Vrunning_se_corrects / (Vrunning_answers) * 100
                    val_raw_acc  = Vrunning_raw_corrects / (Vrunning_answers) * 100
                    val_psd_acc  = Vrunning_psd_corrects / (Vrunning_answers) * 100
                    val_loss = Vrunning_loss     / len_val
                    
                    ############## Smoothing Val Accuracy #########################
                    t_val_acc.append(val_raw_acc)
                    tf_val_acc.append(val_psd_acc)
                    j_val_acc.append(val_se_acc)
                    val_acc_list.append(val_acc)

                    smoothed_val_acc = torch.mean(torch.tensor(val_acc_list[-smoothing_size:]))
                    smoothed_val_raw_acc = torch.mean(torch.tensor(t_val_acc[-smoothing_size:]))
                    smoothed_val_psd_acc = torch.mean(torch.tensor(tf_val_acc[-smoothing_size:]))
                    smoothed_val_se_acc = torch.mean(torch.tensor(j_val_acc[-smoothing_size:]))

                    tbwriter.add_scalar(tag='Smoothed Val Acc Ens Avg', scalar_value=smoothed_val_acc, global_step=val_step)
                    tbwriter.add_scalar(tag='Smoothed Val Acc SE Avg', scalar_value=smoothed_val_se_acc, global_step=val_step)
                    tbwriter.add_scalar(tag='Smoothed Val Acc SE Raw Avg', scalar_value=smoothed_val_raw_acc, global_step=val_step)
                    tbwriter.add_scalar(tag='Smoothed Val Acc SE PSD Avg', scalar_value=smoothed_val_psd_acc, global_step=val_step)
                    ###############################################################

                    ################# Multimodal Acc ##############
                    time_val_epoch = time.time() - time_val_epoch_start
                    print(f"Val : acc {val_acc:.2f}  loss {val_loss:.6f}  {time_val_epoch//60}min {time_val_epoch%60:.2f}sec")
                    print(f"Smoothed Val : acc {smoothed_val_acc:.2f}")
                    ################# Self Ensemble Acc ###########
                    print(f"Self Ens Val : acc {val_se_acc:.2f}  loss {val_loss:.6f}  {time_val_epoch//60}min {time_val_epoch%60:.2f}sec")
                    print(f"Smoothed Self Ens Val : acc {smoothed_val_se_acc:.2f}")
                    # ################ Debug #######################
                    val_acc_debug = Vrunning_corrects_debug / (Vrunning_answers_debug) * 100
                    print(f'Val Debug: acc {val_acc_debug:.2f}')
                    # ##############################################
                    logger.write(f"Val : acc {val_acc:.2f}  loss {val_loss:.6f}  {time_val_epoch//60}min {time_val_epoch%60:.2f}sec\n")
                    logger.flush()
                    val_hist.append({"val_acc":val_acc, "val_loss":val_loss})

                    val_step += 1

                    tbwriter.add_scalar(tag='Val Loss Avg', scalar_value=val_loss, global_step=val_step)
                    tbwriter.add_scalar(tag='Val Acc Ens Avg', scalar_value=val_acc, global_step=val_step)
                    tbwriter.add_scalar(tag='Val Acc SE Avg', scalar_value=val_se_acc, global_step=val_step)
                    tbwriter.add_scalar(tag='Val Acc SE Raw Avg', scalar_value=val_raw_acc, global_step=val_step)
                    tbwriter.add_scalar(tag='Val Acc SE PSD Avg', scalar_value=val_psd_acc, global_step=val_step)
                    tbwriter.add_scalar(tag='Val Acc noEns Avg', scalar_value=val_acc_debug, global_step=val_step)

                    # Weighting
                    inverse_joint_acc = 1 / (smoothed_val_se_acc / 100)
                    inverse_raw_acc = 1 / (smoothed_val_raw_acc / 100)
                    inverse_psd_acc = 1 / (smoothed_val_psd_acc / 100)
     
                    total_inverse_acc = inverse_joint_acc + inverse_raw_acc + inverse_psd_acc
                    t_weight = inverse_raw_acc / total_inverse_acc
                    tf_weight = inverse_psd_acc / total_inverse_acc
                    j_weight = inverse_joint_acc / total_inverse_acc
        
                    tbwriter.add_scalar(tag='time loss weight', scalar_value=t_weight, global_step=val_step)
                    tbwriter.add_scalar(tag='tf loss weight', scalar_value=tf_weight, global_step=val_step)
                    tbwriter.add_scalar(tag='joint loss weight', scalar_value=j_weight, global_step=val_step)

                    if best_val_acc < smoothed_val_acc:
                        best_val_acc = smoothed_val_acc
                    early_stopping(smoothed_val_acc, model, val_step)

             
                if early_stopping.early_stop:
                    print('Early Stopped!')
                    is_break = True
                    
      
            if is_break:
                time_test_epoch_start = time.time()
                with torch.no_grad():
                    ckpt_path_list = list((model_path.iterdir()))
                    ckpt_path_list.sort()
                    ckpt_path = ckpt_path_list[-1]
                    testing_model = Model(device=device).to(device)
                    testing_model.load_state_dict(torch.load(ckpt_path))
                    testing_model.eval()
                    Trunning_loss = 0.0
                    Trunning_corrects = 0
                    Trunning_se_corrects = 0
                    Trunning_answers = 0
                    
                    Trunning_corrects_debug = 0
                    Trunning_answers_debug = 0
                    
                    outputs = []
                    raw_outputs = []
                    tf_outputs = []
                    t_labels_list = []
                    t_labels_debug = []
                    for i, (ttime, tstft, ty, tname, t_dataslice) in enumerate(tqdm(test_loader)):
                        ttime = ttime.to(torch.float32).to(device)
                        tstft = tstft.to(torch.float32).to(device)
                        ttime = ttime.unsqueeze(2)
                        tstft = tstft.unsqueeze(2)
                        t_labels = ty.to(torch.long)
                    
                        time_step_start = time.time()

                        B, L, C, _, _ = tstft.shape
                        t_labels = t_labels.to(device)
                        
                        t_labels_list.append(t_labels.detach())

                        traw, ttf, toutput = testing_model(ttime, tstft)  # model output

                        toutput = toutput.reshape(B*L, 5)

                        t_labels = t_labels.reshape(B*L, )
                        
                        t_labels_debug.append(t_labels.reshape(B, L).detach())

                        t_loss = criterion(toutput, t_labels)

                        toutput = toutput.reshape(B, L, 5)
                        traw = traw.reshape(B, L, 5)
                        ttf = ttf.reshape(B, L, 5)
                        t_labels = t_labels.reshape(B, L)

                        t_probs = nn.functional.softmax(toutput, dim=-1)
                        t_preds = torch.argmax(t_probs, dim=-1)
                        t_raw_probs = nn.functional.softmax(traw, dim=-1)
                        t_raw_preds = torch.argmax(t_raw_probs, dim=-1)
                        t_tf_probs = nn.functional.softmax(ttf, dim=-1)
                        t_tf_preds = torch.argmax(t_tf_probs, dim=-1)
                        
                        outputs.append(t_probs.reshape(B, L, 5).detach())
                        raw_outputs.append(t_raw_probs.reshape(B, L, 5).detach())
                        tf_outputs.append(t_tf_probs.reshape(B, L, 5).detach())

                        Trunning_loss += t_loss.item()
                        time_step = time.time() - time_step_start
                        logger.write(f"{epoch}epoch val {step}step: loss {t_loss:.6f} || {time_step//60}min {time_step%60:.2f}sec\n")
                        logger.flush()
                        step += 1
                            
                        # History
                        Trunning_corrects_debug += torch.sum((t_preds==t_labels))
                        Trunning_answers_debug += tstft.shape[0] * tstft.shape[1]
                    
                    # Ensemble
                    outputs = torch.cat(outputs)
                    raw_outputs = torch.cat(raw_outputs)
                    tf_outputs = torch.cat(tf_outputs)
                    t_labels_list = torch.cat(t_labels_list)   
                    
                    # Self Ensemble
                    t_pred = ensemble(outputs, device=device)
                    traw_pred = ensemble(raw_outputs, device=device)
                    ttf_pred = ensemble(tf_outputs, device=device)

                    self_tpred = torch.argmax(t_pred, dim=-1)

                    # Multimodal Ensemble
                    final_tpred = t_pred + traw_pred + ttf_pred
                    final_tpred = torch.argmax(final_tpred, dim=-1)

                    labels_t = val_aggregate(t_labels_list)

                    t_correct = (final_tpred == labels_t)
                    se_correct = (self_tpred == labels_t)

                    t_correct = torch.sum(t_correct)
                    se_correct = torch.sum(se_correct)

                    Trunning_corrects += t_correct
                    Trunning_se_corrects += se_correct
                    Trunning_answers += len(labels_t)
                    
                    test_acc  = Trunning_corrects / (Trunning_answers) * 100
                    test_se_acc = Trunning_se_corrects / (Trunning_answers) * 100
                    test_loss = Trunning_loss     / len_test

                    time_test_epoch = time.time() - time_test_epoch_start
                    print(f"Test : acc {test_acc:.2f}  loss {test_loss:.6f}  {time_test_epoch//60}min {time_test_epoch%60:.2f}sec")
                    print(f"SE Test : acc {test_se_acc:.2f}  loss {test_loss:.6f}  {time_test_epoch//60}min {time_test_epoch%60:.2f}sec")
                    # ################ Debug #######################
                    test_acc_debug = Trunning_corrects_debug / (Trunning_answers_debug) * 100
                    print(f'Test Debug: acc {test_acc_debug:.2f}')
                    # ##############################################
                    logger.write(f"Test : acc {test_acc:.2f}  loss {test_loss:.6f}  {time_test_epoch//60}min {time_test_epoch%60:.2f}sec\n")
                    logger.flush()
                    test_hist.append({"test_acc":test_acc, "test_loss":test_loss})
                    tbwriter.add_scalar(tag='Test Loss Avg', scalar_value=test_loss, global_step=test_step)
                    tbwriter.add_scalar(tag='Test Acc Ens Avg', scalar_value=test_acc, global_step=test_step)
                    tbwriter.add_scalar(tag='Test Acc SE Avg', scalar_value=test_se_acc, global_step=test_step)
                    tbwriter.add_scalar(tag='Test Acc noEns Avg', scalar_value=test_acc_debug, global_step=test_step)
                return {"train_hist":train_hist, "val_hist":val_hist, "best_val_acc":best_val_acc}

        ### Epoch Log ###
        time_epoch = time.time() - time_epoch_start
        
        train_acc  = running_corrects / (running_answers) * 100
        train_loss = running_loss     / len_train
        tbwriter.add_scalar('Train ACC AVG', scalar_value=train_acc, global_step=epoch)
        tbwriter.add_scalar('Train Loss AVG', scalar_value=train_loss, global_step=epoch)
        
        print(f"Train : acc {train_acc:.2f}  loss {train_loss:.6f}  {time_epoch//60}min {time_epoch%60:.2f}sec")
        logger.write(f"Train : acc {train_acc:.2f}  loss {train_loss:.6f}  {time_epoch//60}min {time_epoch%60:.2f}sec\n")
        logger.flush()
        
        train_hist.append({"train_acc":train_acc, "train_loss":train_loss})
    
    logger.close()
    
    return {"train_hist":train_hist, "val_hist":val_hist, "best_val_acc":best_val_acc}
