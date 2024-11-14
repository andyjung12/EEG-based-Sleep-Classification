from pathlib import Path

import pickle
import argparse
import torch
import numpy as np

from torch import nn
from tools_blend import make_save_directory, make_results_directory, generate_folds, train_check_generate_folds, make_train_test_loader, main_train_no_telegram, load_folds, load_no_seq, train_check_load_no_seq
from model import Model
import os
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='many-to-many')
parser.add_argument('--device', default="0")
parser.add_argument('--fold', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--decay', type=float, default=0)
parser.add_argument('--eps', type=float, default=1e-7)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=1)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--clip_val', type=float, default=1.0)

args = parser.parse_args()

name = args.name + '_GPU:' + args.device + '_lr:' + str(args.lr) + '_decay:' + str(args.decay) + '_step_size:' + str(args.step_size) + '_gamma:' + str(args.gamma) + '_clip_val:' + str(args.clip_val)

if args.fold != -1:
    name = name + '_fold:' + str(args.fold)

make_results_directory(name, "LUNA")

## GPU setting##
device = torch.device('cuda:'+args.device)


## optimizer parameter setting ##
Criterion = nn.CrossEntropyLoss
Optimizer = torch.optim.Adam
init_lr = args.lr
epochs = args.max_epochs
batch_size = args.batch_size
decay = args.decay
step_size = args.step_size
gamma = args.gamma
clip_norm = args.clip_val
eps = args.eps

histories = []
generator = load_folds()  
for fold, (train_fold, train_check_fold, val_fold, test_fold) in enumerate(generator):
    # if fold < 1:
    #     continue
    # torch.manual_seed(43)
    # torch.cuda.manual_seed_all(43)
    # np.random.seed(43)
    # random.seed(43)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    print(f'It is {fold}-th fold.')
    make_save_directory(fold)
    y_ens_train_check, len_train_check = train_check_load_no_seq(fold, mode='train_check')
    y_ens_val, len_val = load_no_seq(fold, mode='val')
    y_ens_test, len_test = load_no_seq(fold, mode='test')
    train_files, val_files, test_files = generate_folds(train_fold, val_fold, test_fold)
    train_check_files = train_check_generate_folds(train_check_fold)
    dataset_cache_files = [
        Path('caches/train_data_cache_'+str(fold) + '.pkl'),
        Path('caches/train_check_data_cache_'+str(fold) + '.pkl'),
        Path('caches/val_data_cache_'+str(fold)+'.pkl'),
        Path('caches/test_data_cache_'+str(fold)+'.pkl')
    ]
    train_loader, train_check_loader, val_loader, test_loader = make_train_test_loader(train_files, train_check_files, val_files, test_files, dataset_cache_files
                        ,batch_size)
 
    model = Model(device=device).to(torch.float32).to(device)

    optimizer = Optimizer(model.parameters(), lr=init_lr, weight_decay=decay, eps=eps)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = Criterion()


    hist = main_train_no_telegram(name         = name, 
                      tag          = fold, 
                      model        = model, 
                      # 
                      train_loader = train_loader, 
                      train_check_loader = train_check_loader,
                      val_loader   = val_loader,
                      test_loader  = test_loader, 

                      train_check_len = len_train_check,
                      labels_train_check = y_ens_train_check,
                      
                      val_len      = len_val,
                      labels_val   = y_ens_val,
                      
                      test_len     = len_test,
                      labels_test  = y_ens_test,
                      
                      batch_size   = batch_size,
                      epochs       = epochs,
                      device       = device,
                      # 
                      criterion    = criterion, 
                      optimizer    = optimizer,
                      scheduler    = scheduler,
                      clip_norm    = clip_norm,
                      fold_num     = fold)
    
