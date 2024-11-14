from pathlib import Path

import pickle
import argparse
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torch import nn
from tools_f1 import make_save_directory, make_results_directory, generate_folds, make_train_test_loader, main_train_no_telegram, load_folds, load_no_seq
from model import Model
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='many-to-many')
parser.add_argument('--device', default="0")
parser.add_argument('--fold', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--decay', type=float, default=0.005)
parser.add_argument('--eps', type=float, default=1e-7)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=1)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--clip_val', type=float, default=1.0)

args = parser.parse_args()

name = args.name + '_GPU:' + args.device + '_lr:' + str(args.lr) + '_decay:' + str(args.decay) + '_step_size:' + str(args.step_size) + '_gamma:' + str(args.gamma) + '_clip_val:' + str(args.clip_val)

if args.fold != -1:
    name = name + '_fold:' + str(args.fold)

make_results_directory(name, "specTNTT")
make_save_directory(name)

## GPU setting##
device = torch.device('cuda:'+args.device)

## optimizer parameter setting ##
# class_weight = [0.3, 0.4, 0.3, 0.2, 0.3]
# class_weight = torch.FloatTensor(class_weight).to(device)
Criterion = nn.CrossEntropyLoss
Optimizer = torch.optim.Adam
#init_lr = 0.0001
#epochs = 30
#batch_size = 32
init_lr = args.lr
epochs = args.max_epochs
batch_size = args.batch_size
eps = args.eps
decay = args.decay
#scale = args.scale
step_size = args.step_size
gamma = args.gamma
clip_norm = args.clip_val




generator = load_folds()   

histories = []
test_acc = []
f1_scores = []
cohens_kappa = []
no_ens_f1_scores = []
no_ens_cohens_kappa = []
se_f1_scores = []
se_cohens_kappa = []
prediction_list = np.array([])
true_label_list = np.array([])

for fold, (train_fold, val_fold, test_fold) in enumerate(generator):
    # if fold != 15:
    #     continue
    print(f'It is {fold}-th fold.')
    y_ens_val, len_val = load_no_seq(fold, mode='val')
    y_ens_test, len_test = load_no_seq(fold, mode='test')
    train_files, val_files, test_files = generate_folds(train_fold, val_fold, test_fold)
    dataset_cache_files = [
        Path('caches/train_data_cache_'+str(fold) + '.pkl'),
        Path('caches/val_data_cache_'+str(fold)+'.pkl'),
        Path('caches/test_data_cache_'+str(fold)+'.pkl')
    ]
    train_loader, val_loader, test_loader = make_train_test_loader(train_files, val_files, test_files, dataset_cache_files
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
                    val_loader   = val_loader,
                    test_loader  = test_loader, 
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
                    fold_num     =fold)
    f1, kappa_score, se_f1, se_kappa_score, no_ens_f1, no_ens_kappa, t_pred_list, labels_t_list = hist
    f1_scores.append(f1)
    cohens_kappa.append(kappa_score)
    no_ens_f1_scores.append(no_ens_f1)
    no_ens_cohens_kappa.append(no_ens_kappa)
    se_f1_scores.append(se_f1)
    se_cohens_kappa.append(se_kappa_score)
    true_label_list = np.append(true_label_list, [labels_t_list])
    prediction_list = np.append(prediction_list, [t_pred_list])

    # 중간 결과 저장
#     with open(f"results/{name}/tb/history{fold}.pkl", "wb") as f:
#         pickle.dump(hist, f)
#     # 중간 결과 수집
#     histories.append(hist)
    
# # 최종 결과 저장
# with open(f"results/{name}/tb/histories.pkl", "wb") as f:
#     pickle.dump(histories, f)

# print('Average Test Score : ', np.array(test_acc).mean())
confusion_mat = confusion_matrix(true_label_list, prediction_list)
print(confusion_mat)
print('Average No Ens F1 Score : ', np.mean(np.array(no_ens_f1_scores), axis=0))
print('Average No Ens Kappa : ', np.array(no_ens_cohens_kappa).mean())
print('Average SE F1 Score : ', np.mean(np.array(se_f1_scores), axis=0))
print('Average SE Kappa : ', np.array(se_cohens_kappa).mean())
print('Average F1 Score : ', np.mean(np.array(f1_scores), axis=0))
print('Average Kappa : ', np.array(cohens_kappa).mean())
