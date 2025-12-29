"""
#Repurposed from the GLAM paper https://github.com/sawlani/GLAM
#Date: 01 Jan 2023
df['GroupId'] = df['ParameterList'].str.extract('(blk\_[-]?\d+)', expand=False)

"""
# the absolute path of the Logs2Graph project
root_path = r'/home/ubuntu/bsc/BootDet/Log2Graph'

import warnings
warnings.filterwarnings('ignore')

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from types import SimpleNamespace
from tqdm import tqdm

from DataLoader import create_loaders, MeanTrainer, DiGCN

##--------------------------------------------
##Step 1. first clear all files under the /processed/~ directory
##--------------------------------------------

def prepare_experiment(dataset_name):
    folder = f'{root_path}/Data/{dataset_name}/processed'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            

    folder = f'{root_path}/Data/{dataset_name}/Raw'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
    ##--------------------------------------------
    ##Step 2. copy all files from a directory to another
    ##--------------------------------------------       
    
    # path to source directory
    src_dir = f'{root_path}/Data/{dataset_name}/Graph/Raw/'
    
    # path to destination directory
    dest_dir = f'{root_path}/Data/{dataset_name}/Raw/'
    
    # getting all the files in the source directory
    my_files = os.listdir(src_dir)
    
    for file_name in my_files:
        # print(file_name)
        # print(type(dest_dir))
        src_file_name = src_dir + file_name
        dest_file_name = dest_dir + file_name
        shutil.copy(src_file_name, dest_file_name)
        
        
            
##--------------------------------------------
##Step 3. define a function to run experiments
##--------------------------------------------          
            
def run_experiment(
    data, # dataset to use
    data_seed=1213, 
    alpha=1.0, 
    beta=0.0,
    epochs=150, 
    model_seed=0, 
    num_layers=1, 
    device=0,
    aggregation='Mean',
    bias=False,
    hidden_dim=64,
    lr=0.1,
    weight_decay=1e-5,
    batch = 64,
    hpo = False
    ):
    device = torch.device('cuda:' + str(device)) if torch.cuda.is_available() else torch.device('cpu')
    
    # =============================================================================
    # Step1. load data using predefined script dataloader.py
    # we should define this function by ourself
    # =============================================================================
    
    train_loader, eval_loader, val_loader, test_loader, num_features, train_dataset, val_dataset, test_dataset, raw_dataset = create_loaders(data_name=data,
                                                                                                                batch_size=batch,
                                                                                                                dense=False,
                                                                                                                data_seed=data_seed)
    
    ##----set seeds for cuda----
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
        
    # =============================================================================
    # Step2. train a GIN model with given parameters
    # =============================================================================
    
    ##----setting parameters----
    model = DiGCN(nfeat = num_features, nhid=hidden_dim, nlayer=num_layers, bias=bias)
    
    ##----important paramter 0----##
    ##the learning rate, weight decay hyperparameter are given here
    ##In GLAM they use SGD, however, we will use Adam in our paper
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay) 
    
    if aggregation=='Mean':
        trainer = MeanTrainer(
            model=model,
            optimizer=optimizer,
            alpha=alpha,
            beta=beta,
            device=device
            )
    
    epochinfo = []
    for epoch in tqdm(range(epochs+1)):
        train_loss = trainer.train(train_loader)
        val_loss   = trainer.validate(val_loader)
        ap, roc_auc, dists, labels = trainer.test(test_loader=eval_loader)
        TEMP = SimpleNamespace()
        TEMP.epoch_no   = epoch
        TEMP.train_loss = train_loss
        TEMP.val_loss   = val_loss
        TEMP.ap         = ap
        TEMP.roc_auc    = roc_auc
        TEMP.dists      = dists
        TEMP.labels     = labels
        epochinfo.append(TEMP)
        
    best_idx = np.argmin([e.val_loss for e in epochinfo[1:]]) + 1
    best = epochinfo[best_idx]
    ##----record the best epoch's information----
    important_epoch_info = {}
    important_epoch_info['svdd'] = epochinfo[best_idx]
    important_epoch_info['last'] = epochinfo[-1]
    
    if hpo:
        return important_epoch_info, epochinfo, train_dataset, test_dataset, raw_dataset
    else:
        final_ap, final_roc_auc, final_dists, final_labels = trainer.test(test_loader=test_loader)
        print(f'\nBest epoch (by val_loss): {best.epoch_no}')
        print(f'  train_loss = {best.train_loss:.6f}')
        print(f'  val_loss   = {best.val_loss:.6f}')
        print(f'  ROC-AUC    = {best.roc_auc:.3f}, AP = {best.ap:.3f}')
        return final_ap, final_roc_auc, final_dists, final_labels
