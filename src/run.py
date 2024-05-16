import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import os
import gc
import configparser
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train import *
from utils import *
from data import *
from model import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='config file path')
    parser.add_argument('--weights', type=bool, help='config file path',default=True)
    args = parser.parse_args()
    old_stdout = sys.stdout
    name = args.config_file.split("/")[-1][:-4]
    log_file = open(f"./logs/{name}.log","w")
    sys.stdout = log_file
    
    print('###################### MultiAspectGraph ######################')

    print(args.config_file)
    params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items,train_mat,rec_i_64,rec_u_64 = data_param_prepare(args.config_file)
    
    print('Load Configuration OK, show them below')
    print('Configuration:')
    print(params)

    mag = MultiAspectGraph(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat,rec_i_64,rec_u_64,args.weights)
    # ultragcn.load_state_dict(torch.load("ultragcn_gowalla_0.pt"))
    mag = mag.to(params['device'])
    optimizer = torch.optim.Adam(mag.parameters(), lr=params['lr'])

    train(mag, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params,train_mat)

    print('END')
    sys.stdout = old_stdout
    log_file.close()
