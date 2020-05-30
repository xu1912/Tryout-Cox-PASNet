#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
dtype = torch.FloatTensor

MiRNA_Nodes = 1525
Hidden_Nodes_1 = 800
Hidden_Nodes_2 = 400
Hidden_Nodes_3 = 100
Hidden_Nodes_4 = 100
Hidden_Nodes_5 = 100
Hidden_Nodes_6 = 100
Hidden_Nodes_7 = 100
Hidden_Nodes_8 = 50
Out_Nodes = 30
Initial_Learning_Rate = [0.03, 0.01, 0.001, 0.00075]
L2_Lambda = [0.1, 0.01, 0.005, 0.001]
num_epochs = 600
Num_EPOCHS = 2000
Dropout_Rate = [0.8, 0.8, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
patience = 20
x_train, ytime_train, yevent_train, age_train, cstage_train, hgrade_train, race_black_train, race_white_train = load_data("C:/Users/jld_c/Desktop/Interesting papers/DL_cox/Data/autoencoder_test1/train.csv", dtype)
x_valid, ytime_valid, yevent_valid, age_valid, cstage_valid, hgrade_valid, race_black_valid, race_white_valid = load_data("C:/Users/jld_c/Desktop/Interesting papers/DL_cox/Data/autoencoder_test1/validation.csv", dtype)
x_test, ytime_test, yevent_test, age_test, cstage_test, hgrade_test, race_black_test, race_white_test = load_data("C:/Users/jld_c/Desktop/Interesting papers/DL_cox/Data/autoencoder_test1/test.csv", dtype)
opt_l2_loss = 0
opt_lr_loss = 0
opt_loss = torch.Tensor([float("Inf")])
if torch.cuda.is_available():
    opt_loss = opt_loss.cuda()
opt_c_index_va = 0
opt_c_index_tr = 0
for l2 in L2_Lambda:
    for lr in Initial_Learning_Rate:
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        loss_train, loss_valid, c_index_tr, c_index_va = trainCoxPASNet(x_train, age_train, cstage_train, hgrade_train, race_black_train, race_white_train, ytime_train, yevent_train,                                                                 x_valid, age_valid, cstage_valid, hgrade_valid, race_black_valid, race_white_valid, ytime_valid, yevent_valid,                                                                 MiRNA_Nodes, Hidden_Nodes_1, Hidden_Nodes_2, Hidden_Nodes_3, Hidden_Nodes_4, Hidden_Nodes_5, Hidden_Nodes_6, Hidden_Nodes_7, Hidden_Nodes_8, Out_Nodes,                                                                 lr, l2, num_epochs, Dropout_Rate, patience)
        #print(prof)
        if loss_valid < opt_loss:
            opt_l2_loss = l2
            opt_lr_loss = lr
            opt_loss = loss_valid
            opt_c_index_tr = c_index_tr
            opt_c_index_va = c_index_va
        print ("L2: ", l2, "LR: ", lr, "Loss in Validation: ", loss_valid)
#with torch.autograd.profiler.profile(use_cuda=True) as prof:
loss_train, loss_test, c_index_tr, c_index_te = trainCoxPASNet(x_train, age_train, cstage_train, hgrade_train, race_black_train, race_white_train, ytime_train, yevent_train,                             x_test, age_test, cstage_test, hgrade_test, race_black_test, race_white_test, ytime_test, yevent_test,                             MiRNA_Nodes, Hidden_Nodes_1, Hidden_Nodes_2, Hidden_Nodes_3, Hidden_Nodes_4, Hidden_Nodes_5, Hidden_Nodes_6, Hidden_Nodes_7, Hidden_Nodes_8, Out_Nodes,                             opt_lr_loss, opt_l2_loss, Num_EPOCHS, Dropout_Rate, patience)
#print(prof)
print ("Optimal L2: ", opt_l2_loss, "Optimal LR: ", opt_lr_loss)
print("C-index in Test: ", c_index_te)

