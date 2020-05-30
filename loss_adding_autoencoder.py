#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch

def merged_loss_function(pred, ytime, yevent, x_d, x):
    n_observed = yevent.sum(0)
    ytime_indicator = R_set(ytime)
    n_subjects = x.size(0)
    pdist = torch.nn.PairwiseDistance(p=2)
    distance = pdist(x_d, x)
    sum_dist = distance.sum(0)
    if torch.cuda.is_available():
        ytime_indicator = ytime_indicator.cuda()
    risk_set_sum = ytime_indicator.mm(torch.exp(pred)) 
    diff = pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
    cost = (- (sum_diff_in_observed / n_observed) + (sum_dist / n_subjects)).reshape((-1,))
    return(cost)

