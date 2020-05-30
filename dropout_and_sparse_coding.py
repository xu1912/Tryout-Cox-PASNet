#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import math

def dropout_mask(n_node, drop_p):
    keep_p = 1.0 - drop_p
    mask = torch.Tensor(np.random.binomial(1, keep_p, size = n_node))
    if torch.cuda.is_available():
        mask = mask.cuda()
    return mask

def s_mask(sparse_level, param_matrix, nonzero_param_1D, dtype):
    non_neg_param_1D = torch.abs(nonzero_param_1D)
    num_param = nonzero_param_1D.size(0)
    top_k = math.ceil(num_param*(100-sparse_level)*0.01)
    sorted_non_neg_param_1D, indices = torch.topk(non_neg_param_1D, top_k)
    param_mask = torch.abs(param_matrix) > sorted_non_neg_param_1D.min()
    param_mask = param_mask.type(dtype)
    if torch.cuda.is_available():
        param_mask = param_mask.cuda()
    return param_mask

