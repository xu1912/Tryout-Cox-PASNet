#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn

class Cox_PASNet(nn.Module):
    def __init__(self, MiRNA_Nodes, Hidden_Nodes_1, Hidden_Nodes_2, Out_Nodes):
        super(Cox_PASNet, self).__init__()
        self.tanh = nn.Tanh()
        self.sc1 = nn.Linear(MiRNA_Nodes, Hidden_Nodes_1)
        self.sc2 = nn.Linear(Hidden_Nodes_1, Hidden_Nodes_2)
        self.sc3 = nn.Linear(Hidden_Nodes_2, Out_Nodes)
        
        self.sc4 = nn.Linear(Out_Nodes+5, 1, bias = False)
        
        self.dc1 = nn.Linear(Out_Nodes, Hidden_Nodes_2)
        self.dc2 = nn.Linear(Hidden_Nodes_2, Hidden_Nodes_1)
        self.dc3 = nn.Linear(Hidden_Nodes_1, MiRNA_Nodes)
        
        self.sc4.weight.data.uniform_(-0.001, 0.001)
        self.do_m1 = torch.ones(MiRNA_Nodes)
        self.do_m2 = torch.ones(Hidden_Nodes_1)
        self.do_m3 = torch.ones(Hidden_Nodes_2)
        if torch.cuda.is_available():
            self.do_m1 = self.do_m1.cuda()
            self.do_m2 = self.do_m2.cuda()
            self.do_m3 = self.do_m3.cuda()
    def forward(self, x_1, x_2, x_3, x_4, x_5, x_6):
        if self.training == True:
            x_1 = x_1.mul(self.do_m1)
        x_1 = self.tanh(self.sc1(x_1))
        if self.training == True:
            x_1 = x_1.mul(self.do_m2)
        x_1 = self.tanh(self.sc2(x_1))
        if self.training == True:
            x_1 = x_1.mul(self.do_m3)
        x_1 = self.tanh(self.sc3(x_1))
        x_d = self.tanh(self.dc1(x_1))
        x_d = self.tanh(self.dc2(x_d))
        x_d = self.dc3(x_d)
        x_cat = torch.cat((x_1, x_2, x_3, x_4, x_5, x_6), 1)
        lin_pred = self.sc4(x_cat)
        return (lin_pred, x_d)

