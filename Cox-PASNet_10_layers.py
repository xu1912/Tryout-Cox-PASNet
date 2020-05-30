#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn

class Cox_PASNet(nn.Module):
    def __init__(self, MiRNA_Nodes, Hidden_Nodes_1, Hidden_Nodes_2, Hidden_Nodes_3, Hidden_Nodes_4, Hidden_Nodes_5, Hidden_Nodes_6, Hidden_Nodes_7, Hidden_Nodes_8, Out_Nodes):
        super(Cox_PASNet, self).__init__()
        self.tanh = nn.Tanh()
        self.sc1 = nn.Linear(MiRNA_Nodes, Hidden_Nodes_1)
        self.sc2 = nn.Linear(Hidden_Nodes_1, Hidden_Nodes_2)
        self.sc3 = nn.Linear(Hidden_Nodes_2, Hidden_Nodes_3)
        self.sc4 = nn.Linear(Hidden_Nodes_3, Hidden_Nodes_4)
        self.sc5 = nn.Linear(Hidden_Nodes_4, Hidden_Nodes_5)
        
        self.sc6 = nn.Linear(Hidden_Nodes_5, Hidden_Nodes_6)
        self.sc7 = nn.Linear(Hidden_Nodes_6, Hidden_Nodes_7)
        self.sc8 = nn.Linear(Hidden_Nodes_7, Hidden_Nodes_8)
        self.sc9 = nn.Linear(Hidden_Nodes_8, Out_Nodes)
        self.sc10 = nn.Linear(Out_Nodes+5, 1, bias = False)
        
        self.sc10.weight.data.uniform_(-0.001, 0.001)
        
        self.do_m1 = torch.ones(MiRNA_Nodes)
        self.do_m2 = torch.ones(Hidden_Nodes_1)
        self.do_m3 = torch.ones(Hidden_Nodes_2)
        self.do_m4 = torch.ones(Hidden_Nodes_3)
        self.do_m5 = torch.ones(Hidden_Nodes_4)
        self.do_m6 = torch.ones(Hidden_Nodes_5)
        self.do_m7 = torch.ones(Hidden_Nodes_6)
        self.do_m8 = torch.ones(Hidden_Nodes_7)
        self.do_m9 = torch.ones(Hidden_Nodes_8)
        if torch.cuda.is_available():
            self.do_m1 = self.do_m1.cuda()
            self.do_m2 = self.do_m2.cuda()
            self.do_m3 = self.do_m3.cuda()
            self.do_m4 = self.do_m4.cuda()
            self.do_m5 = self.do_m5.cuda()
            self.do_m6 = self.do_m6.cuda()
            self.do_m7 = self.do_m7.cuda()
            self.do_m8 = self.do_m8.cuda()
            self.do_m9 = self.do_m9.cuda()
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
        if self.training == True:
            x_1 = x_1.mul(self.do_m4)
        x_1 = self.tanh(self.sc4(x_1))
        if self.training == True:
            x_1 = x_1.mul(self.do_m5)
        x_1 = self.tanh(self.sc5(x_1))
        if self.training == True:
            x_1 = x_1.mul(self.do_m6)
        x_1 = self.tanh(self.sc6(x_1))
        if self.training == True:
            x_1 = x_1.mul(self.do_m7)
        x_1 = self.tanh(self.sc7(x_1))
        if self.training == True:
            x_1 = x_1.mul(self.do_m8)
        x_1 = self.tanh(self.sc8(x_1))
        if self.training == True:
            x_1 = x_1.mul(self.do_m9)
        x_1 = self.tanh(self.sc9(x_1))
        x_cat = torch.cat((x_1, x_2, x_3, x_4, x_5, x_6), 1)
        lin_pred = self.sc10(x_cat)
        return (lin_pred)

