import numpy as np
import pdb

from modules.base import ModuleBase

# only support one net, one loss
class ListModel(ModuleBase):
    def __init__(self, net, loss):
        super(ModuleBase, self).__init__()
        self.net = net
        self.loss_func = loss
    
    def forward(self, x, label):
        y = x # batch_size x dim
        for idx in range(len(self.net)):
            y = self.net[idx].forward(y)
        loss = self.loss_func.forward(y, label)

        return y, loss
    
    def backward(self):
        loss_grad = self.loss_func.backward()
        for idx in range(len(self.net)):
            loss_grad = self.net[len(self.net) - idx - 1].backward(loss_grad)
    
    def state_dict(self):
        paramters = []
        for idx, m in enumerate(self.net):
            paramters.append(m.params)
        return paramters
