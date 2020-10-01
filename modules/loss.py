import numpy as np
import pdb

from modules.base import ModuleBase

class BinaryCrossEntropyLoss(ModuleBase):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
    def forward(self, x, label):
        # label: batch_size
        # x: batch_size x 1
        batch_size = x.shape[0]
        blabel = label.reshape([batch_size, 1])
        bce = -(blabel * np.log(x) + (1 - blabel) * log(1 - x)) # batch_size  x 1
        self.results['blabel'] = blabel
        self.results['x'] = x
        return np.sum(bce) / batch_size
    def backward(self):
        blabel = self.results['blabel']
        batch_size = blabel.shape[0]
        x = self.results['x']
        grad_input = -(blabel / x - (1 - blabel) / (1 -  x))
        self.results = {}
        return grad_input / batch_size

class CrossEntropyLoss(ModuleBase):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    def forward(self, x, label):
        # label: batch_size
        # x:  batch_size  x num_classes
        batch_size = x.shape[0]
        num_classes = x.shape[1]
        onehot = np.eye(num_classes)[label]
        loss = -np.sum(np.log(x) * onehot)
        self.results['x'] = x
        self.results['onehot'] = onehot
        return loss / batch_size
    def backward(self):
        revx = 1 / self.results['x']
        batch_size = revx.shape[0]
        grad_input = -revx * self.results['onehot']
        self.results = {}
        return grad_input / batch_size

class L2Loss(ModuleBase):
    def __init__(self):
        super(L2Loss, self).__init__()
    def forward(self, x, groud_truth):
        # x: batch_size x dim
        batch_size = x.shape[0]
        diff = x - groud_truth # batch_size x dim
        self.result['diff'] = diff
        return np.sum(diff ** 2) / batch_size
    def backward(self):
        batch_size = self.results['diff'].shape[0]
        grad_input = 2 * self.results['diff']
        self.results = {}
        return grad_input / batch_size