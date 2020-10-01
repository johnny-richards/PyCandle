import numpy as np
import pdb

class OptimizerBase(object):
    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum_memory = []
    
    def apply_gradient(self, model):
        if self.momentum_memory == []:
           self._init_mometum(model)

        for idx in range(len(model.net)):
            for k, v in model.net[idx].params.items():
                grad = model.net[idx].grads[k]
                delta = self._compute_step(idx, k, grad) # momentum and gradient descent
                delta -= self.weight_decay * v # L2 regularization
                model.net[idx].params[k] += delta
    
    def _compute_step(self, idx, k, grad):
        raise NotImplementedError
    
    def _init_mometum(self):
        raise NotImplementedError
    
    def zero_grad(self, model):
        for idx in range(len(model.net)):
            for k, v in model.net[idx].grads.items():
                model.net[idx].grads[k] = np.zeros(v.shape)

class SGD(OptimizerBase):
    def __init__(self, lr, weight_decay, mu=0.9):
        super(SGD, self).__init__(lr, weight_decay)
        self.mu = mu
    
    def _compute_step(self, idx, k, grad):
        momentum = self.mu * self.momentum_memory[idx][k]['momentum'] - self.lr * grad
        self.momentum_memory[idx][k]['momentum'] = momentum
        return momentum
    
    def _init_mometum(self, model):
        for m in model.net:
            mom_dict = {}
            for k, v in m.grads.items():
                mom_dict[k] = {'momentum': np.zeros(v.shape)}
            self.momentum_memory.append(mom_dict)

class Adam(OptimizerBase):
    def __init__(self, lr, weight_decay, beta1=0.9, beta2=0.999, eps=1e-8):
        super(Adam, self).__init__(lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
    
    def _compute_step(self, idx, k, grad):
        first = self.momentum_memory[idx][k]['first']
        second = self.momentum_memory[idx][k]['second']
        time = self.momentum_memory[idx][k]['time']

        first = self.beta1 * first + (1 - self.beta1) * grad
        second = self.beta2 * second + (1 - self.beta2) * (grad ** 2)
        time += 1

        self.momentum_memory[idx][k]['first'] = first
        self.momentum_memory[idx][k]['second'] = second
        self.momentum_memory[idx][k]['time'] = time

        # bias correct
        output_m = first / (1 - self.beta1 ** time)
        output_v = second / (1 - self.beta2 ** time)

        return -self.lr * output_m / (np.sqrt(output_v) + self.eps)

    
    def _init_mometum(self, model):
        for m in model.net:
            mom_dict = {}
            for k, v in m.grads.items():
                mom_dict[k] = {'first': np.zeros(v.shape), 'second': np.zeros(v.shape), 'time': 0}
            self.momentum_memory.append(mom_dict)