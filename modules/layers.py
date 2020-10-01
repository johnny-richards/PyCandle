import numpy as np
import pdb

from modules.base import ModuleBase

class Linear(ModuleBase):
    def __init__(self, input_dim, output_dim, bias=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.params['weight'] = self.W
        if bias:
            self.b = np.zeros(output_dim)
            self.params['bias'] = self.b
    
    def forward(self, x):
        # check type
        assert isinstance(x, np.ndarray)
        # x: batch_size x input_dim
        batch_size = x.shape[0]
        y = np.dot(x, self.W)
        if self.bias:
            y += np.tile(self.b, (batch_size, 1))
        self.results['x'] = x
        return y

    def backward(self, grad):
        # check type
        assert isinstance(grad, np.ndarray)
        # grad: batch_size x output_dim
        grad_input = np.dot(grad, self.W.T) # batch_size x input_dim
        # gradient of such layer
        x = self.results['x'] # batch_size x input_dim
        batch_size = x.shape[0]
        input_dim = x.shape[1]
        output_dim = grad.shape[1]
        x_unsqueeze = x.reshape((batch_size, input_dim, 1))
        grad_unsqueeze = grad.reshape(batch_size, 1, output_dim)
        Wgrad = np.sum(np.array([np.dot(cur_x, grad_unsqueeze[idx])
                for idx, cur_x in enumerate(x_unsqueeze)]), 0) # input_dim x output_dim
        if self.bias:
            bgrad = np.sum(grad, 0) # output_dim
        # save grad
        self.grads['weight'] = Wgrad
        if self.bias:
            self.grads['bias'] = bgrad
        # initialize result
        self.results = {}
        return grad_input

class Sigmoid(ModuleBase):
    def __init__(self):
        super(Sigmoid, self).__init__()
    def forward(self, x):
        # x: batch_size x dim
        evalue = np.exp(-x)
        sigmoid = 1 / (1 + evalue)
        self.results['sigmoid'] = sigmoid
        return sigmoid
    def backward(self, grad):
        sigmoid = self.results['sigmoid']
        grad_input = grad * sigmoid * (1 - sigmoid)
        self.results = {}
        return grad_input

class Softmax(ModuleBase):
    def __init__(self):
        super(Softmax, self).__init__()
    def forward(self, x):
        # x: batch_size x dim
        xmax = np.max(x, 1) # batch_size
        xmax_expand = np.tile(xmax.reshape(x.shape[0], 1), (1, x.shape[1]))
        evalue = np.exp(x - xmax_expand)
        esum = np.sum(evalue, 1)
        esum_expand = np.tile(esum.reshape(x.shape[0], 1), (1, x.shape[1]))
        softmax = evalue / esum_expand # batch_size x dim
        self.results['softmax'] = softmax
        return softmax
    def backward(self, grad):
        softmax = self.results['softmax']
        self.results = {}
        W1 = np.array([np.diag(q) for q in softmax]) # batch_size x dim x dim
        q = softmax.reshape(softmax.shape[0], softmax.shape[1], 1) # batch_size x dim x 1
        qt = softmax.reshape(softmax.shape[0], 1, softmax.shape[1]) # batch_size x 1 x dim
        W2 = np.array([np.dot(q[k], qt[k]) for k in range(q.shape[0])]) # batch_size x dim x dim
        W = W1 - W2
        grad_expand = grad.reshape(grad.shape[0], 1, grad.shape[1]) # batch_size x 1 x dim
        grad_input_expand = np.array([np.dot(grad_expand[k], W[k]) for k in range(grad.shape[0])]) # batch_size x 1 x dim
        grad_input = grad_input_expand.reshape(grad.shape[0], grad.shape[1]) # batch_size x dim
        return grad_input

class Tanh(ModuleBase):
    def __init__(self):
        super(Tanh, self).__init__()
    def forward(self, x):
        # x: batch_size x dim
        evalue = np.exp(x)
        value = (evalue - 1 / evalue) / (evalue + 1 / evalue)
        self.results['tanh'] = value
        return value
    def backward(self, grad):
        value = self.results['tanh']
        self.results = {}
        grad_input = grad * (1 - value ** 2)
        return grad_input

class ReLU(ModuleBase):
    def __init__(self):
        super(ReLU, self).__init__()
    def forward(self, x):
        # x: batch size x dim
        mask = (x > 0.0).astype(np.float)
        y = mask * x
        self.results['mask'] = mask
        return y
    def backward(self, grad):
        grad_input = grad * self.results['mask']
        self.results = {}
        return grad_input

if __name__=='__main__':
    # check gradient
    input_dim = 8
    output_dim = 5
    batch_size = 2
    # model
    linear = Linear(input_dim, output_dim)
    print('{}'.format(linear.__class__.__name__))
    