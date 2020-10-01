import numpy as np
import pdb

class Tensor(object):
    def __init__(self, value, require_grad=False, parents=None, type=np.float):
        # tensor value
        self.value = np.array(value).astype(type)
        self.shape = self.value.shape
        # gradient
        self.grad = np.zeros(self.shape).astype(np.float) if require_grad else None
        # required gradient
        self.require_grad = require_grad
        # parents
        if parents is None:
            self.parents = []
        else:
            self.parents = parents
    
    def zero_grad(self):
        self.grad = np.zeros(self.shape).astype(np.float)
    
    def backward(self, grad=None):
        if grad is None:
            grad = np.array(1.0)
        grad = np.array(grad).astype(np.float)
        
        self.grad += grad
        
        for node in self.parents:
            grad_input = node['grad_fn'](grad)
            node['tensor'].backward(grad_input)
    
    # add
    def __add__(self, x):
        # value
        value = self.value + x.value
        # require gradient
        require_grad = self.require_grad or x.require_grad
        # gradient
        parents = []
        if self.require_grad:
            def grad_fn1(grad):
                return grad
            parents.append(dict(tensor=self, grad_fn=grad_fn1))
        if x.require_grad:
            def grad_fn2(grad):
                return grad
            parents.append(dict(tensor=x, grad_fn=grad_fn2))
        return Tensor(value, require_grad, parents)
    
    # sub
    def __sub__(self, x):
        # value
        value = self.value - x.value
        # require gradient
        require_grad = self.require_grad or x.require_grad
        # gradient
        parents = []
        if self.require_grad:
            def grad_fn1(grad):
                return grad
            parents.append(dict(tensor=self, grad_fn=grad_fn1))
        if x.require_grad:
            def grad_fn2(grad):
                return -grad
            parents.append(dict(tensor=x, grad_fn=grad_fn2))
        return Tensor(value, require_grad, parents)

    
    # element multiply
    def __mul__(self, x):
        # value
        value = self.value * x.value
        # require gradient
        require_grad = self.require_grad or x.require_grad
        # gradient
        parents = []
        if self.require_grad:
            def grad_fn1(grad):
                return x.value * grad
            parents.append(dict(tensor=self, grad_fn=grad_fn1))
        if x.require_grad:
            def grad_fn2(grad):
                return self.value * grad
            parents.append(dict(tensor=x, grad_fn=grad_fn2))
        return Tensor(value, require_grad, parents)
    
    # matrix multiply
    def __matmul__(self, x):
        # value
        value = self.value @ x.value
        # require gradient
        require_grad = self.require_grad or x .require_grad
        # gradient
        parents = []
        if self.require_grad:
            def grad_fn1(grad):
                return grad @ x.value.T
            parents.append(dict(tensor=x, grad_fn=grad_fn1))
        if x.require_grad:
            def grad_fn2(grad):
                return self.value.T @ grad
            parents.append(dict(tensor=x, grad_fn=grad_fn2))
        return Tensor(value, require_grad, parents)
    
    # sum
    def sum(self, dim=None):
        # value
        value = self.value.sum(dim)
        # require gradient
        require_grad = self.require_grad
        # gradient
        parents = []
        if self.require_grad:
            def grad_fn(grad):
                if dim is None:
                    grad_input = np.tile(grad, self.value.shape)
                elif type(dim) is int:
                    tshape = [1] * len(self.value.shape)
                    tshape[dim] = self.value.shape[dim]
                    rshape = list(self.value.shape)
                    rshape[dim] = 1
                    grad_input = np.tile(grad.reshape(rshape), tshape)
                elif type(dim) is list or type(dim) is tuple:
                    tshape = [1] * len(self.value.shape)
                    for d in dim:
                        tshape[d] = self.value.shape[d]
                    rshape = list(self.value.shape)
                    for d in dim:
                        rshape[d] = 1
                    grad_input = np.tile(grad.reshape(rshape, tshape))
                return grad_input
            parents.append(dict(tensor=self, grad_fn=grad_fn))
        return Tensor(value, require_grad, parents)
    
    # exp
    def exp(self):
        # value
        value = np.exp(self.value)
        # require gradient
        require_grad = self.require_grad
        # gradient
        parents = []
        if self.require_grad:
            def grad_fn(grad):
                return grad * value
            parents.append(dict(tensor=self, grad_fn=grad_fn))
        return Tensor(value, require_grad, parents)
    
    # log
    def log(self):
        # value
        value = np.log(self.value)
        # require gradient
        require_grad = self.require_grad
        # gradient
        parents = []
        if self.require_grad:
            def grad_fn(grad):
                return grad / self.value
            parents.append(dict(tensor=self, grad_fn=grad_fn))
        return Tensor(value, require_grad, parents)
    
    # view
    def view(self, shape):
        # value
        value = self.value.reshape(shape)
        # require gradient
        require_grad = self.require_grad
        # gradient
        parents = []
        if self.require_grad:
            def grad_fn(grad):
                return grad.reshape(self.shape)
            parents.append(dict(tensor=self, grad_fn=grad_fn))
        return Tensor(value, require_grad, parents)

if __name__ == '__main__':
    x = Tensor([1,2,3,4], require_grad=True)
    y = Tensor([2,3,4,6], require_grad=True)
    z = x * y
    r = z - x
    r *= r
    r += r
    r = r.view([1,4,1]).sum()
    # r = r.sum()
    r.backward()
    # print('r1 value {} r2 value{}'.format(r1.value, r2.value))
    print('x grad {} y grad {} z grad {} r grad {}'.format(x.grad, y.grad, z.grad, r.grad))
    print('parents {}'.format(r.parents))