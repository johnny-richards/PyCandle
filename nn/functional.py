import numpy as np
from nn.tensor import Tensor

# sigmoid
def sigmoid(x):
    # value
    evalue = np.exp(-x.value)
    svalue = 1 / (1 + evalue)
    # require grad
    require_grad = x.require_grad
    # gradient
    parents = []
    if x.require_grad:
        def grad_fn(grad):
            return svalue * (1 - svalue) * grad
        parents.append(dict(tensor=x, grad_fn=grad_fn))
    return Tensor(svalue, require_grad, parents)

# softmax
def softmax(x):
    # value
    xmax = np.max(x.value, 1)
    xmax_expand = np.tile(xmax.reshape(x.shape[0], 1), (1, x.shape[1]))
    evalue = np.exp(x - xmax_expand)
    esum = np.sum(evalue, 1)
    esum_expand = np.tile(esum.reshape(x.shape[0], 1), (1, x.shape[1]))
    svalue = evalue / esum_expand
    # require grad
    require_grad = x.require_grad
    # gradient
    parents = []
    if x.require_grad:
        def grad_fn(grad):
            W1 = np.array([np.diag(q) for q in svalue])
            q = svalue.reshape(svalue.shape[0], svalue.shape[1], 1)
            qt = svalue.reshape(svalue.shape[0], 1, svalue.shape[1])
            W2 = np.array([np.dot(q[k], qt[k]) for k in range(q.shape[0])])
            W = W1 - W2
            grad_expand = grad.reshape(grad.shape[0], 1, grad.shape[1])
            grad_input_expand = np.array([np.dot(grad_expand[k], W[k]) for k in range(grad.shape[0])])
            grad_input = grad_input_expand.reshape(grad.shape[0], grad.shape[1])
            return grad_input
        parents.append(dict(tensor=x, grad_fn=grad_fn))
    return Tensor(svalue, require_grad, parents)

# batch multiplication
def bmm(x, y):
    # value
    x_matrix = [m for m in x.value]
    y_matrix = [m for m in y.value]
    value = np.array([xm @ y_matrix[idx] for idx, xm in enumerate(x_matrix)])
    # require grad
    require_grad = x.require_grad or y.require_grad
    # gradient
    parents = []
    if x.require_grad:
        def grad_fn1(grad):
            g_matrix = [m for m in grad]
            grad_input = np.array([gm @ y_matrix[idx].T for idx, gm in enumerate(g_matrix)])
            return grad_input
        parents.append(dict(tensor=x, grad_fn=grad_fn1))
    
    if y.require_grad:
        def grad_fn2(grad):
            g_matrix = [m for m in grad]
            grad_input = np.array([xm.T @ g_matrix[idx] for idx, xm in enumerate(x_matrix)])
            return grad_input
        parents.append(dict(tensor=x, grad_fn=grad_fn2))
    return Tensor(value, require_grad, parents)