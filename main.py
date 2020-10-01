import numpy as np
import modules
from modules import Linear
from modules import Softmax
from modules import ReLU
from modules import ListModel
from modules import CrossEntropyLoss

from optimizer import SGD, Adam
from data import MNISTDataLoader

import pdb

num_epochs = 20
lr = 0.001
# mu = 0.9
weight_decay = 0.0001
batch_size = 128
print_every = 50

def main():
    # optimizer = SGD(lr, weight_decay, mu=mu)
    optimizer = Adam(lr, weight_decay)
    model = ListModel(net = [Linear(784, 400),
                             ReLU(),
                             Linear(400, 100),
                             ReLU(),
                             Linear(100, 10),
                             Softmax()],
                      loss = CrossEntropyLoss())
    for epoch in range(num_epochs):
        print('epoch number: {}'.format(epoch))
        train(model, optimizer)
        valid(model)

def train(model, optimizer):
    train_loader = MNISTDataLoader(batch_size, './datasets/mnist', True, True)
    total_predict = 0
    correct_predict = 0
    total_batches = len(train_loader)
    for idx, batch in enumerate(train_loader):
        data, label = batch
        optimizer.zero_grad(model) # zero model grads
        data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
        predicted, loss = model.forward(data, label)
        model.backward()
        optimizer.apply_gradient(model) # update model params
        # pdb.set_trace()
        # correctness
        predicted_label = np.argmax(predicted, 1)
        correct_predict_cur = np.sum((predicted_label == label).astype(np.float))
        total_predict_cur = predicted_label.shape[0]
        accuracy_cur = correct_predict_cur / total_predict_cur
        correct_predict += correct_predict_cur
        total_predict += total_predict_cur
        if idx % print_every == 0:
            print('batch {} of {}: accuracy {} loss {}'.format(idx, total_batches, accuracy_cur, loss))

    print('train epoch accuracy: {}'.format(correct_predict / total_predict))

def valid(model):
    valid_loader = MNISTDataLoader(batch_size, './datasets/mnist', False, False)
    total_predict = 0
    correct_predict = 0
    total_batches = len(valid_loader)
    for idx, batch in enumerate(valid_loader):
        data, label = batch
        data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
        predicted, loss = model.forward(data, label)
        # correctness
        predicted_label = np.argmax(predicted, 1)
        correct_predict_cur = np.sum((predicted_label == label).astype(np.float))
        total_predict_cur = predicted_label.shape[0]
        accuracy_cur = correct_predict_cur / total_predict_cur
        correct_predict += correct_predict_cur
        total_predict += total_predict_cur
        if idx % print_every == 0:
            print('batch {} of {}: accuracy {} loss {}'.format(idx, total_batches, accuracy_cur, loss))
    print('valid epoch accuracy: {}'.format(correct_predict / total_predict))


if __name__ == '__main__':
    main()