import numpy as np
import os
import random

import pdb

class DataLoader(object):
    def __init__(self, batch_size, data_dir, is_train, shuffle):
        super(DataLoader, self).__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.is_train = is_train
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

class MNISTDataLoader(DataLoader):
    def __init__(self, batch_size, data_dir, is_train, shuffle):
        super(MNISTDataLoader, self).__init__(batch_size, data_dir, is_train, shuffle)
        if self.is_train:
            data_file = 'train-images-idx3-ubyte'
            label_file = 'train-labels-idx1-ubyte'
        else:
            data_file = 't10k-images-idx3-ubyte'
            label_file = 't10k-labels-idx1-ubyte'
        with open(os.path.join(self.data_dir, data_file)) as fd:
            raw_data = np.fromfile(file=fd, dtype=np.uint8)
        with open(os.path.join(self.data_dir, label_file)) as fl:
            raw_label = np.fromfile(file=fl, dtype=np.uint8)
        label = raw_label[8:]
        data = np.reshape(raw_data[16:], (label.shape[0], 28, 28, 1))
        # zip and shuffle
        self.dataset = list(zip(data, label))
        if shuffle:
            random.shuffle(self.dataset)
        # batchify
    
    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise StopIteration
        data, label = zip(*self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size])
        normalized_data = np.array(data).astype(np.float) / 255.0 - 0.5 # [-0.5, 0.5]
        output_label = np.array(label)

        return normalized_data, output_label
    
    def __len__(self):
        return (len(self.dataset) - 1) // self.batch_size + 1

if __name__ == '__main__':
    batch_size = 64
    mnist_train_loader = MNISTDataLoader(batch_size, './datasets/mnist', True, True)
    mnist_valid_loader = MNISTDataLoader(batch_size, './datasets/mnist', False, False)
    print('data length {}'.format(len(mnist_train_loader)))

    for idx, batch in enumerate(mnist_train_loader):
        data, label = batch
        try:
            assert len(data)==batch_size
            assert len(label)==batch_size
        except:
            # pdb.set_trace()
            pass
    assert idx == (60000 - 1) // batch_size

    for idx, batch in enumerate(mnist_valid_loader):
        data, label = batch
        try:
            assert len(data)==batch_size
            assert len(label)==batch_size
        except:
            pdb.set_trace()
    assert idx == (60000 - 1) // batch_size