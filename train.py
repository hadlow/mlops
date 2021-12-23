# -*- coding: utf-8 -*-
import struct as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def process_image_file(image_file):
    image_file.seek(0)
    magic = st.unpack('>4B', image_file.read(4))

    n_images = st.unpack('>I', image_file.read(4))[0]
    n_rows = st.unpack('>I', image_file.read(4))[0]
    n_columns = st.unpack('>I', image_file.read(4))[0]
    n_bytes = n_images * n_rows * n_columns

    images = np.zeros((n_images, n_rows * n_columns))
    images = np.asarray(st.unpack('>' + 'B' * n_bytes, image_file.read(n_bytes))).reshape((n_images, n_rows * n_columns))
    images = torch.tensor(images)

    return images

def process_label_file(label_file):
    label_file.seek(0)
    magic = st.unpack('>4B', label_file.read(4))

    n_labels = st.unpack('>I', label_file.read(4))[0]

    labels = np.zeros((n_labels))
    labels = np.asarray(st.unpack('>' + 'B' * n_labels, label_file.read(n_labels)))

    targets = np.array([labels]).reshape(-1)

    one_hot_labels = np.eye(10)[targets]
    one_hot_labels = torch.tensor(one_hot_labels)

    return one_hot_labels

def dataset():
    home = './data/'

    test_images = open(home + 't10k-images-idx3-ubyte', 'rb')
    test_labels = open(home + 't10k-labels-idx1-ubyte', 'rb')
    train_images = open(home + 'train-images-idx3-ubyte', 'rb')
    train_labels = open(home + 'train-labels-idx1-ubyte', 'rb')
    
    train_images = process_image_file(train_images)
    test_images = process_image_file(test_images)
    train_labels = process_label_file(train_labels)
    test_labels = process_label_file(test_labels)
    
    return ((train_images, test_images), (train_labels, test_labels))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))

        return x

((train_images, test_images), (train_labels, test_labels)) = dataset()
train_images, test_images = train_images / 255.0, test_images / 255.0

n_training_examples = 60000
n_epochs = 1
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 1000
batch_id = 0
train_losses = []

network = Net()
optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum = momentum)

for epoch in range(n_epochs):
    network.train()

    shuffle_index = np.random.permutation(n_training_examples)
    batch_id = 0

    for batch_start in shuffle_index:
        batch_id = batch_id + 1

        batch_end = batch_start + batch_size_train

        batch_x = train_images[batch_start:batch_end]
        batch_y = train_labels[batch_start:batch_end]

        optimizer.zero_grad()
        output = network(batch_x)
        loss = F.nll_loss(output, torch.max(batch_y, 1)[1])
        loss.backward()
        optimizer.step()

        if batch_id % log_interval == 0:
            print("Batch: {}, Epoch: {}, train loss: {}", batch_id, format(epoch + 1), format(loss.item()))
            train_losses.append(loss.item())

