from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
from torch.nn import Adam, CrossEntropyLoss

from nn import Net

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
args = parser.parse_args()

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# model
model = Net()

# optimizer
optimizer = Adam(model.parameters(), 
                 lr=args.lr, 
                 weight_decay=args.weight_decay)

# loss function
criterion = CrossEntropyLoss()

# checking if GPU is available
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    model = model.cuda()
    criterion = criterion.cuda()
    
#print(model)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    """
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    x_val, y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()
    """    

    output_train = model(x_train)
    loss_train = criterion(output_train, y_train)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()

    loss_val = criterion(output_val, y_val)
    
    print('Epoch: {}'.format(epoch + 1),
          'loss_train: {}'.format(loss_train.item()),
          'acc_train: {}'.format(),
          'loss_val: {}'.format(),
          'acc_val: {}'.format(),
          'time: {}'.format(time.time() - t))




# defining the number of epochs
n_epochs = 25

# empty list to store training losses
train_losses = []

# empty list to store validation losses
val_losses = []

# training the model
t_total = time.time()
for epoch in range(n_epochs):
    train(epoch)
print('Finish Training!')
print('Total time elapsed: {}'.format(time.time() - t_total))    