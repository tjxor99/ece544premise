import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from model import *

from dataset import train_dataset, test_dataset, validation_datset

# os.makedirs('models', True) # Directory to save / load models
def Validate():
	err_count = 0
	count = 0
	for datapoint in dataset.validation_datset():
		conjecture = datapoint.conjecture
		statement = datapoint.statement
		label = datapoint.label
		prediction = F(conjecture, statement)

		count += 1
		if prediction != label:
			err_count += 1

	return err_count / count


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch Size')
parser.add_argument('--epochs', type = int, default = 5, help = 'Number of training episodes')
parser.add_argument('--num_steps', type = int, default = 1, help = 'Number of update steps for equation 1 or 2')
parser.add_argument('--lr', type = float, default = 1e-3, help = 'Initial learning rate for RMSProp')
parser.add_argument('--weight_decay', type = float, default = 1e-4, help = "Weight decay parameter for RMSProp")
parser.add_argument('--lr_decay', type = float, default = 3., help = 'Multiplicative Factor by which to decay learning rate by after each epoch, > 1')

args = parser.parse_args()
print(args)





cuda = True if torch.cuda.is_available() else False


# Loss Function
loss = nn.BCELoss() # Binary Cross-Entropy Loss

# Define Model
F = FormulaNet(args.num_steps, args.batch_size)


# Enable GPU if available.
if cuda: 
	F.cuda()

# Optimizers
opt = torch.optim.RMSprop(F.parameters(), lr = args.lr, alpha = args.weight_decay)

""" 
Begin Training
"""
lr = args.lr
for epoch in range(args.epochs):
	# I will assume the graphs are shuffled somehow.
	batch_number = 1
	for datapoint in train_dataset():
		conjecture_graph_batch = []
		statement_graph_batch = []
		label_batch = []
		for _ in range(args.batch_size):
			# Map the graph object into an array of one hot vectors for both conjecture and statement.
			conjecture_graph = datapoint.conjecture
			statement_graph = datapoint.statement
			label = datapoint.label

			conjecture_graph_batch.append(conjecture_graph)
			statement_graph_batch.append(statement_graph)
			label_batch.append(label)

		predict_val, prediction = F(conjecture_graph_batch, statement_graph_batch)

		label_batch_tensor = torch.Tensor(label_batch)

		# Compute loss due to prediction. How to make label_scores just a scalar? argmax?
		curr_loss = loss(predict_val, label_batch_tensor)

		# Backpropogation.
		curr_loss.backward()
		opt.step()

		print("Trained over batch number ", batch_number)
		print("Training Loss: ", curr_loss)

		batch_number += 1


	# End of epoch.

	# Costruct new optimizers after each epoch.
	lr = lr / args.lr_decay
	opt = torch.optim.RMSProp(F.parameters(), lr = lr, alpha = args.weight_decay)

	print("Epoch # "+str(epoch + 1)+" done.")

	# Validate
	# val_err = Validate()
	# print("Validation Error: "+str(val_err))

