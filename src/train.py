import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from model import *

from dataset import train_dataset, test_dataset, validation_dataset, get_token_dict_from_file
# from test import Validate

# os.makedirs('models', True) # Directory to save / load models
def Validate():
	tokens_to_index = get_token_dict_from_file()
	count = 0
	conjectures = []
	statements = []
	labels = []
	for datapoint in validation_dataset():
		conjecture = datapoint.conjecture
		statement = datapoint.statement

		for node_id, node_obj in conjecture.nodes.items(): # Find and replace unknowns
			if node_obj not in tokens_to_index.keys(): # UNKOWN token
				node_obj.token = "UNKOWN"

		conjectures.append(conjecture)
		statements.append(statement)
		labels.append(datapoint.label)


		count += 1

		if count == 100: # Just validate only over a few.
			break

	prediction_val, prediction_label  = F(conjectures, statements)

	prediction_label = prediction_label.numpy()
	err_count = np.where(prediction_label != labels)[0].size
	print("Fraction of Incorrect Validations: ", err_count / count)

	return err_count / count




parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch Size')
parser.add_argument('--epochs', type = int, default = 5, help = 'Number of training episodes')
parser.add_argument('--num_steps', type = int, default = 1, help = 'Number of update steps for equation 1 or 2')
parser.add_argument('--lr', type = float, default = 1e-3, help = 'Initial learning rate for RMSProp')
parser.add_argument('--weight_decay', type = float, default = 1e-4, help = "Weight decay parameter for RMSProp")
parser.add_argument('--lr_decay', type = float, default = 3., help = 'Multiplicative Factor by which to decay learning rate by after each epoch, > 1')

parser.add_argument('--load_model', type = bool, default = False, help  = 'Set True if loading model')
parser.add_argument('--model_path', type = str, default = None, help = 'Path to Model File')
parser.add_argument('--opt_path', type = str, default = None, help = 'Path to Optimizer File')

parser.add_argument('--start_epoch', type = int, default = 0, help = 'Epoch to resume with')


# parser.add_argument('--start_epoch', type = int, default = 0, help = 'Epoch to resume with')

args = parser.parse_args()
print(args)


cuda_available = torch.cuda.is_available()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss Function
loss = nn.BCELoss() # Binary Cross-Entropy Loss

# Define Model. Decide whether to load (first case) or start new (else)
F = FormulaNet(args.num_steps, args.batch_size, loss, cuda_available)
opt = torch.optim.RMSprop(F.parameters(), lr = args.lr, alpha = args.weight_decay)
if cuda_available: 
	F.cuda()
	print("Cuda Enabled!")


# Enabling loading
# if args.load_model is True: # Load Model
# 	if args.model_path is None:
# 		raise ValueError('Specify Model Path')
# 	if args.opt_path is None:
# 		raise ValueError('Specify Optimizer Path')

# 	F = F.load_state_dict(torch.load(args.model_path))
# 	opt = opt.load_state_dict(torch.load(args.opt_path))

# else: # New Model
# 	# Enable GPU if available.
# 	if cuda: 
# 		F.cuda()
# 	# Optimizers

""" 
Begin Training
"""
lr = args.lr
for epoch in range(args.start_epoch, args.epochs):
	# I will assume the graphs are shuffled somehow.
	batch_index = 0
	batch_number = 1 # Number of batches iterated.
	conjecture_graph_batch = []
	statement_graph_batch = []
	label_batch = []
	for datapoint in train_dataset():
		if batch_index != args.batch_size:
			# Map the graph object into an array of one hot vectors for both conjecture and statement.
			conjecture_graph = datapoint.conjecture
			statement_graph = datapoint.statement
			label = datapoint.label

			conjecture_graph_batch.append(conjecture_graph)
			statement_graph_batch.append(statement_graph)
			label_batch.append(label)

			batch_index += 1
			continue
		else:
			batch_index = 0

		predict_val, prediction = F(conjecture_graph_batch, statement_graph_batch)

		if cuda_available:
			label_batch_tensor = torch.Tensor(label_batch).cuda()
		else:
			label_batch_tensor = torch.Tensor(label_batch)

		# Compute loss due to prediction. How to make label_scores just a scalar? argmax?
		curr_loss = loss(predict_val, label_batch_tensor)

		# Backpropogation.
		curr_loss.backward()
		opt.step()

		print("Trained %d Batches" %batch_number)

		batch_number += 1

		# Train after this many batches.
		if batch_number % 100 == 0:
			if batch_number > 0:
				print("Batch Number: %d" %batch_number)
				val_err = Validate()
				print("Validation Error: %f" %valid)

	# --------------- End of Epoch --------------- #

	# Costruct new optimizers after each epoch.
	lr = lr / args.lr_decay
	opt = torch.optim.RMSprop(F.parameters(), lr = lr, alpha = args.weight_decay)

	print("Epoch # "+str(epoch + 1)+" done.")
	

	# Save Model After Each Epoch
	MODEL_PATH = args.model_path 
	if MODEL_PATH is None: # If no model path was specified. Write to default model_path in ../model
		MODEL_DIR = os.path.join("..", "models")
		if not os.path.exists(MODEL_DIR):
			os.makedirs(MODEL_DIR)
		MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")

	torch.save(F.state_dict(), MODEL_PATH)


	# Save Optimizer to be used for next epoch.
	OPT_PATH = args.opt_path
	if OPT_PATH is None:
		MODEL_DIR = os.path.join("..", "models")
		if not os.path.exists(MODEL_DIR):
			os.makedirs(MODEL_DIR)
		OPT_PATH = os.path.join(MODEL_DIR, "opt.pt")

	torch.save(opt.state_dict(), OPT_PATH)

	print("Models and Optimizers Saved.")



	# Validate Model after Each Epoch
	val_err = Validate()
	print("Validation Error: "+str(val_err))

