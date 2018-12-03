import utils
import torch
import torch.nn as nn

import numpy as np
from model import FormulaNet
from dataset import validation_dataset, get_token_dict_from_file
import os


# os.makedirs('models', True) # Directory to save / load models
def Validate(num_datapoints):
	tokens_to_index = get_token_dict_from_file()

	err_count = 0
	count = 0
	for datapoint in validation_dataset():
		conjecture = datapoint.conjecture
		statement = datapoint.statement
		label = datapoint.label

		# for node_id, node_obj in conjecture.nodes.items(): # Find and replace unknowns
		# 	if node_obj.token not in tokens_to_index.keys(): # UNKOWN token
		# 		node_obj.token = "UNKNOWN"

		# for node_id, node_obj in statement.nodes.items():
		# 	if node_obj.token not in tokens_to_index.keys():
		# 		node_obj.token = "UNKNOWN"

		prediction_val = F([conjecture], [statement])
		_, prediction_label = torch.max(prediction_val, dim = 1)

		if cuda_available:
			prediction_label = prediction_label.cpu()
		prediction_label = prediction_label.numpy()

		# print(label)
		# print(prediction_label)

		if datapoint.label != prediction_label[0]:
			err_count += 1

		count += 1

		if count % 100 == 0:
			print("Count: ",count)

		if count == num_datapoints:
			break

	print("Fraction of Incorrect Validations: ", err_count / count)

	return err_count / count


cuda_available = torch.cuda.is_available()

F = FormulaNet(1, cuda_available)

# Load Model
MODEL_DIR = os.path.join("..", "models")
file_path = os.path.join(MODEL_DIR, 'last.pth.tar')
utils.load_checkpoint(F, file_path)
# model_file = os.path.join(MODEL_DIR, "model.pt")


# if cuda_available:
# 	F.cuda()

# F.load_state_dict(torch.load(model_file, map_location = "cpu"))
F.eval()

print("Model Loaded!")
err_fract = Validate(10000)

print("Validation Error: ", err_fract)
