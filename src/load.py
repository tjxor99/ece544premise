import utils
import torch
import torch.nn as nn

import numpy as np
from model import FormulaNet
from dataset import validation_dataset, get_token_dict_from_file, test_dataset
import os


# os.makedirs('models', True) # Directory to save / load models
def Validate(num_datapoints):
	tokens_to_index = get_token_dict_from_file()

	err_count = 0
	count = 0
	# for datapoint in validation_dataset():
	for datapoint in test_dataset():
		conjecture = datapoint.conjecture
		statement = datapoint.statement
		label = datapoint.label

		conjecture_statement = [conjecture, statement]

		prediction_val = F([conjecture_statement])
		_, prediction_label = torch.max(prediction_val, dim = 1)

		if cuda_available:
			prediction_label = prediction_label.cpu()
		prediction_label = prediction_label.numpy()

		if datapoint.label != prediction_label[0]:
			err_count += 1

		count += 1

		if count % 100 == 0:
			print("Count: ",count)

#		if count == num_datapoints:
#			break

	print("Fraction of Incorrect Validations: ", err_count / count)

	return err_count / count


cuda_available = torch.cuda.is_available()

num_steps = int(input("Number of Update Iterations:\n"))
F = FormulaNet(num_steps, cuda_available)

# Load Model
#MODEL_DIR = os.path.join("..", "models")
MODEL_DIR = os.path.join("..", "models2")
file_path = os.path.join(MODEL_DIR, 'last.pth.tar')
utils.load_checkpoint(F, file_path, cuda_available)
# model_file = os.path.join(MODEL_DIR, "model.pt")


# if cuda_available:
# 	F.cuda()

# F.load_state_dict(torch.load(model_file, map_location = "cpu"))
F.eval()

print("Model Loaded!")
err_fract = Validate(5000)

print("Validation Error: ", err_fract)
