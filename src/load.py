
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
	conjectures = []
	statements = []
	labels = []
	for datapoint in validation_dataset():
		conjecture = datapoint.conjecture
		statement = datapoint.statement
		label = datapoint.label

		for node_id, node_obj in conjecture.nodes.items(): # Find and replace unknowns
			if node_obj not in tokens_to_index.keys(): # UNKOWN token
				node_obj.token = "UNKNOWN"

		prediction_val = F([conjecture], [statement])
		_, prediction_label = torch.max(prediction_val, dim = 1)


		if cuda_available:
			prediction_label = prediction_label.cpu()
		prediction_label = prediction_label.numpy()

		if datapoint.label != prediction_label[0]:
			err_count += 1

		count += 1

		if count == num_datapoints:
			break

	print("Fraction of Incorrect Validations: ", err_count / count)

	return err_count / count



MODEL_DIR = os.path.join("saved_model", "models")
model_file = os.path.join(MODEL_DIR, "model.pt")


loss = nn.BCEWithLogitsLoss() # Binary Cross-Entropy Loss
F = FormulaNet(5, loss)

F.load_state_dict(torch.load(model_file, map_location = "cpu"))
F.eval()

print("Model Loaded!")
err_fract = Validate(300)

print("Validation Error: ", err_fract)