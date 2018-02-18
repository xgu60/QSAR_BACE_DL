import pandas as pd
import torch
from torch.utils.data.dataset import Dataset



class QSAR_Dataset(Dataset):
	'''read data from CSV file
	   Argument: CSV file name
	'''
	def __init__(self, csv_path):
		xy = pd.read_csv(csv_path)		
		self.len = xy.shape[0]	
		self.x_data = torch.from_numpy(xy.iloc[:, 2:].as_matrix()).float()
		self.y_data = torch.from_numpy(xy.iloc[:, 1:2].as_matrix()).float()

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len

def getValidationData(csv_path):
	xy = pd.read_csv(csv_path)	
	x_data = torch.from_numpy(xy.iloc[:, 2:].as_matrix()).float()
	y_data = torch.from_numpy(xy.iloc[:, 1:2].as_matrix()).float()
	return x_data, y_data

def getTestData(csv_path):
	df = pd.read_csv(csv_path)	
	data = torch.from_numpy(df.iloc[:, 1:2].as_matrix()).float()	
	return data




