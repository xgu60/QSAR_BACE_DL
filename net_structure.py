import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
	''' a simple dnn
	arguments: n_features e.g 1000
	           n_output e.g 1
	'''

	def __init__(self, n_features, n_output):
		super(DNN, self).__init__()
		self.h1 = nn.Linear(n_features, 4000)
		nn.init.xavier_uniform(self.h1.weight)
		self.h1_bn = nn.BatchNorm1d(4000)
		self.h1_relu = nn.ReLU()
		self.h1_dropout = nn.Dropout(p=0.90)
		self.h2 = nn.Linear(4000, 2000)
		nn.init.xavier_uniform(self.h2.weight)
		self.h2_bn = nn.BatchNorm1d(2000)
		self.h2_relu = nn.ReLU()
		self.h2_dropout = nn.Dropout(p=0.75)
		self.h3 = nn.Linear(2000, 1000)
		nn.init.xavier_uniform(self.h3.weight)
		self.h3_bn = nn.BatchNorm1d(1000)
		self.h3_relu = nn.ReLU()
		self.h3_dropout = nn.Dropout(p=0.50)						
		self.output = nn.Linear(1000, n_output)

	def forward(self, x):
		x = self.h1(x)
		x = self.h1_bn(x)
		x = self.h1_relu(x)
		x = self.h1_dropout(x)
		x = self.h2(x)
		x = self.h2_bn(x)
		x = self.h2_relu(x)
		x = self.h2_dropout(x)
		x = self.h3(x)
		x = self.h3_bn(x)
		x = self.h3_relu(x)	
		x = self.h3_dropout(x)					
		return self.output(x)

class CNN(nn.Module):

	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=1,
				out_channels=16,
				kernel_size=5,
				stride=1,
				padding=2
				),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=4)
			)
		
		self.conv2 = nn.Sequential(
			nn.Conv2d(
				in_channels=16,
				out_channels=32,
				kernel_size=3,
				stride=1,
				padding=1
				),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
			)
		
		self.conv3 = nn.Sequential(
			nn.Conv2d(
				in_channels=32,
				out_channels=64,
				kernel_size=3,
				stride=1,
				padding=1
				),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)			
			)

		self.fc = nn.Sequential(
			nn.Linear(1024, 1000),
			nn.BatchNorm1d(1000),
			nn.ReLU(),
			nn.Dropout(p=0.50),
			nn.Linear(1000, 1)			
			)

		'''	
		self.fc = nn.Sequential(
			nn.Linear(1024, 1000),
			nn.BatchNorm1d(1000),
			nn.ReLU(),
			nn.Dropout(p=0.75),
			nn.Linear(1000, 1000),
			nn.BatchNorm1d(1000),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(1000, 1000),
			nn.BatchNorm1d(1000),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(1000, 1)
			)
		'''
		
	

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		
		x = x.view(x.size(0), -1)
		output = self.fc(x)
		return output





if __name__ == "__main__":
	net = DNN(777, 10)	
	print(net)
	

