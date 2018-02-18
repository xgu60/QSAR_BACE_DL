import pandas as pd
import numpy as np

def preprocess(folder, shuffle, seed, pctl):
	#folder describe the path to folder contain data
	#shuffle (True or False), whether shuffle the data
	#seed was used to generate random state
	#pctl describe how you cut the data for training, validation
	print("the seed is: ")
	print(seed)
	folderpath = folder
	df = pd.read_csv(folderpath + "data/fingerprinters_4096.csv")	

	df.to_csv(folderpath + 'data/processed_data.csv', index=False)

	if shuffle:
		df = df.sample(frac=1, random_state=seed)
	#print(df.shape[0])
	df.iloc[:int(df.shape[0] * pctl), 0:].to_csv(folderpath + 'data/training.csv', index=False)
	df.iloc[int(df.shape[0] * pctl) : , 0:].to_csv(folderpath + 'data/validation.csv', index=False)
	#print(df.head())
	#print(df2.head())

def preprocess2(folder, shuffle, seed, pctls):
	#folder describe the path to folder contain data
	#shuffle (True or False), whether shuffle the data
	#seed was used to generate random state
	#pctl describe how you cut the data for training, validation and test
	
	folderpath = folder
	df = pd.read_csv(folderpath + "data/fingerprinters_4096.csv")
	if folder == "ccf/":
		df["ic50"] = - np.log10(df["ic50"] / 1E9)
	if folder == "pg/":
		df["ic50"] = - df["ic50"]

	df.to_csv(folderpath + 'data/processed_data.csv', index=False)

	if shuffle:
		df = df.sample(frac=1, random_state=seed)
	#print(df.shape[0])
	df.iloc[:int(df.shape[0] * pctls[0]), 1:].to_csv(folderpath + 'training.csv', index=False)
	df.iloc[int(df.shape[0] * pctls[0]) : int(df.shape[0] * pctls[1]), 1:].to_csv(folderpath + 'validation.csv', index=False)
	df.iloc[int(df.shape[0] * pctls[2]):, 2: ].to_csv(folderpath + 'test.csv', index=False)
	df.iloc[int(df.shape[0] * pctls[2]):, 1:2].to_csv(folderpath + 'test_label.csv', index=False)
	#print(df.head())
	#print(df2.head())

if __name__ == "__main__":
	preprocess("bace/", True, 1, 0.7)