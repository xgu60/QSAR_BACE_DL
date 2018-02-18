
import numpy as np
import pandas as pd
from preprocess import preprocess
from train import train_model
from test import test_model

if __name__ == "__main__":
	folderpath = "bace/"
	shuffle = True
	randomSeeds = range(6,7)
	train_pctl = 0.7
	train_vis = True
	epoch_num = 51
	batch_size = 128
	test_vis = True
	sp = [40, 45, 50]
	train_r2s = []
	val_r2s = []
	aucs = []
	for seed in randomSeeds:
		preprocess(folder=folderpath, shuffle=shuffle, seed=seed, pctl=train_pctl)
		train_model(folder=folderpath, visual=train_vis, epoch=epoch_num, batchSize=batch_size, savePoints=sp)
		train_r2, val_r2, auc = test_model(folder=folderpath, visual=test_vis, savePoints=sp)
		train_r2s.append(train_r2)
		val_r2s.append(val_r2)
		aucs.append(auc)
		#store res in temp file
		res = {"train_r2" : train_r2s, 
			"val_r2" : val_r2s, 
			"auc" : aucs}
		df = pd.DataFrame(res, columns=["train_r2", "val_r2", "auc"])
		df.loc["mean"] = df.mean()
		df.to_csv("exp_temp.csv")

	
	#print(train_r2s)
	train_r2_mean = np.mean(train_r2s)	
	print("train r squared mean: ")
	print(train_r2_mean)

	#print(val_r2s)
	val_r2_mean = np.mean(val_r2s)	
	print("validation r squared mean: ")
	print(val_r2_mean)

	#print(aucs)
	auc_mean = np.mean(aucs)	
	print("AUC mean: ")
	print(auc_mean)	

	res = {"seeds": randomSeeds, 
			"train_r2" : train_r2s, 
			"val_r2" : val_r2s, 
			"auc" : aucs}
	df = pd.DataFrame(res, columns=["seeds", "train_r2", "val_r2", "auc"])
	df.loc["mean"] = df.mean()
	df.to_csv("exp_results.csv")

	