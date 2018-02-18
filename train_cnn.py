from data_loader import QSAR_Dataset, getValidationData, getTestData
from net_structure import DNN, CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train_model(folder, visual, epoch, batchSize, savePoints):
	TURNON_VISUAL = visual
	EPOCH_NUM = epoch
	BATCH_SIZE = batchSize
	FEATURES = 4096
	OUTPUTS = 1
	folderpath = folder
	TRAIN_FILE = "data/training.csv"
	VALIDATION_FILE = "data/validation.csv"	
	NET_SAVE_POINTS = savePoints


	data = QSAR_Dataset(folderpath + TRAIN_FILE)	
	train_data = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True, 
		num_workers=2)
	val_data, val_label = getValidationData(folderpath + VALIDATION_FILE)
	

	net = CNN().cuda()
	val_data = Variable(val_data.view((val_data.size(0),1,64,64)).cuda())
	val_label = Variable(val_label.cuda())

	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr=0.05, betas=(0.9, 0.99))

	if TURNON_VISUAL:
		epochs = []
		train = []
		validation = []
		plt.ion()
		plt.xlabel('epoch', fontsize=16)
		plt.ylabel('mse loss', fontsize=16)
		plt.yscale('log')
		plt.grid(True)

	for epoch in range(EPOCH_NUM):
		for i, (inputs, labels) in enumerate(train_data):
			net.train()
			inputs, labels = Variable(inputs.view((inputs.size(0),1,64,64)).cuda()), Variable(labels.cuda())
			y_pred = net(inputs)
			loss = criterion(y_pred, labels)					

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()	

			if epoch in NET_SAVE_POINTS and i == 0:
				torch.save(net, folderpath + 'net/epoch' + str(epoch) + '.pkl')

			if epoch % 5 == 0 and i == 0:
				#evaluate the trained network
				net.eval()
				y_pred = net(inputs)
				train_loss = criterion(y_pred, labels).data

				val_pred = net(val_data)
				val_loss = criterion(val_pred, val_label).data
				print((train_loss[0], val_loss[0]))
			
				if TURNON_VISUAL:
					#update plot					
					epochs.append(epoch)
					train.append(train_loss[0])
					validation.append(val_loss[0])		
					plt.plot(epochs, train, 'b-')
					plt.plot(epochs, validation, 'r-')
					plt.pause(0.01)

	if TURNON_VISUAL:
		plt.plot(epochs, train, 'b-', label='training')
		plt.plot(epochs, validation, 'r-', label='validation')
		plt.legend(loc='upper right', shadow=True)
		plt.ioff()
		
		plt.show()

if __name__ == "__main__":
	train_model("bace/", True, 101, 32, [80,90,100])

	
	




	 
		    

    

    





