#%%
#This script is used to predict hyrophobicity scores using transformer-based NN

###Import packages ########################
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud
import math
import time
import matplotlib as plt

import d2l
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt
from pysmilesutils.pysmilesutils.tokenize import SMILESAtomTokenizer
from torch.utils.data import DataLoader

print("Using PyTorch backend")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")




##########################    misc functions    ###########################
#.isnumeric doens't work on negative or decimal numbers (annoying), so this function takes is place to make sure
#input values are only floats.
def is_float(num_in):
    try: 
        float(num_in)
        return True
    except ValueError:
        return False

##########################    misc functions end   ###########################




##### Data wrangling and data prep ########

print("Reading ChEMB data")

#hardcoded input. You can change it here for wherever the file is for you. 
file_name="chem_data.csv"

input_chem_data = pd.read_csv(file_name, usecols = ["Smiles", "AlogP"], header=0)


#filter the data to remove any empty values or those that aren't numbers 
input_chem_data = input_chem_data[(input_chem_data["Smiles"] != "") & 
                                  (input_chem_data.Smiles.str.len() < 1000) & 
                                   (input_chem_data["AlogP"].notna()) &
                                   ((input_chem_data["AlogP"].apply(is_float) | (input_chem_data["AlogP"].str.isnumeric()))) 
                                  ]

input_chem_data = input_chem_data.reset_index(drop=True)

smiles = input_chem_data["Smiles"].astype("string") #convert to string

alogp = input_chem_data["AlogP"].astype(float) #convert column to float
#alogp = input_chem_data[is_float(input_chem_data["AlogP"])

alogp = (alogp - np.min(alogp)) / (np.max(alogp) - np.min(alogp)) #normalize the values between 0 and 1 so they can be used 

smiles = smiles.head(50000)
alogp = alogp.head(50000)



tokenizer = SMILESAtomTokenizer(smiles=list(smiles))  #tokenize the smiles to numbers for use

smiles_array = tokenizer(smiles) #get the tokenized values



tensor_lengths = []

for j in range(len(smiles_array)):
    tensor_lengths.append(len(smiles_array[j]))

largest_tensor_size = max(tensor_lengths)


for k in range(len(smiles_array)):

    pad_length = largest_tensor_size - len(smiles_array[k])
    
    smiles_array[k] = nn.functional.pad(smiles_array[k], pad=(0,pad_length), mode='constant', value=0)
    




smiles_tensor = torch.stack(smiles_array)





alogp_tensor = torch.tensor(alogp.values.astype(float), dtype=torch.float32)
alogp_tensor = torch.chunk(alogp_tensor,len(input_chem_data))
alogp_tensor = torch.stack(alogp_tensor)





custom_dataset = torch.utils.data.TensorDataset(smiles_tensor, alogp_tensor)

input_dim = len(custom_dataset[0][0])
#print(input_dim)
#print(len(custom_dataset[0][0]))
#exit()
model = nn.Sequential(
    nn.Linear(input_dim,50),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(50,5),
    nn.ReLU(),
    nn.Linear(5,40),
    nn.ReLU(),
    #nn.Dropout(),
    nn.Linear(40,1),
    nn.Sigmoid()
    
)


#loss_fn = nn.BCELoss()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs=100
batch_size=100

#custom_dataset = torch.utils.data.TensorDataset(x, y)
train_len = int(len(custom_dataset)*0.8)
train_set, test_set = tud.random_split(custom_dataset, [train_len, len(custom_dataset)-train_len])


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

#train_loader_y = DataLoader(train_set_y, batch_size=batch_size, shuffle=True)
#test_loader_y = DataLoader(test_set_y, batch_size=batch_size, shuffle=False)

#print(train_loader_x, train_loader_y)

model.train()
num_correct = 0
num_samples = 0
total_loss = 0

for epoch in range(n_epochs+1):
    for data,targets in train_loader:
            #print("firstbatch", batch[0])
            #print("second batch", batch[1])

            #exit() 
            #data,targets = batch[:, 0], batch[:, 1]
        #data, targets = batch[0], batch[1]
        
        

        data = data.to(device=device)
        data = torch.tensor(data).to(torch.float32)
        #data = data.long()

        #print(data.shape)

        targets = targets.to(device=device)
        targets = torch.tensor(targets).to(torch.float32)
        #targets = targets.long()

        
        #print(data, targets)
            

        pred = model(data)

        #pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, targets)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()

        num_correct += (pred.round() == targets.round()).sum().item()
    #print(num_correct)

    
        num_samples += pred.size(0)
    

    acc = float(num_correct) / float(num_samples) * 100
    print(f'Finished epoch {epoch}, latest loss {loss}, accuracy {acc}')
    

num_correct = 0
num_samples = 0
total_loss = 0
#model.eval()
for x_test, y_test in test_loader:

    x_test = x_test.to(device=device)
    x_test = torch.tensor(x_test).to(torch.float32)

    y_test = y_test.to(device=device)
    y_test = torch.tensor(y_test).to(torch.float32)
    #y_test = y_test.long()

    #print(x_test.size(0))
    #print(y_test.size(0))

    predictions=model(x_test)

    #_, predictions = eval_score.max(1)
    

    print(f'Predictions are: {predictions},\n Actual values are: {y_test}')
    

    num_correct += (predictions.round() == y_test.round()).sum().item()
    #print(num_correct)

    
    num_samples += predictions.size(0)
    print(f'Num correct predictions = {num_correct}, num things = {num_samples}')

acc = float(num_correct) / float(num_samples) * 100
print(acc)

print("Model accuracy: %.2f%%" % (acc))

