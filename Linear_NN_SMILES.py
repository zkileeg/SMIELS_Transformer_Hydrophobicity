#%%
#This script is used to predict hyrophobicity scores using a simple linear-regression NN
#This is part of me testing different models to see which perform well using a smiles-based dataset
'''
The steps are simple:

Step 1: get data in and format it 
Step 2: tokenize the smiles files. Need to get numeric data for use
Step 3: set the model
Step 4: train and test



'''

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


#function is meant to pad the tensor with 0 to make it usable. otherwise the different lengths will mess up the training
def pad_tensor(array_in):
    tensor_lengths = []   #initialize tensor

    for j in range(len(array_in)):
        tensor_lengths.append(len(array_in[j]))

    largest_tensor_size = max(tensor_lengths)


    for k in range(len(array_in)):

        pad_length = largest_tensor_size - len(array_in[k])
        
        array_in[k] = nn.functional.pad(array_in[k], pad=(0,pad_length), mode='constant', value=0)
    
    return array_in
##########################    misc functions end   ###########################




##### Data wrangling and data prep ########

print("Reading ChEMB data")

#hardcoded input. This script is purposely meant to be used specifically on this data set and not multipurpose.
####However, if you want to use it for a different dataset, change input here. ###
file_name="chem_data.csv"

input_chem_data = pd.read_csv(file_name, usecols = ["Smiles", "AlogP"], header=0)


#filter the data to remove any empty values or those that aren't numbers. Essentially parse the data to remove stuff that 
#will mess up the script. 
input_chem_data = input_chem_data[(input_chem_data["Smiles"] != "") & 
                                  (input_chem_data.Smiles.str.len() < 1000) & 
                                   (input_chem_data["AlogP"].notna()) &
                                   ((input_chem_data["AlogP"].apply(is_float) | (input_chem_data["AlogP"].str.isnumeric()))) 
                                  ]

input_chem_data = input_chem_data.reset_index(drop=True) #fix index so it starts at 0 again instead of random numbers

smiles = input_chem_data["Smiles"].astype("string") #convert to string

alogp = input_chem_data["AlogP"].astype(float) #convert column to float
#alogp = input_chem_data[is_float(input_chem_data["AlogP"])

alogp = (alogp - np.min(alogp)) / (np.max(alogp) - np.min(alogp)) #normalize the values between 0 and 1 so they can be used 

#this is meant to subset for training testing. The actual dataset is massive so a subset is good. 
smiles = smiles.head(500)
alogp = alogp.head(500)



tokenizer = SMILESAtomTokenizer(smiles=list(smiles))  #tokenize the smiles to numbers for use

smiles_array = tokenizer(smiles) #get the tokenized values


smiles_array = pad_tensor(smiles_array)

smiles_tensor = torch.stack(smiles_array)

#print(smiles_tensor)




alogp_tensor = torch.tensor(alogp.values.astype(float), dtype=torch.float32)
alogp_tensor = torch.chunk(alogp_tensor,len(input_chem_data))
alogp_tensor = torch.stack(alogp_tensor)





custom_dataset = torch.utils.data.TensorDataset(smiles_tensor, alogp_tensor)

input_dim = len(custom_dataset[0][0])

###set the model. Model has four linear layers and one dropout layer with ReLU layers in between 
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

###tune some hyperparamater stuff
n_epochs=100
batch_size=100

#split dataset into train and test sets, then create the dataloader
train_len = int(len(custom_dataset)*0.8)
train_set, test_set = tud.random_split(custom_dataset, [train_len, len(custom_dataset)-train_len])


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


###training model
model.train()
num_correct = 0
num_samples = 0
total_loss = 0

for epoch in range(n_epochs+1):
    for data,targets in train_loader:
            
        data = data.to(device=device)
        data = torch.tensor(data).to(torch.float32)
        

        targets = targets.to(device=device)
        targets = torch.tensor(targets).to(torch.float32)
        
            

        pred = model(data)
        loss = loss_fn(pred, targets)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()

        num_correct += (pred.round() == targets.round()).sum().item()
        num_samples += pred.size(0)
    

    acc = float(num_correct) / float(num_samples) * 100
    print(f'Finished epoch {epoch}, latest loss {loss}, accuracy {acc}')
###end training


####testing####
model.eval()
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
    

   # print(f'Predictions are: {predictions},\n Actual values are: {y_test}')
    

    num_correct += (predictions.round() == y_test.round()).sum().item()
    #print(num_correct)

    
    num_samples += predictions.size(0)
    print(f'Num correct predictions = {num_correct}, num samples = {num_samples}')
#end test

acc = float(num_correct) / float(num_samples) * 100
print("Model accuracy: %.2f%%" % (acc))

