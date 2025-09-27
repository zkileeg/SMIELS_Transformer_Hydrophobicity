#This script is used to predict hyrophobicity scores using transformer-based NN

###Import packages ########################
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
from pysmilesutils.pysmilesutils.tokenize import SMILESAtomTokenizer
from torch.utils.data import DataLoader

##############################


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
#file_name = "C:/Users/kileegza/Documents/Class/Dat112_MachineLearning/data.csv"

#get the file in
#input_chem_data = pd.read_csv(file_name, sep = ';', usecols = ["Smiles", "AlogP"], header=0)
input_chem_data = pd.read_csv(file_name, usecols = ["Smiles", "AlogP"], header=0)
#input_chem_data = pd.read_csv("C:/Users/kileegza/Documents/Class/Dat112_MachineLearning/chem_properties.csv", header=0)




#print(input_chem_data)


#print(input_chem_data)
#input_chem_data = pd.read_csv("/mnt/c/Users/kileegza/Documents/Class/Dat112_MachineLearning/chem_properties.csv", header=0)
input_chem_data = input_chem_data[(input_chem_data["Smiles"] != "") & 
                                  (input_chem_data.Smiles.str.len() < 1000) & 
                                   (input_chem_data["AlogP"].notna()) &
                                   ((input_chem_data["AlogP"].apply(is_float) | (input_chem_data["AlogP"].str.isnumeric()))) 
                                  ]

input_chem_data = input_chem_data.reset_index(drop=True)

smiles = input_chem_data["Smiles"].astype("string")
print(smiles.dtypes)

#exit()
 
#print(input_chem_data)

smiles = smiles.head(5)

input_chem_data = input_chem_data.head(5)

tokenizer = SMILESAtomTokenizer(smiles=list(smiles))

smiles_array = tokenizer(smiles)

#print(smiles_array[4])


#for i in range(len(smiles)):
    #print(input_chem_data.loc[i,"Smiles"])
#    input_chem_data.loc[i,"Smiles"] = tokenizer(smiles[i])

#smiles_array = input_chem_data["Smiles"]


#print(smiles_array)



#print(smiles_array[1][0])

tensor_lengths = []

for j in range(len(smiles_array)):
    tensor_lengths.append(len(smiles_array[j]))

largest_tensor_size = max(tensor_lengths)


for k in range(len(smiles_array)):

    pad_length = largest_tensor_size - len(smiles_array[k])
    #print("Pad length is", pad_length)
    smiles_array[k] = nn.functional.pad(smiles_array[k], pad=(0,pad_length), mode='constant', value=0)
    #print(smiles_array[k][0])
    #print(len(smiles_array[k][0]))

#print(len(smiles_array[2]))



#smiles_test = []
#for sublist in smiles_array:
    #sublist = sublist.to(torch.float)
#    smiles_test.extend(sublist)
    

#thingy = np.array(smiles_test)




smiles_tensor = torch.stack(smiles_array)
#smiles_tensor = smiles_test.to(torch.float)
#smiles_tensor = smiles_array




alogp_tensor = torch.tensor(input_chem_data["AlogP"].values.astype(float), dtype=torch.float32)
alogp_tensor = torch.chunk(alogp_tensor,len(input_chem_data))
alogp_tensor = torch.stack(alogp_tensor)

#test_tensor = torch.stack(smiles_tensor, alogp_tensor, dim=1)

print(smiles_tensor)
print(smiles_tensor.shape)

print(alogp_tensor)
print(alogp_tensor.shape)

#print (test_tensor)

#dataset_test = torch.cat(smiles_tensor, alogp_tensor, dim=1)

#custom_dataset = smiles_tensor + alogp_tensor



custom_dataset = torch.utils.data.TensorDataset(smiles_tensor, alogp_tensor)

print(custom_dataset)












##### Data wrangling and data prep  end ########


######### Hyper paramters   #############

#num_tokens, num_inputs, num_heads, hidden_size, num_layers
num_tokens = largest_tensor_size   #had to get total number of tokens before padding, so can reuse it here
#num_inputs = len(input_chem_data)   #the number of inputs will be the length of the input dataset
num_inputs = 1
num_heads = 5
num_layers = 2
hidden_size = 256
dropout_num = 0.5

learning_rate = 0.001
batch_size = 2
num_epochs = 2

#print(num_inputs)

print(type(num_tokens))
print(type(num_inputs))







######### hyper parameters end

###Positional encoding to give relative information about position of tokens in the sequence 


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 0::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


#########################     setup model       ###########################

        

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        #self.flatten = nn.Flatten()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp 
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()


    def generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))
    

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)


    def forward(self, src, has_mask=True):
       if has_mask:
           device = src.device
           if self.src_mask is None or self.src_mask.size(0) != len(src):
               mask = self.generate_square_subsequent_mask(len(src)).to(device)
               self.src_mask = mask
               
       else:
         self.src_mask = None


       src = self.input_emb(src) * math.sqrt(self.ninp)
       src = self.pos_encoder(src)
       #output = self.encoder(src, mask=self.src_mask)
       #output = self.decoder(src, self.src_mask)
       return nn.functional.log_softmax(src, dim=-1)

model = TransformerModel(num_tokens, num_inputs, num_heads, hidden_size, num_layers, dropout_num).to(device)


########################## model setup end ################################################


#### load data #####

#x_train, x_test, y_train, y_test = train_test_split(smiles_tensor, alogp_tensor, test_size = 0.2, random_state=42)
#convert into a format that will work. Needs to be numerical value in and out. 
#x_train = x_train.astype('int8')
#y_train = y_train.astype('float64')
#x_test = x_test.astype('int8')
#y_test = y_test.astype('float64')

#print("x train = ", x_train, "\n xtest = ", x_test, "\n ytrain = ", y_train, "\n y test = ", y_test)

#print ("x train dtype = ", x_train.dtype)



#train_loader = DataLoader(list(zip(x_train,y_train)),shuffle=True, batch_size=batch_size)
#test_loader = DataLoader(list(zip(x_train,y_train)), shuffle=True, batch_size=batch_size)

train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

#SmilesData = CustomDataset(features=smiles_tensor, labels=alogp_tensor)
#train_load = DataLoader(SmilesData, batch_size=batch_size, shuffle=True)
print(train_loader)

for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}:")
    print("Features shape:", batch_features.shape)
    print("Labels shape:", batch_labels.shape)
    print("First few features:", batch_features[:2])
    print("First few labels:", batch_labels[:2])



##################

##### Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)



####### Model training ############


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    

print("Training model")


        
model.train()
for epoch in range(num_epochs):
    print ("epoch:", epoch)
    for batch_idx, (data, targets) in enumerate(train_loader):
        
        data = data.to(device=device)
        data = torch.tensor(data).to(torch.int64)

        #print(data)

        targets = targets.to(device=device)
        targets = targets.long()

        #print(targets)

        

        scores = model(data)
        loss = loss_fn(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



# test model and evaluate 

model.eval()
y_pred=model(x_test)

print(y_pred)
acc = (y_pred.round() == y_test.float().mean())
print(acc)
acc=float(acc)

print("Model accuracy: %.2f%%" % (acc*100))


###### end ###########