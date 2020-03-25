#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.nn.utils.rnn import *
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time
from ctcdecode import CTCBeamDecoder
import datetime
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# In[2]:


N_STATES = 138
N_PHONEMES = N_STATES // 3

PHONEME_MAP = [
    '_',  # "+BREATH+"
    '+',  # "+COUGH+"
    '~',  # "+NOISE+"
    '!',  # "+SMACK+"
    '-',  # "+UH+"
    '@',  # "+UM+"
    'a',  # "AA"
    'A',  # "AE"
    'h',  # "AH"
    'o',  # "AO"
    'w',  # "AW"
    'y',  # "AY"
    'b',  # "B"
    'c',  # "CH"
    'd',  # "D"
    'D',  # "DH"
    'e',  # "EH"
    'r',  # "ER"
    'E',  # "EY"
    'f',  # "F"
    'g',  # "G"
    'H',  # "HH"
    'i',  # "IH"
    'I',  # "IY"
    'j',  # "JH"
    'k',  # "K"
    'l',  # "L"
    'm',  # "M"
    'n',  # "N"
    'G',  # "NG"
    'O',  # "OW"
    'Y',  # "OY"
    'p',  # "P"
    'R',  # "R"
    's',  # "S"
    'S',  # "SH"
    '.',  # "SIL"
    't',  # "T"
    'T',  # "TH"
    'u',  # "UH"
    'U',  # "UW"
    'v',  # "V"
    'W',  # "W"
    '?',  # "Y"
    'z',  # "Z"
    'Z',  # "ZH"
]


PHONEME_MAP_TEST = [
    ' ',
    '_',  # "+BREATH+"
    '+',  # "+COUGH+"
    '~',  # "+NOISE+"
    '!',  # "+SMACK+"
    '-',  # "+UH+"
    '@',  # "+UM+"
    'a',  # "AA"
    'A',  # "AE"
    'h',  # "AH"
    'o',  # "AO"
    'w',  # "AW"
    'y',  # "AY"
    'b',  # "B"
    'c',  # "CH"
    'd',  # "D"
    'D',  # "DH"
    'e',  # "EH"
    'r',  # "ER"
    'E',  # "EY"
    'f',  # "F"
    'g',  # "G"
    'H',  # "HH"
    'i',  # "IH"
    'I',  # "IY"
    'j',  # "JH"
    'k',  # "K"
    'l',  # "L"
    'm',  # "M"
    'n',  # "N"
    'G',  # "NG"
    'O',  # "OW"
    'Y',  # "OY"
    'p',  # "P"
    'R',  # "R"
    's',  # "S"
    'S',  # "SH"
    '.',  # "SIL"
    't',  # "T"
    'T',  # "TH"
    'u',  # "UH"
    'U',  # "UW"
    'v',  # "V"
    'W',  # "W"
    '?',  # "Y"
    'z',  # "Z"
    'Z',  # "ZH"
]


# In[3]:


X = np.load("wsj0_train.npy",encoding='bytes')
Y= np.load('wsj0_train_merged_labels.npy',encoding='bytes')
X_val = np.load("wsj0_dev.npy",encoding='bytes')
Y_val= np.load('wsj0_dev_merged_labels.npy',encoding='bytes')
X = [torch.LongTensor(n) for n in X]
Y = [torch.LongTensor(n) for n in Y]
X_val = [torch.LongTensor(n) for n in X_val]
Y_val = [torch.LongTensor(n) for n in Y_val]
X_lens = torch.LongTensor([len(seq) for seq in X])
Y_lens = torch.LongTensor([len(seq) for seq in Y])
X_lens_val = torch.LongTensor([len(seq) for seq in X_val])
Y_lens_val = torch.LongTensor([len(seq) for seq in Y_val])


# In[5]:


class TextDataset(Dataset):
    
    def __init__(self,x,y,lenx,leny):
        self.x = x
        self.y = y
        self.lenx = lenx
        self.leny = leny
    
    def __getitem__(self,i):
        xx = self.x[i]
        yy = self.y[i]
        lenxx = self.lenx[i]
        lenyy = self.leny[i]

        return xx,yy,lenxx,lenyy
    
    def __len__(self):
        return len(self.y)




# Collate function. Transform a list of sequences into a batch. Passed as an argument to the DataLoader.
# Returns data of the format seq_len x batch_size
def collate(seq_list):

    inputs = [torch.LongTensor(n[0]) for n in seq_list]
    targets = [torch.LongTensor(n[1]) for n in seq_list]
   
    input_lens = [len(seq) for seq in inputs]
    target_lens = [len(seq) for seq in targets]
    

    return pad_sequence(inputs),pad_sequence(targets, batch_first=True),input_lens, target_lens


# In[7]:


class Model(nn.Module):
    def __init__(self, in_vocab, out_vocab, embed_size, hidden_size):
        super(Model, self).__init__()
#         self.embed = nn.Embedding(in_vocab, embed_size)
        self.lstm = nn.LSTM(in_vocab, hidden_size, bidirectional=True,num_layers =3)
        self.output = nn.Linear(hidden_size * 2, out_vocab)
    
    def forward(self, X, lengths):
#         X = self.embed(X)
#         print(X.shape)
#         print(lengths)
        packed_X = pack_padded_sequence(X, lengths, enforce_sorted=False)
        packed_out = self.lstm(packed_X)[0]
        out, out_lens = pad_packed_sequence(packed_out)
        # Log softmax after output layer is required for use in `nn.CTCLoss`.
        out = self.output(out).log_softmax(2)
        return out, out_lens


# In[8]:


model = Model(40,47,4,512)
model = model.to(DEVICE)
# model.load_state_dict(torch.load(PATH))
# model.eval()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=1e-6)
split = 5000000
train_dataset = TextDataset(X,Y,X_lens,Y_lens)
val_dataset = TextDataset(X_val,Y_val,X_lens_val,Y_lens_val)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, collate_fn = collate)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=32, collate_fn = collate, drop_last=True)


# In[9]:


train_dataset[0][0].shape


# In[10]:


train_dataset[0][1].shape


# In[11]:



def train_epoch(model, optimizer, train_loader, val_loader):
    criterion = nn.CTCLoss()
    criterion = criterion.to(DEVICE)
    before = time.time()
    print("training", len(train_loader), "number of batches")
    for batch_idx, (inputs,targets,leninput,lentarget) in enumerate(train_loader):
        if batch_idx == 0:
            first_time = time.time()
            
        inputs = inputs.float()
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs,lenoutput = model(inputs,leninput) # 3D
        loss = criterion(outputs, targets, torch.tensor(lenoutput), torch.tensor(lentarget)) # Loss of the flattened outputs
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx == 0:
            print("Time elapsed", time.time() - first_time)
            
        if batch_idx % 100 == 0 and batch_idx != 0:
            after = time.time()
            print("Time: ", after - before)
            print("Loss per word: ", loss.item() / batch_idx)
            print("Perplexity: ", np.exp(loss.item() / batch_idx))
            after = before
    
    val_loss = 0
    batch_id=0
    for inputs,targets,leninput,lentarget in val_loader:
        batch_id+=1
        inputs = inputs.float()
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs,lenoutput = model(inputs,leninput)
        loss = criterion(outputs, targets, torch.tensor(lenoutput), torch.tensor(lentarget))
        val_loss+=loss.item()
    val_lpw = val_loss / batch_id
    print("\nValidation loss per word:",val_lpw)
    print("Validation perplexity :",np.exp(val_lpw),"\n")
    return val_lpw


# In[12]:


for i in range(1):
    model.train()
    train_epoch(model, optimizer, train_loader, train_loader)


# In[ ]:


# torch.save(model.state_dict(),'model_' + str(datetime.datetime.now()) + '.pt')


# In[13]:


class TestDataset(Dataset):
    
    def __init__(self,x,lenx):
        self.x = x
        self.lenx = lenx

    
    def __getitem__(self,i):
        xx = self.x[i]
        lenxx = self.lenx[i]
        
 
        return xx,lenxx
    
    def __len__(self):
        return len(self.x)


# In[14]:


def collate2(seq_list):

    inputs = [torch.LongTensor(n[0]) for n in seq_list]
    input_lens = [len(seq) for seq in inputs]

    return pad_sequence(inputs),input_lens


# In[15]:


X_test = np.load("wsj0_test.npy",encoding='bytes')
X_test = [torch.LongTensor(n) for n in X_test]
X_test_lens = torch.LongTensor([len(seq) for seq in X_test])




test_dataset = TestDataset(X_test,X_test_lens)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=len(X_test), collate_fn = collate2)


# In[16]:


outputs =0
lenoutput=0
for batch_idx, (inputs,leninput) in enumerate(test_loader):
    inputs = inputs.float()
    inputs = inputs.to(DEVICE)
    model.eval()
    outputs,lenoutput = model(inputs,leninput)

del inputs


# In[17]:


decoder = CTCBeamDecoder(PHONEME_MAP_TEST, beam_width=100, log_probs_input=True)

out, _, _, out_lens = decoder.decode(outputs.transpose(0, 1), lenoutput)

del outputs
del lenoutput


# In[33]:


output = []
for i in range(len(out)):
    out_seq = out[i, 0, :out_lens[i, 0]]
    out_ctc = ''
    for j in out_seq:
        out_ctc += PHONEME_MAP_TEST[j+1]
    
    output.append(out_ctc)


# In[35]:


output


# In[22]:


f = open("result/hw3p2_" + str(datetime.datetime.now()) + ".csv","w+")
f.writelines('id,Predicted\n')
    
for i in range(len(output)):
    f.writelines(str(i) +',' + output[i] + '\n')
    
f.close()


del best_seq
del best_pron


# In[28]:


optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, weight_decay=1e-6)
for i in range(10):
    model.train()
    train_epoch(model, optimizer, train_loader, train_loader)
    
outputs =0
lenoutput=0
for batch_idx, (inputs,leninput) in enumerate(test_loader):
    inputs = inputs.float()
    inputs = inputs.to(DEVICE)
    model.eval()
    outputs,lenoutput = model(inputs,leninput)

del inputs
    
    
decoder = CTCBeamDecoder(PHONEME_MAP_TEST, beam_width=100, log_probs_input=True)

out, _, _, out_lens = decoder.decode(outputs.transpose(0, 1), lenoutput)

del outputs
del lenoutput
    
output = []
for i in range(len(out)):
    out_seq = out[i, 0, :out_lens[i, 0]]
    out_ctc = ''
    for j in out_seq:
        out_ctc += PHONEME_MAP_TEST[j+1]
    
    output.append(out_ctc)


# In[ ]:


f = open("result/hw3p2_" + str(datetime.datetime.now()) + ".csv","w+")
f.writelines('id,Predicted\n')
    
for i in range(len(output)):
    f.writelines(str(i) +',' + output[i] + '\n')
    
f.close()


del best_seq
del best_pron

