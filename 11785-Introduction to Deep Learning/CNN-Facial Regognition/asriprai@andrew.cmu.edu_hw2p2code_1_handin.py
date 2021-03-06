./._hw2p2-classification.py                                                                         000644  000765  000024  00000001212 13553423645 021725  0                                                                                                    ustar 00asvinsripraiwalsupakit          staff                           000000  000000                                                                                                                                                                             Mac OS X            	   2  X     �                                      ATTR      �  $  f                 $     com.apple.lastuseddate#PS      4   5  )com.apple.metadata:kMDItemDownloadedDate   i   �  %com.apple.metadata:kMDItemWhereFroms   N   <  com.apple.quarantine �'�]    �a�    bplist00�3A��/���
                            bplist00�_chttp://localhost:8888/nbconvert/script/Documents/HW2/Part2/hw2p2-classification.ipynb?download=true_Nhttp://localhost:8888/notebooks/Documents/HW2/Part2/hw2p2-classification.ipynbq                            �q/0083;5dae27a5;Safari;8F89A306-4C2E-4D50-AAF8-F772B9721D9F                                                                                                                                                                                                                                                                                                                                                                                       hw2p2-classification.py                                                                             000644  000765  000024  00000045653 13553423645 021374  0                                                                                                    ustar 00asvinsripraiwalsupakit          staff                           000000  000000                                                                                                                                                                         
# coding: utf-8

# # Recitation - 6
# ___
# 
# * Custom Dataset & DataLoader
# * Torchvision ImageFolder Dataset
# * Residual Block
# * CNN model with Residual Block
# * Loss Fucntions (Center Loss and Triplet Loss)

# ## Imports

# In[1]:


import os
import numpy as np
from PIL import Image
import datetime

from matplotlib import pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ## Custom DataSet with DataLoader
# ___
# We have used a subset of the data given for the Face Classification and Verification problem in Part 2 of the homework

# In[2]:


class ImageDataset(Dataset):
    def __init__(self, file_list, target_list):
        self.file_list = file_list
        self.target_list = target_list
        self.n_class = len(list(set(target_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        label = self.target_list[index]
        return img, label


# #### Parse the given directory to accumulate all the images

# In[3]:


def parse_data(datadir):
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                ID_list.append(root.split('/')[-1])

    # construct a dictionary, where key and value correspond to ID and target
    uniqueID_list = list(set(ID_list))
    class_n = len(uniqueID_list)
    target_dict = dict(zip(uniqueID_list, range(class_n)))
    label_list = [target_dict[ID_key] for ID_key in ID_list]

    print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
    return img_list, label_list, class_n


# In[4]:


# img_list, label_list, class_n = parse_data('11-785hw2p2-f19/train_data/medium')
# img_list_dev, label_list_dev, class_n_dev = parse_data('11-785hw2p2-f19/validation_classification/medium')


# In[5]:


# trainset = ImageDataset(img_list, label_list)
# devset = ImageDataset(img_list_dev, label_list_dev)


# In[6]:


# train_data_item, train_data_label = trainset.__getitem__(0)


# In[7]:


# print('data item shape: {}\t data item label: {}'.format(train_data_item.shape, train_data_label))


# In[8]:


# train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1, drop_last=False)
# dev_dataloader = DataLoader(devset, batch_size=64, shuffle=True, num_workers=1, drop_last=False)


# In[9]:


f = open("11-785hw2p2-f19/test_order_classification.txt","r")
index_test = []
for i in f:
    index_test.append((i.replace('\n','')).replace('.jpg',''))
    
f.close()


# ## Torchvision DataSet and DataLoader

# In[10]:


# imageFolder_dataset = torchvision.datasets.ImageFolder(root='11-785hw2p2-f19/train_data/mini/', 
#                                                        transform=torchvision.transforms.ToTensor())


# In[11]:


# imageFolder_dataloader = DataLoader(imageFolder_dataset, batch_size=10, shuffle=True, num_workers=1)


# In[12]:


# imageFolder_dataset.__len__(), len(imageFolder_dataset.classes)


# ## Residual Block
# 
# Resnet: https://arxiv.org/pdf/1512.03385.pdf
# 
# Here is a basic usage of shortcut in Resnet

# In[13]:


class BasicBlock(nn.Module):

    def __init__(self, channel_size, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.shortcut = nn.Conv2d(channel_size, channel_size, kernel_size=1, stride=stride, bias=False)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(self.bn1(self.conv1(F.relu(self.bn1(self.conv1(self.bn1(self.conv1(F.relu(self.bn1(self.conv1(x)))))))))))))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# ## CNN Model with Residual Block 

# In[14]:


class Network(nn.Module):
    def __init__(self, num_feats, hidden_sizes, num_classes, feat_dim=10):
        super(Network, self).__init__()
        
        self.hidden_sizes = [num_feats] + hidden_sizes + [num_classes]
        
        self.layers = []
        for idx, channel_size in enumerate(hidden_sizes):
            self.layers.append(nn.Conv2d(in_channels=self.hidden_sizes[idx], 
                                         out_channels=self.hidden_sizes[idx+1], 
                                         kernel_size=3, stride=2, bias=False))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(BasicBlock(channel_size = channel_size))
            
        self.layers = nn.Sequential(*self.layers)
        self.linear_label = nn.Linear(self.hidden_sizes[-2], self.hidden_sizes[-1], bias=False)
        
        # For creating the embedding to be passed into the Center Loss criterion
        self.linear_closs = nn.Linear(self.hidden_sizes[-2], feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)
    
    def forward(self, x, evalMode=False):
        output = x
        output = self.layers(output)
            
        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])
        
        label_output = self.linear_label(output)
        label_output = label_output/torch.norm(self.linear_label.weight, dim=1)
        
        # Create the feature embedding for the Center Loss
        closs_output = self.linear_closs(output)
        closs_output = self.relu_closs(closs_output)

        return closs_output, label_output

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


# ### Training & Testing Model

# In[15]:


def train(model, data_loader, test_loader, task='Classification'):
    model.train()

    print(datetime.datetime.now())
    for epoch in range(numEpochs):
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(data_loader):
            
            
            feats, labels = feats.to(device), labels.to(device)

            
            optimizer.zero_grad()
            outputs = model(feats)[1]

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()

            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50))
                avg_loss = 0.0    
            
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        
        print(datetime.datetime.now())
        
        if task == 'Classification':
            val_loss, val_acc = test_classify(model, test_loader)
            train_loss, train_acc = test_classify(model, data_loader)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
        else:
            test_verify(model, test_loader)
        


def test_classify(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[1]
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        loss = criterion(outputs, labels.long())
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total


def test_verify(model, test_loader):
    raise NotImplementedError


# #### Dataset, DataLoader and Constant Declarations

# In[16]:


train_dataset = torchvision.datasets.ImageFolder(root='11-785hw2p2-f19/train_data/medium/', 
                                                 transform=torchvision.transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, 
                                               shuffle=True, num_workers=8)

dev_dataset = torchvision.datasets.ImageFolder(root='11-785hw2p2-f19/validation_classification/medium/', 
                                               transform=torchvision.transforms.ToTensor())
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=256, 
                                             shuffle=True, num_workers=8)


# In[17]:


numEpochs = 1
num_feats = 3

learningRate = 1e-2
weightDecay = 5e-5

hidden_sizes = [64,128,256]
num_classes = len(train_dataset.classes)
# print(num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[18]:


network = Network(num_feats, hidden_sizes, num_classes)
network.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)


# In[19]:



# numEpochs = 10
# network.train()
# network.to(device)
# train(network, train_dataloader, dev_dataloader)


# In[20]:


test_dataset = torchvision.datasets.ImageFolder(root='11-785hw2p2-f19/test_classification', 
                                               transform=torchvision.transforms.ToTensor())
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), 
                                             shuffle=False, num_workers=8)


# In[21]:


def test(model,test_dataloader):
    
    f = open("result/classification " + str(datetime.datetime.now()) + ".csv","w+")
    f.writelines('id,label\n')
    
    

    network.eval()




    for batch_num, (feats, labels) in enumerate(test_dataloader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[1]
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        pred_label = torch.tensor(np.zeros(len(pred_labels)))
        print(pred_labels)
        for i in range(len(pred_labels)):
             pred_label[i]=torch.tensor(int(train_dataset.classes[pred_labels[i]]))
       
        pred_labels = np.array(pred_label).astype(int)
        print(pred_labels)

        del feats
        del labels
        

    for i in range(len(pred_labels)):
        f.writelines(str(index_test[i]) +',' + str(int(pred_labels[i])) + '\n')
    
    del pred_labels


# In[22]:


# test(network2,test_dataloader)


# ## Center Loss
# ___
# The following piece of code for Center Loss has been pulled and modified based on the code from the GitHub Repo: https://github.com/KaiyangZhou/pytorch-center-loss
#     
# <b>Reference:</b>
# <i>Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.</i>

# In[23]:


class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, device=torch.device('cpu')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) +                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


# In[24]:


def train_closs(model, data_loader, test_loader, task='Classification'):
    model.train()

    print(datetime.datetime.now())
    for epoch in range(numEpochs):
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer_label.zero_grad()
            optimizer_closs.zero_grad()
            
            feature, outputs = model(feats)

            l_loss = criterion_label(outputs, labels.long())
            c_loss = criterion_closs(feature, labels.long())
            loss = l_loss + closs_weight * c_loss
            
            loss.backward()
            
            optimizer_label.step()
            # by doing so, weight_cent would not impact on the learning of centers
            for param in criterion_closs.parameters():
                param.grad.data *= (1. / closs_weight)
            optimizer_closs.step()
            
            avg_loss += loss.item()

            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50))
                avg_loss = 0.0    
            
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        
        if task == 'Classification':
            val_loss, val_acc = test_classify_closs(model, test_loader)
            train_loss, train_acc = test_classify_closs(model, data_loader)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
        else:
            test_verify(model, test_loader)
            
        print(datetime.datetime.now())


def test_classify_closs(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        feature, outputs = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        l_loss = criterion_label(outputs, labels.long())
        c_loss = criterion_closs(feature, labels.long())
        loss = l_loss + closs_weight * c_loss
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total


# In[25]:


numEpochs = 5
num_feats = 3

learningRate = 1e-2
weightDecay = 5e-5

closs_weight = 1
lr_cent = 0.5
feat_dim = 10

network2 = Network(num_feats, hidden_sizes, num_classes, feat_dim)
network2.apply(init_weights)

criterion_label = nn.CrossEntropyLoss()
criterion_closs = CenterLoss(num_classes, feat_dim, device)
optimizer_label = torch.optim.SGD(network2.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
# optimizer_label = torch.optim.Adam(network2.parameters(), lr=learningRate)
optimizer_closs = torch.optim.SGD(criterion_closs.parameters(), lr=lr_cent)


# In[26]:


# optimizer_label = torch.optim.SGD(network2.parameters(), lr=1e-4, weight_decay=weightDecay, momentum=0.9)
# optimizer_label = torch.optim.Adam(network2.parameters(), lr=1e-5)
numEpochs = 1
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)


# In[27]:


test(network2,test_dataloader)


# In[28]:


torch.save(network2.state_dict(),'model_' + str(datetime.datetime.now()) + '.pt')


# In[29]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
test(network2,test_dataloader)


# In[30]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
test(network2,test_dataloader)


# In[31]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
test(network2,test_dataloader)


# In[32]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
test(network2,test_dataloader)


# In[33]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
test(network2,test_dataloader)


# In[34]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
test(network2,test_dataloader)


# In[35]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
test(network2,test_dataloader)


# In[36]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
test(network2,test_dataloader)


# In[ ]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
test(network2,test_dataloader)


# In[ ]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
test(network2,test_dataloader)


# In[ ]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
test(network2,test_dataloader)

                                                                                     ./._hw2p2-verification.py                                                                           000644  000765  000024  00000001206 13553423710 021410  0                                                                                                    ustar 00asvinsripraiwalsupakit          staff                           000000  000000                                                                                                                                                                             Mac OS X            	   2  T     �                                      ATTR      �  $  b                 $     com.apple.lastuseddate#PS      4   5  )com.apple.metadata:kMDItemDownloadedDate   i   �  %com.apple.metadata:kMDItemWhereFroms   J   <  com.apple.quarantine �'�]    �s    bplist00�3A��/�Av`
                            bplist00�_ahttp://localhost:8888/nbconvert/script/Documents/HW2/Part2/hw2p2-verification.ipynb?download=true_Lhttp://localhost:8888/notebooks/Documents/HW2/Part2/hw2p2-verification.ipynbo                            �q/0083;5dae27c8;Safari;EDEC8156-1EE6-4CC0-98A9-F8E3EE993D2B                                                                                                                                                                                                                                                                                                                                                                                           hw2p2-verification.py                                                                               000644  000765  000024  00000045143 13553423710 021046  0                                                                                                    ustar 00asvinsripraiwalsupakit          staff                           000000  000000                                                                                                                                                                         
# coding: utf-8

# # Recitation - 6
# ___
# 
# * Custom Dataset & DataLoader
# * Torchvision ImageFolder Dataset
# * Residual Block
# * CNN model with Residual Block
# * Loss Fucntions (Center Loss and Triplet Loss)

# ## Imports

# In[1]:


import os
import numpy as np
from PIL import Image
import datetime

from matplotlib import pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ## Custom DataSet with DataLoader
# ___
# We have used a subset of the data given for the Face Classification and Verification problem in Part 2 of the homework

# In[2]:


class ImageDataset(Dataset):
    def __init__(self, file_list, target_list):
        self.file_list = file_list
        self.target_list = target_list
        self.n_class = len(list(set(target_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        label = self.target_list[index]
        return img, label


# #### Parse the given directory to accumulate all the images

# In[3]:


def parse_data(datadir):
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                ID_list.append(root.split('/')[-1])

    # construct a dictionary, where key and value correspond to ID and target
    uniqueID_list = list(set(ID_list))
    class_n = len(uniqueID_list)
    target_dict = dict(zip(uniqueID_list, range(class_n)))
    label_list = [target_dict[ID_key] for ID_key in ID_list]

    print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
    return img_list, label_list, class_n


# In[4]:


img_list, label_list, class_n = parse_data('11-785hw2p2-f19/train_data/medium')
img_list_dev, label_list_dev, class_n_dev = parse_data('11-785hw2p2-f19/validation_classification/medium')
img_list_test, label_list_test, class_n_test = parse_data('11-785hw2p2-f19/test_verification')


# In[5]:


img_title = [ img_list_test[i].replace('11-785hw2p2-f19/test_verification/','') for i in range(len(img_list_test))]


# In[6]:


img_idx = {}
for idx, k in enumerate(img_title):
    img_idx[k] = idx


# In[7]:


# img_idx['317662.jpg']


# In[8]:


trainset = ImageDataset(img_list, label_list)
devset = ImageDataset(img_list_dev, label_list_dev)
testset = ImageDataset(img_list_test, label_list_test)


# In[9]:


train_data_item, train_data_label = trainset.__getitem__(0)
testset.__getitem__(169391)
testset.__getitem__(0)[0]


# In[10]:


print('data item shape: {}\t data item label: {}'.format(train_data_item.shape, train_data_label))


# In[11]:


train_dataloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1, drop_last=False)
dev_dataloader = DataLoader(devset, batch_size=128, shuffle=True, num_workers=1, drop_last=False)
test_dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)


# In[12]:


f = open("11-785hw2p2-f19/verification.csv","r")
index_test1 = []
index_test2 = []
f.readline()
for i in f:
    j = i.split(",")
    k = j[0].split(" ")
    index_test1.append(k[0])
    index_test2.append(k[1])
    
f.close()


# In[13]:


# index_test2[-5:]


# ## Torchvision DataSet and DataLoader

# In[14]:


# imageFolder_dataset = torchvision.datasets.ImageFolder(root='11-785hw2p2-f19/train_data/mini/', 
#                                                        transform=torchvision.transforms.ToTensor())


# In[15]:


# imageFolder_dataloader = DataLoader(imageFolder_dataset, batch_size=10, shuffle=True, num_workers=1)


# In[16]:


# imageFolder_dataset.__len__(), len(imageFolder_dataset.classes)


# ## Residual Block
# 
# Resnet: https://arxiv.org/pdf/1512.03385.pdf
# 
# Here is a basic usage of shortcut in Resnet

# In[17]:


class BasicBlock(nn.Module):

    def __init__(self, channel_size, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.shortcut = nn.Conv2d(channel_size, channel_size, kernel_size=1, stride=stride, bias=False)


    def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(F.relu(self.bn1(self.conv1(x))))))
        out = F.relu(self.bn1(self.conv1(self.bn1(self.conv1(F.relu(self.bn1(self.conv1(self.bn1(self.conv1(F.relu(self.bn1(self.conv1(x)))))))))))))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    


# ## CNN Model with Residual Block 

# In[18]:


class Network(nn.Module):
    def __init__(self, num_feats, hidden_sizes, num_classes, feat_dim=10):
        super(Network, self).__init__()
        
        self.hidden_sizes = [num_feats] + hidden_sizes + [num_classes]
        
        self.layers = []
        for idx, channel_size in enumerate(hidden_sizes):
            self.layers.append(nn.Conv2d(in_channels=self.hidden_sizes[idx], 
                                         out_channels=self.hidden_sizes[idx+1], 
                                         kernel_size=3, stride=2, bias=False))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(BasicBlock(channel_size = channel_size))
            
        self.layers = nn.Sequential(*self.layers)
        self.linear_label = nn.Linear(self.hidden_sizes[-2], self.hidden_sizes[-1], bias=False)
        
        # For creating the embedding to be passed into the Center Loss criterion
        self.linear_closs = nn.Linear(self.hidden_sizes[-2], feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)
    
    def forward(self, x, evalMode=False):
        output = x
        output = self.layers(output)
            
        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])
        
        label_output = self.linear_label(output)
        label_output = label_output/torch.norm(self.linear_label.weight, dim=1)
        
        # Create the feature embedding for the Center Loss
        closs_output = self.linear_closs(output)
        closs_output = self.relu_closs(closs_output)

        return closs_output, label_output

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


# ### Training & Testing Model

# In[19]:


def train(model, data_loader, test_loader, task='Classification'):
    model.train()

    print(datetime.datetime.now())
    for epoch in range(numEpochs):
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(data_loader):
            
            
            feats, labels = feats.to(device), labels.to(device)

            
            optimizer.zero_grad()
            outputs = model(feats)[1]

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()

            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50))
                avg_loss = 0.0    
            
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        
        print(datetime.datetime.now())
        
        if task == 'Classification':
            val_loss, val_acc = test_classify(model, test_loader)
            train_loss, train_acc = test_classify(model, data_loader)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
        else:
            test_verify(model, test_loader)
        


def test_classify(model, test_loader):
#     model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[1]
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        loss = criterion(outputs, labels.long())
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total


def test_verify(model, test_loader):
    raise NotImplementedError


# #### Dataset, DataLoader and Constant Declarations

# In[20]:


# train_dataset = torchvision.datasets.ImageFolder(root='11-785hw2p2-f19/train_data/medium/', 
#                                                  transform=torchvision.transforms.ToTensor())
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, 
#                                                shuffle=True, num_workers=8)

# dev_dataset = torchvision.datasets.ImageFolder(root='11-785hw2p2-f19/validation_classification/medium/', 
#                                                transform=torchvision.transforms.ToTensor())
# dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=256, 
#                                              shuffle=True, num_workers=8)


# In[21]:


numEpochs = 1
num_feats = 3

learningRate = 1e-2
weightDecay = 5e-5

hidden_sizes = [64,128,256]
num_classes =2300
# num_classes = len(train_dataset.classes)
# print(num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[22]:


network = Network(num_feats, hidden_sizes, num_classes)
network.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)


# In[23]:



# numEpochs = 10
# network.train()
# network.to(device)
# train(network, train_dataloader, dev_dataloader)


# In[24]:


# test_dataset = torchvision.datasets.ImageFolder(root='11-785hw2p2-f19/test_verification', 
#                                                transform=torchvision.transforms.ToTensor())
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), 
#                                              shuffle=False, num_workers=8)


# In[25]:


# def test(model,test_dataloader):
    
#     f = open("result/classification " + str(datetime.datetime.now()) + ".csv","w+")
#     f.writelines('id,label\n')
    
    
#     for batch_num, (feats, labels) in enumerate(test_dataloader):
#         feats, labels = feats.to(device), labels.to(device)
#         outputs = model(feats)[1]
#         _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
#         pred_labels = pred_labels.view(-1)
#         pred_label = torch.tensor(np.zeros(len(pred_labels)))
#         print(pred_labels)
#         for i in range(len(pred_labels)):
#              pred_label[i]=torch.tensor(int(train_dataset.classes[pred_labels[i]]))
       
#         pred_labels = np.array(pred_label).astype(int)
#         print(pred_labels)

#         del feats
#         del labels
        

#     for i in range(len(pred_labels)):
#         f.writelines(str(index_test[i]) +',' + str(int(pred_labels[i])) + '\n')
    
#     del pred_labels


# In[26]:


# test(network2,test_dataloader)


# ## Center Loss
# ___
# The following piece of code for Center Loss has been pulled and modified based on the code from the GitHub Repo: https://github.com/KaiyangZhou/pytorch-center-loss
#     
# <b>Reference:</b>
# <i>Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.</i>

# In[27]:


class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, device=torch.device('cpu')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) +                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


# In[28]:


def train_closs(model, data_loader, test_loader, task='Classification'):
    model.train()

    print(datetime.datetime.now())
    for epoch in range(numEpochs):
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer_label.zero_grad()
            optimizer_closs.zero_grad()
            
            feature, outputs = model(feats)

            l_loss = criterion_label(outputs, labels.long())
            c_loss = criterion_closs(feature, labels.long())
            loss = l_loss + closs_weight * c_loss
            
            loss.backward()
            
            optimizer_label.step()
            # by doing so, weight_cent would not impact on the learning of centers
            for param in criterion_closs.parameters():
                param.grad.data *= (1. / closs_weight)
            optimizer_closs.step()
            
            avg_loss += loss.item()

            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50))
                avg_loss = 0.0    
            
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        
        if task == 'Classification':
            val_loss, val_acc = test_classify_closs(model, test_loader)
            train_loss, train_acc = test_classify_closs(model, data_loader)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
        else:
            test_verify(model, test_loader)
            
        print(datetime.datetime.now())


def test_classify_closs(model, test_loader):
#     model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        feature, outputs = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        l_loss = criterion_label(outputs, labels.long())
        c_loss = criterion_closs(feature, labels.long())
        loss = l_loss + closs_weight * c_loss
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total


# In[29]:


numEpochs = 5
num_feats = 3

learningRate = 1e-2
weightDecay = 5e-5

closs_weight = 1
lr_cent = 0.5
feat_dim = 10

network2 = Network(num_feats, hidden_sizes, num_classes, feat_dim)
network2.apply(init_weights)

criterion_label = nn.CrossEntropyLoss()
criterion_closs = CenterLoss(num_classes, feat_dim, device)
optimizer_label = torch.optim.SGD(network2.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
# optimizer_label = torch.optim.Adam(network2.parameters(), lr=learningRate)
optimizer_closs = torch.optim.SGD(criterion_closs.parameters(), lr=lr_cent)


# In[30]:


# optimizer_label = torch.optim.SGD(network2.parameters(), lr=1e-4, weight_decay=weightDecay, momentum=0.9)
# optimizer_label = torch.optim.Adam(network2.parameters(), lr=1e-5)
numEpochs = 1
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)


# In[31]:


def writefile():
    f = open("result/verification " + str(datetime.datetime.now()) + ".csv","w+")
    f.writelines('trial,score\n')


    for i in range(len(index_test1)):


        img1 = img_idx[index_test1[i]]
        img2 = img_idx[index_test2[i]]

        img1 = network2(testset.__getitem__(img1)[0].reshape(1,3,32,32).to(device))[1]
        img2 = network2(testset.__getitem__(img2)[0].reshape(1,3,32,32).to(device))[1]

        f.writelines(index_test1[i] + " " + index_test2[i] +","+ str(float(F.cosine_similarity(img1, img2, dim=1, eps=1e-8).cpu())) + "\n")



# In[32]:


# writefile()


# In[33]:


torch.save(network2.state_dict(),'model_' + str(datetime.datetime.now()) + '.pt')


# In[34]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)


# In[35]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
writefile()


# In[36]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)


# In[37]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
writefile()


# In[ ]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)


# In[ ]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
writefile()


# In[ ]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)


# In[ ]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)
writefile()


# In[ ]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)


# In[ ]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)


# In[ ]:


numEpochs = 5
network2.train()
network2.to(device)
train_closs(network2, train_dataloader, dev_dataloader)

                                                                                                                                                                                                                                                                                                                                                                                                                             ./._writeup.txt                                                                                     000644  000765  000024  00000000752 13553424434 017665  0                                                                                                    ustar 00asvinsripraiwalsupakit          staff                           000000  000000                                                                                                                                                                             Mac OS X            	   2  �     �    TEXTMSWD                         ATTR      �  4   �                 4     com.apple.lastuseddate#PS      D   *  $com.apple.metadata:_kMDItemUserTags    n   Y  7com.apple.metadata:kMDLabel_okxn6tisc6h55vvpiirnm72ste     �   #  com.apple.quarantine !)�]    �2w"    bplist00�                            	�xn���[غX�I�D}�������7Щ�v~/��4�Pr�_ݷ�����pya�������@fH;y^���3��%�շ��N}��xq/0082;5dae291d;Microsoft\x20Word;                       writeup.txt                                                                                         000644  000765  000024  00000000763 13553424434 017315  0                                                                                                    ustar 00asvinsripraiwalsupakit          staff                           000000  000000                                                                                                                                                                         Summarize of the Model SpecificationModel use: Resnet 3 Basic Blocks( 6x conv  64  + 6x conv  128 + 6x conv  256)Loss: Centre loss (Learning Rate = 0.5)Centre Loss Weight: 1Batch Size: 256Number of Feature: 3Feature Dimension: 10Optimizer: SGD (Learning Rate = 1e-2, Weight Decay = 5e-5)Number of epochs to get such accuracy: ~10Approach used for verification: Cosine similarity of the output vector of each image (class output before Softmax)Asvin Sripraiwalsupakit AndrewID: asriprai                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             