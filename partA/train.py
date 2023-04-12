import wandb
import torch 
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn  import functional
from pytorch_lightning.loggers import WandbLogger
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import random


class CNN(pl.LightningModule):
  def __init__(self,filters,activation,BatchNorm,dropout,learning_rate,input_size,kernel_size,pool_kernel_size,pool_stride):
    dense_size = input_size
    for i in filters:
      dense_size = (dense_size-kernel_size+1-pool_kernel_size)//pool_stride +1
    self.dense_size = dense_size
    super(CNN,self).__init__()
    self.train_step_acc = []
    self.train_step_loss = []
    self.val_step_acc = []
    self.val_step_loss = []

    self.learning_rate = learning_rate
    layers = []
    layers.append(nn.Conv2d(3,filters[0],kernel_size = kernel_size,stride = 1,padding = 0))
    layers.append(nn.MaxPool2d(kernel_size = pool_kernel_size,stride = pool_stride))
    layers.append(activation)
    for i in range(0,3):
      layers.append(nn.Conv2d(filters[i],filters[i+1],kernel_size = kernel_size,stride = 1,padding = 0))
      layers.append(nn.MaxPool2d(kernel_size = pool_kernel_size,stride = pool_stride))
      layers.append( activation)

    layers.append(nn.Conv2d(filters[3],filters[4],kernel_size = kernel_size,stride = 1,padding = 0))
    layers.append(nn.MaxPool2d(kernel_size = pool_kernel_size,stride = pool_stride))
    layers.append( activation)
    layers.append(nn.Flatten())
    
    if(BatchNorm == True):
      layers.append(nn.BatchNorm1d(filters[4]*self.dense_size*self.dense_size))
    layers.append(nn.Dropout(p=dropout))

    layers.append(nn.Linear(filters[4]*self.dense_size*self.dense_size,256 ))
    layers.append( activation)
    if(BatchNorm == True):
      layers.append(nn.BatchNorm1d(256))
    layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(256,10 ))

    self.layers = nn.Sequential(*layers)
        
    self.loss = nn.CrossEntropyLoss()

  def forward(self,x):
    return self.layers(x)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(),lr= self.learning_rate)

  def training_step(self,batch):
    trainX,trainY = batch
    output = self(trainX)
    loss = self.loss(output,trainY)
    acc = (output.argmax(dim = 1) == trainY).float().mean()
    self.train_step_acc.append(acc)
    self.train_step_loss.append(loss)

    self.log('train_loss1', loss,on_epoch = True,on_step = False,prog_bar=True,metric_attribute="train_loss")
    self.log('train_acc1', acc,on_epoch = True,on_step = False,prog_bar=True,metric_attribute="train_acc")
    return loss

  def on_train_epoch_end(self):
    
    train_acc =  torch.stack(self.train_step_acc).mean()
    train_loss =  torch.stack(self.train_step_loss).mean()
    val_acc =  torch.stack(self.val_step_acc).mean()
    val_loss =  torch.stack(self.val_step_loss).mean()

    wandb.log({"train_loss":train_loss.item(),"train_acc":train_acc.item(),"val_loss":val_loss.item(),"val_acc":val_acc.item()})
    self.train_step_acc.clear() 
    self.train_step_loss.clear() 
    self.val_step_acc.clear() 
    self.val_step_loss.clear() 


  def validation_step(self, batch,batch_idx):
    trainX,trainY = batch
    output = self(trainX)
    loss = self.loss(output,trainY)
    acc = (output.argmax(dim = 1) == trainY).float().mean()
    self.val_step_acc.append(acc)
    self.val_step_loss.append(loss)
    self.log('val_loss1', loss,on_epoch = True,on_step = False,prog_bar=True,sync_dist=True)
    self.log('val_acc1', acc,on_epoch = True,on_step = False,prog_bar=True,sync_dist=True)
    return loss

  def test_step(self, batch,batch_idx):
    trainX,trainY = batch
    output = self(trainX)
    loss = self.loss(output,trainY)
    acc = (output.argmax(dim = 1) == trainY).float().mean()
    self.log('test_loss', loss,on_epoch = True,on_step = False,prog_bar=True)
    self.log('test_acc', acc,on_epoch = True,on_step = False,prog_bar=True)
    return loss

  def predict_step(self, batch,batch_idx,dataloader_idx=0):
    trainX,trainY = batch
    output = self(trainX)
    return output.argmax(dim = 1)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-wp','--wandb_project',required = False,metavar="",default ='Assignment-021',type=str,help = "Project name used to track experiments in Weights & Biases dashboard" )
parser.add_argument('-we','--wandb_entity',required = False,metavar="",default ='saisreeram',type=str,help = "Wandb Entity used to track experiments in the Weights & Biases dashboard.")
parser.add_argument('-dp','--dataset_path',required = False,metavar="",default ='inaturalist_12K',type=str,help = 'Give folder name only no /')
parser.add_argument('-e','--epochs',required = False,metavar="",default =1,type=int,help = "Number of epochs to train the model." )
parser.add_argument('-b','--batch_size',required = False,metavar="",default =16,type=int,help = "Batch size used to train the model.")

parser.add_argument('-lr','--learning_rate',required = False,metavar="",default =0.0001,type=float,help = "Learning rate used to optimize model parameters" )
parser.add_argument('-a','--activation',required = False,metavar="",default ='GELU',type=str,choices = ["ReLU", "GELU", "SiLU", "Mish"],help = 'choices: ["ReLU", "GELU", "SiLU", "Mish"]')

parser.add_argument('-bn','--batch_normalisation',required = False,metavar="",default =False,type=str,choices = [True,False],help = 'choices: [True,False]')
parser.add_argument('-da','--data_augmentation',required = False,metavar="",default =False,type=str,choices = [True,False],help = 'choices: [True,False]')
parser.add_argument('-do','--dropout',required = False,metavar="",default =0.4,type=float,help = 'Value of dropout ')
parser.add_argument('-iz','--input_size',required = False,metavar="",default =128,type=int,help = 'shape of the image while loading ')

parser.add_argument('-ks','--kernel_size',required = False,metavar="",default =3,type=int,help = "Conv kernel size" )
parser.add_argument('-pks','--pool_kernel_size',required = False,metavar="",default =2,type=int,help = "Max2dpool kernel size" )

parser.add_argument('-fz','--filter_size',required = False,metavar="",default =64,type=int,help = "filter size" )
parser.add_argument('-fl','--filter_organisation',required = False,metavar="",default ="same",type=str,choices = ["same", "half", "double"],help = "filter_organisation with same size or half or double" )

args = parser.parse_args()
wandb_entity = args.wandb_entity
wandb_project = args.wandb_project
dataset_path = args.dataset_path
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
activation = args.activation
batch_normalisation = args.batch_normalisation
data_augmentation = args.data_augmentation
dropout = args.dropout
input_size = args.input_size
kernel_size = args.kernel_size
pool_kernel_size = args.pool_kernel_size

filter_size = args.filter_size
filter_organisation = args.filter_organisation
if(filter_organisation == "same"):
    filters = [filter_size]*5
elif filter_organisation == "half":
    filters = [filter_size,filter_size//2,filter_size//4,filter_size//8,filter_size//16]
else :
    filters = [filter_size,filter_size*2,filter_size*4,filter_size*8,filter_size*16]


wandb.login(key = "8d6c17aa48af2229c26cbc16513ef266358c0b96")
wandb.init(project=wandb_project,entity = wandb_entity)



baseDir = dataset_path
trainDir = baseDir+"/train/"
testDir = baseDir+"/val/"
outputclasses=["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]


transform = transforms.Compose([
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

train_dataset = datasets.ImageFolder(root=baseDir+'/train',transform=transform)
test_dataset = datasets.ImageFolder(root=baseDir+'/val',transform=transform)
 
trainSize = int(0.8 * len(train_dataset))
valSize = len(train_dataset) - trainSize

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [trainSize, valSize])

transform_aug = transforms.Compose([
            transforms.Resize((input_size,input_size)),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

if data_augmentation == True :
    train_dataset.transform = transform_aug

train_dataset = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           )

val_dataset = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           )

test_dataset = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           )

activation_map = {"ReLU":nn.ReLU(), "GELU":nn.GELU(), "SiLU":nn.SiLU(), "Mish":nn.Mish()}

activation = activation_map[activation]
cnn = CNN(filters,activation,batch_normalisation,dropout,learning_rate,input_size,kernel_size,pool_kernel_size,2) 
trainer = pl.Trainer(max_epochs=epochs) 
trainer.fit(cnn,train_dataset,val_dataset)

wandb.finish()

