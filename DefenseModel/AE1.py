import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
from time import time
import os 

from load_data import *


class DenoiseAutoEncoder(nn.Module):
    def __init__(self):
        super(DenoiseAutoEncoder, self).__init__()
        # Encoder
        self.Encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),       # [, 64, 480, 480]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),      # [, 64, 480, 480]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # [, 64, 240, 240]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),      # [, 64, 240, 240]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1),     # [, 128, 240, 240]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1),    # [, 128, 240, 240]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1, 1),    # [, 256, 240, 240]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # [, 256, 120, 120]
            nn.BatchNorm2d(256)   
        )
        
        # decoder
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3 ,1, 1),      # [, 128, 120, 120]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, 1),   # [, 128, 240, 240]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),       # [, 64, 240, 240]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 1, 1),        # [, 32, 240, 240]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),        # [, 32, 240, 240]
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),     # [, 16, 480, 480]
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 3, 1, 1),         # [, 3, 480, 480]
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder, decoder

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default='/home/featurize/Proj/Datasets/train_PGD',
                help="path to the input folder")
ap.add_argument("-n", "--epochs", default=1000,
                help="epochs for train")
ap.add_argument("-b", "--batchsize", default=8,
                help="batch size for train")
args = vars(ap.parse_args())

use_gpu = torch.cuda.is_available()
imgSize = (480, 480)
Epochs = int(args["epochs"])
Batchsize = int(args["batchsize"])
modelFolder = './model'
if not os.path.isdir(modelFolder):
    os.mkdir(modelFolder)
storeName = modelFolder + 'AE.pth'

dst = labelTestDataLoader(args["input"].split(','), imgSize)
trainloader = DataLoader(dst, batch_size=Batchsize, num_workers=1)

DAEmodel = DenoiseAutoEncoder()
if use_gpu:
    DAEmodel = torch.nn.DataParallel(DAEmodel, device_ids=range(torch.cuda.device_count()))
    DAEmodel = DAEmodel.cuda()

LR = 0.0003
optimizer = optim.Adam(DAEmodel.parameters(), lr=LR)
loss_func = nn.MSELoss()


def val(model):
    dst = labelTestDataLoader(args["input"].split(','), imgSize)
    testloader = DataLoader(dst, batch_size=1, num_workers=1)
    loss_func = nn.MSELoss()
    val_loss = []
    for i, (XI, labels, ims) in enumerate(testloader):
        if (i>100) :
            break
        if use_gpu:
            x = Variable(XI.cuda(0))
        else:
            x = Variable(XI)
        
        _, output = DAEmodel(x)
        loss = loss_func(output, x)
        val_loss.append(loss.item())
        print(loss.item())
    return val_loss

bestmodel = -1
for epoch in range(Epochs):
    DAEmodel.train(True)
    train_loss= []
    start = time()

    for i, (XI, labels, ims) in enumerate(trainloader):
        if (i>100) :
            break
        if use_gpu:
            x = Variable(XI.cuda(0))
        else:
            x = Variable(XI)
        
        
        _, output = DAEmodel(x)
       
        loss = loss_func(output, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        train_loss.append(loss.item())
        print(loss.item())
        
    print('!')
    DAEmodel.eval()
    val_loss = val(DAEmodel)
    print(len(train_loss), len(val_loss))
    print ('%s %s %s %s\n' % (epoch, sum(train_loss)/len(train_loss), sum(val_loss)/len(val_loss), time()-start))
    torch.save(DAEmodel.state_dict(), storeName + str(epoch))
    if (bestmodel == -1 or bestmodel > sum(val_loss)/len(val_loss)):
        bestmodel = sum(val_loss)/len(val_loss)
        torch.save(DAEmodel.state_dict(), storeName)