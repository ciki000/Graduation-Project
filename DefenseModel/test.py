import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from  torchvision import utils as vutils
from PIL import Image
import argparse
import os

from load_data import *
from roi_pooling import roi_pooling_ims

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default='/home/featurize/zwk/Datasets/train_PGD',
                help="path to the input folder")
ap.add_argument("-m", "--model", default='/home/featurize/zwk/Pretrain/fh02.pth',
                help="path to the model file")
args = vars(ap.parse_args())

use_gpu = torch.cuda.is_available()
print(use_gpu)
numClasses = 7
numPoints = 4
imgSize = (480, 480)
batchSize = 1
resume_file = str(args["model"])


provNum, alphaNum, adNum = 38, 25, 35
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

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

class wR2(nn.Module):
    def __init__(self, num_classes=1000):
        super(wR2, self).__init__()
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8,
            hidden9,
            hidden10
        )
        self.classifier = nn.Sequential(
            nn.Linear(23232, 100),
            nn.Linear(100, 100),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x1 = self.features(x)
        x11 = x1.view(x1.size(0), -1)
        x = self.classifier(x11)
        return x


class fh02(nn.Module):
    def __init__(self, num_points, num_classes, wrPath=None):
        super(fh02, self).__init__()
        self.load_wR2(wrPath)
        self.classifier1 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, provNum),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, alphaNum),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.classifier5 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.classifier6 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.classifier7 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )

    def load_wR2(self, path):
        self.wR2 = wR2(numPoints)
        self.wR2 = torch.nn.DataParallel(self.wR2, device_ids=range(torch.cuda.device_count()))
        if not path is None:
            self.wR2.load_state_dict(torch.load(path))

    def forward(self, x):
        x0 = self.wR2.module.features[0](x)
        _x1 = self.wR2.module.features[1](x0)
        x2 = self.wR2.module.features[2](_x1)
        _x3 = self.wR2.module.features[3](x2)
        x4 = self.wR2.module.features[4](_x3)
        _x5 = self.wR2.module.features[5](x4)

        x6 = self.wR2.module.features[6](_x5)
        x7 = self.wR2.module.features[7](x6)
        x8 = self.wR2.module.features[8](x7)
        x9 = self.wR2.module.features[9](x8)
        x9 = x9.view(x9.size(0), -1)
        boxLoc = self.wR2.module.classifier(x9)

        h1, w1 = _x1.data.size()[2], _x1.data.size()[3]
        p1 = Variable(torch.FloatTensor([[w1,0,0,0],[0,h1,0,0],[0,0,w1,0],[0,0,0,h1]]).cuda(), requires_grad=False)
        h2, w2 = _x3.data.size()[2], _x3.data.size()[3]
        p2 = Variable(torch.FloatTensor([[w2,0,0,0],[0,h2,0,0],[0,0,w2,0],[0,0,0,h2]]).cuda(), requires_grad=False)
        h3, w3 = _x5.data.size()[2], _x5.data.size()[3]
        p3 = Variable(torch.FloatTensor([[w3,0,0,0],[0,h3,0,0],[0,0,w3,0],[0,0,0,h3]]).cuda(), requires_grad=False)

        # x, y, w, h --> x1, y1, x2, y2
        assert boxLoc.data.size()[1] == 4
        postfix = Variable(torch.FloatTensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]).cuda(), requires_grad=False)
        boxNew = boxLoc.mm(postfix).clamp(min=0, max=1)

        roi1 = roi_pooling_ims(_x1, boxNew.mm(p1), size=(16, 8))
        roi2 = roi_pooling_ims(_x3, boxNew.mm(p2), size=(16, 8))
        roi3 = roi_pooling_ims(_x5, boxNew.mm(p3), size=(16, 8))
        rois = torch.cat((roi1, roi2, roi3), 1)

        _rois = rois.view(rois.size(0), -1)

        y0 = self.classifier1(_rois)
        y1 = self.classifier2(_rois)
        y2 = self.classifier3(_rois)
        y3 = self.classifier4(_rois)
        y4 = self.classifier5(_rois)
        y5 = self.classifier6(_rois)
        y6 = self.classifier7(_rois)
        return boxLoc, [y0, y1, y2, y3, y4, y5, y6]

def save_image(t, name):
    dir = 'box'
    img = t.cpu().squeeze().numpy()
    img = np.transpose(img * 255, (1,2,0))
    if not os.path.exists(dir):
      os.makedirs(dir)
    cv2.imwrite('./'+dir+'/test'+name+'.jpg', img)
    
def FGSM(model, x, labels, id, eps=0.01, clip_min=0.0, clip_max=1.0):
    x_new = x #+ torch.Tensor(np.random.uniform(-eps, eps, x.shape)).type_as(x).cuda()
    x_new = Variable(x_new, requires_grad=True)
    loss_func = nn.CrossEntropyLoss()
    fps_pred, y_pred = model(x_new)
    #print(x.shape)
    loss = 0
    for j in range(7):
        l = torch.tensor([labels[j]]).cuda()
        loss += loss_func(y_pred[j], l)
    #print(loss)

    model.zero_grad()
    loss.backward()
    grad = x_new.grad.cpu().detach().numpy()
    #print(grad.shape)
    #np.savetxt('grad', grad)
    grad = np.sign(grad)
    pertubation = grad * eps
    adv_x = x.cpu().detach().numpy() + pertubation
    adv_x = np.clip(adv_x, clip_min, clip_max)

    x_adv = torch.from_numpy(adv_x).cuda()
    adv_fps_pred, adv_y_pred = model(x_adv)
    ##save_image(x, str(id))
    ##save_image(x_adv, str(id)+'_adv')
    adv_loss = 0
    for j in range(7):

        l = torch.tensor([labels[j]]).cuda()
        adv_loss += loss_func(adv_y_pred[j], l)
    #print(adv_loss)
    adv_outputY = [el.data.cpu().numpy().tolist() for el in adv_y_pred]
    adv_labelPred = [t[0].index(max(t[0])) for t in adv_outputY]
    #print(provinces[labels[0]], alphabets[labels[1]], [ads[labels[i]] for i in range(2, 7)])
    #print(provinces[adv_labelPred[0]], alphabets[adv_labelPred[1]], [ads[adv_labelPred[i]] for i in range(2, 7)])
    ##print()
    return x_adv

def PGD(model, x, labels, id, iter=20, eps=0.001, clip_min=0.0, clip_max=1.0):
    save_image(x, str(id))
    
    for i in range(iter):
        x = FGSM(model, x, labels, id, eps=eps)

    save_image(x, str(id)+'_adv')
    fps_pred, y_pred = model_conv(x)
    outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
    labelPred = [t[0].index(max(t[0])) for t in outputY]
    print(provinces[labelPred[0]], alphabets[labelPred[1]], [ads[labelPred[i]] for i in range(2, 7)])
    print()

def printlicence(label):
    print(provinces[label[0]], alphabets[label[1]], [ads[label[i]] for i in range(2, 7)])

def showbox(x, lu, rd, name):
    for i in range(lu[1], rd[1]):
        x[0][0][i][lu[0]] = 0
        x[0][1][i][lu[0]] = 0
        x[0][2][i][lu[0]] = 1
        
        x[0][0][i][rd[0]] = 0
        x[0][1][i][rd[0]] = 0
        x[0][2][i][rd[0]] = 1
        
    for i in range(lu[0], rd[0]):
        x[0][0][lu[1]][i] = 0
        x[0][1][lu[1]][i] = 0
        x[0][2][lu[1]][i] = 1
        
        x[0][0][rd[1]][i] = 0
        x[0][1][rd[1]][i] = 0
        x[0][2][rd[1]][i] = 1
    save_image(x, 'box'+name)
    
def close(x, pos1, pos2):
    if (abs(x[0][0][pos1[0]][pos1[1]]-x[0][0][pos2[0]][pos2[1]]) * 255.0 > 20.0):
        return False
    if (abs(x[0][1][pos1[0]][pos1[1]]-x[0][1][pos2[0]][pos2[1]]) * 255.0 > 20.0):
        return False
    if (abs(x[0][2][pos1[0]][pos1[1]]-x[0][2][pos2[0]][pos2[1]]) * 255.0 > 20.0):
        return False
    return True
def ave(x, lu, rd):
    y = x.clone()
    #print(lu[1], rd[1])
    #print(lu[0], rd[0])
    for i in range(lu[1], rd[1]):
        for j in range(lu[0], rd[0]):
            cnt = 1.0
            if (close(x, (i, j), (i-1, j))):
                cnt += 1
                y[0][0][i][j] += x[0][0][i-1][j]
                y[0][1][i][j] += x[0][1][i-1][j]
                y[0][2][i][j] += x[0][2][i-1][j]
            if (close(x, (i, j), (i+1, j))):
                cnt += 1
                y[0][0][i][j] += x[0][0][i+1][j]
                y[0][1][i][j] += x[0][1][i+1][j]
                y[0][2][i][j] += x[0][2][i+1][j]
            if (close(x, (i, j), (i, j-1))):
                cnt += 1
                y[0][0][i][j] += x[0][0][i][j-1]
                y[0][1][i][j] += x[0][1][i][j-1]
                y[0][2][i][j] += x[0][2][i][j-1]
            if (close(x, (i, j), (i, j+1))):
                cnt += 1
                y[0][0][i][j] += x[0][0][i][j+1]
                y[0][1][i][j] += x[0][1][i][j+1]
                y[0][2][i][j] += x[0][2][i][j+1]
            y[0][0][i][j] /= cnt
            y[0][1][i][j] /= cnt
            y[0][2][i][j] /= cnt
    return y
                      
        
model_conv = fh02(numPoints, numClasses)
model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
model_conv.load_state_dict(torch.load(resume_file))
model_conv = model_conv.cuda()
model_conv.eval()

resume_file = './AE.pth'
DAEmodel = DenoiseAutoEncoder()
DAEmodel = torch.nn.DataParallel(DAEmodel, device_ids=range(torch.cuda.device_count()))
DAEmodel.load_state_dict(torch.load(resume_file))
DAEmodel = DAEmodel.cuda()
DAEmodel.eval()

dst = labelTestDataLoader(args["input"].split(','), imgSize)
trainloader = DataLoader(dst, batch_size=1, num_workers=1)

error = 0
correct = 0
for i, (XI, labels, ims) in enumerate(trainloader):
    
    YI = [int(num) for num in labels[0].split('_')]
    if use_gpu:
        x = Variable(XI.cuda(0))
    else:
        x = Variable(XI)
    '''
    eps = 0.05
    x = x + torch.Tensor(np.random.uniform(-eps, eps, x.shape)).type_as(x).cuda()
    '''
    _, output = DAEmodel(x)
    
    fps_pred, y_pred = model_conv(x)
    [cx, cy, w, h] = fps_pred.data.cpu().numpy()[0].tolist()

    left_up = [int((cx - w/2)*imgSize[1]), int((cy - h/2)*imgSize[0])]
    right_down = [int((cx + w/2)*imgSize[1]), int((cy + h/2)*imgSize[0])]
    
    #print(x.size())
    #showbox(x, left_up, right_down, str(i))
    
    output_p = ave(x, left_up, right_down)
    '''
    save_image(x, str(i))
    save_image(output.detach(), str(i)+'_1')
    save_image(output_p.detach(), str(i)+'_2')
    '''
    fps_pred, y_pred = model_conv(output)
    outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
    labelPred = [t[0].index(max(t[0])) for t in outputY]
    #print(YI[1:])
    #break
    #if (YI == labelPred):
    if (YI[1:] == labelPred[1:]):
        correct += 1
    else:
        error +=1
    if (i % 10 == 0):
        print('T:',correct,'F:',error)
    if (i == 99):
        break
print('T:',correct,'F:',error)
    