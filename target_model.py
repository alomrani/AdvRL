import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import softmax
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.tensor(x).float()).cpu().numpy()
    
    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits  # difference between the correct class and all other classes
            diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        else:
            raise ValueError('Wrong loss.')
        return loss.flatten()




class CifarNet(nn.Module):
    """Basic CNN architecture."""

    def __init__(self, in_channels=1):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 8, 1)
        self.conv2 = nn.Conv2d(64, 128, 6, 2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2)
        self.fc = nn.Linear(128 * 3 * 3, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 3 * 3)
        x = self.fc(x)
        return x

class CifarNet2(nn.Module):
    def __init__(self):
        super(CifarNet2, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3,96,3), # 96*30*30
            nn.GroupNorm(32, 96),
            nn.ELU(),
            
            nn.Dropout2d(0.2),
            
            nn.Conv2d(96, 96, 3), # 96*28*28
            nn.GroupNorm(32, 96),
            nn.ELU(),
            
            nn.Conv2d(96, 96, 3), # 96*26*26
            nn.GroupNorm(32, 96),
            nn.ELU(),
            
            nn.Dropout2d(0.5),
            
            nn.Conv2d(96, 192, 3), # 192*24*24
            nn.GroupNorm(32, 192),
            nn.ELU(),
            
            nn.Conv2d(192, 192, 3), # 192*22*22
            nn.GroupNorm(32, 192),
            nn.ELU(),
           
            nn.Dropout2d(0.5),
            
            nn.Conv2d(192, 256, 3), # 256*20*20
            nn.GroupNorm(32, 256),
            nn.ELU(),
            
            nn.Conv2d(256, 256, 1), # 256*20*20
            nn.GroupNorm(32, 256),
            nn.ELU(),
            
            nn.Conv2d(256, 10, 1), # 10*20*20
            nn.AvgPool2d(20) # 10*1*1
        )

    def forward(self,x):
        out = self.conv_layer(x)
        out = out.view(-1,10)

        return out

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.tensor(x).float()).cpu().numpy()
    
    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits  # difference between the correct class and all other classes
            diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        else:
            raise ValueError('Wrong loss.')
        return loss.flatten()