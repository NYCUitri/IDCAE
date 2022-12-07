"""
 @file   keras_model.py
 @brief  Script for keras model definition
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
#########################################################################
import torch
import torch.nn as nn
import random
# from torch.autograd import Variable
# from torchvision.datasets import CIFAR10
#########################################################################
# pytorch model
#########################################################################
class Encoder(nn.Module):
    def __init__(self, paramF, paramM, classNum):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            #nn.Flatten(),

            # DenseBlock
            nn.Linear(paramF * paramM, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # DenseBlock 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # DenseBlock 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # DenseBlock 16
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(16, classNum)
        ) 
    
    def forward(self, x):
        latent = self.encoder(x)
        cls_output = self.classifier(latent)
        return latent, cls_output

class Decoder(nn.Module):
    def __init__(self, paramF, paramM, classNum):
        super(Decoder, self).__init__()

        self.condition_layer_Hr = nn.Sequential(
            nn.Linear(classNum, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Sigmoid(),

            # nn.Linear(16, 32),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),

            # nn.Linear(32, 32)
        )
        self.condition_layer_Hb = nn.Sequential(
            nn.Linear(classNum, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # nn.Sigmoid(),

            # nn.Linear(16, 32),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),

            # nn.Linear(32, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # DenseBlock 128
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, paramF * paramM)
        )

    def forward(self, latent, label, nm_label):
        m_Hr = self.condition_layer_Hr(label)
        m_Hb = self.condition_layer_Hb(label)
        # m_Hr, m_Hb = m_cond[:, :16], m_cond[:, 16:32]
        m_cond_latent = latent * m_Hr + m_Hb
        m_output = self.decoder(m_cond_latent)
        
        nm_Hr = self.condition_layer_Hr(nm_label)
        nm_Hb = self.condition_layer_Hb(nm_label)

        # nm_cond = self.condition_layer(nm_label)
        # nm_Hr, nm_Hb = nm_cond[:, :16], nm_cond[:, 16:32]
        nm_cond_latent = latent * nm_Hr + nm_Hb
        nm_output = self.decoder(nm_cond_latent)
        # print("M HR, Hb", m_Hr, m_Hb)
        # print("NM HR, Hb", nm_Hr, nm_Hb)
        return m_output, nm_output

    """ def predict(self, x, label):
        cond_latent = self.condition(label, latent)
        output = self.decoder(cond_latent)

        return output """

""" class FiLMLayer(nn.Module):
    def __init__(self, classNum):
        super(FiLMLayer, self).__init__()

        # Conditioning (Hr, Hb)
        # Hadamard Product
        self.condition_r = nn.Sequential(
            nn.Linear(classNum, 16),
            nn.Sigmoid(),
            nn.Linear(16, 16)
        )

        self.condition_b = nn.Sequential(
            nn.Linear(classNum, 16),
            nn.Sigmoid(), 
            nn.Linear(16, 16)
        )

    def forward(self, label, latent): 
        Hr = self.condition_r(label)
        Hb = self.condition_b(label)

        cond_latent = latent * Hr + Hb
        return cond_latent """

# Reshape Layer
class Reshape(nn.Module): 
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    
    def forward(self, x):
        # return x.view(self.shape)
        return x.view(-1, self.shape[0], self.shape[1])


##############################################################
# Loss Function
import numpy as np
import torch.nn as nn
##############################################################
class CustomLoss(nn.Module):
    def __init__(self, C, dim, batch_size):
        super(CustomLoss, self).__init__()
        
        self.const_vector = np.empty(shape=(batch_size, dim))
        self.const_vector.fill(C)
        self.const_vector = torch.Tensor(self.const_vector).to(device=torch.device('cuda'), non_blocking=True, dtype=torch.float32)

    def forward(self, m_output, nm_output, input):
        #print(nm_output[1])
        #nm_diff = torch.abs(nm_output - self.const_vector)
        #print(nm_diff[1])
        #nm_loss = torch.sum(nm_diff)
        #nm_loss = torch.sqrt(nm_dist) 
        #print(nm_loss)
        
        #m_diff = torch.abs(m_output - input)

        loss_fn = nn.L1Loss()
        #m_loss = torch.sum(m_diff)
        #m_loss = torch.sqrt(m_dist)
        #print(m_loss)
        m_loss = loss_fn(m_output, input)

        if input.shape[0] < self.const_vector.shape[0]:
            nm_loss = loss_fn(nm_output, self.const_vector[:input.shape[0]])
        else:
            nm_loss = loss_fn(nm_output, self.const_vector)

        return m_loss, nm_loss