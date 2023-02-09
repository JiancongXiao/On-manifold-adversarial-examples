import copy
import numpy as np
from collections import Iterable
# from scipy.stats import truncnorm
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import time
from adversarialbox.utils import to_var

# --- White-box attacks ---
def idx2onehot(idx, n):

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n).cuda()
    onehot.scatter_(1, idx, 1)

    return onehot

class labelDataset(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.num_data = len(dataset)
        self.label = label
        self.loader = DataLoader(self.dataset, shuffle=False, batch_size=self.num_data, num_workers=4)
        for i,(x,y) in enumerate(self.loader):
            self.data = x[y==self.label]
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        image = self.data[idx]
        return image, self.label

class VAEFGSMAttack(object):
    def __init__(self, encoder, decoder, d2c, epsilon=1):
        """
        One step fast gradient sign method
        """
        self.encoder = encoder
        self.decoder = decoder
        self.d2c = d2c
        self.loss_fn = nn.CrossEntropyLoss()
        self.epsilon = epsilon
        
    def perturb(self, x, y):
        means, log_vars = self.encoder(x,y)

        z_var = to_var(means, requires_grad=True)
        y_var = to_var(y)

        scores = self.d2c(z_var, y_var)
        loss = self.loss_fn(scores, y_var)
        loss.backward()
        grad_sign = z_var.grad.data.sign()
        z = means + self.epsilon * grad_sign
        x_adv = self.decoder(z,y)
        return x_adv

class VAEPGDAttack(object):
    def __init__(self, encoder, decoder, d2c, epsilon=1, k=40, a=0.01):
        """
        One step fast gradient sign method
        """
        self.encoder = encoder
        self.decoder = decoder
        self.d2c = d2c
        self.loss_fn = nn.CrossEntropyLoss()
        self.epsilon = epsilon
        self.k = k
        self.a = a
        
    def perturb(self, x, y):

        means, log_vars = self.encoder(x,y)
        z = means
        for i in range(self.k):
            z_var = to_var(z, requires_grad=True)
            y_var = to_var(y)

            scores = self.d2c(z_var, y_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad_sign = z_var.grad.data.sign()

            z_temp = z + self.a * grad_sign
            Delta_z = z_temp-means
            z = z + torch.clamp(Delta_z,self.epsilon, self.epsilon)
        x_adv = self.decoder(z,y)
        return x_adv
    
class CifarVAEFGSMAttack(object):
    def __init__(self, encoder, decoder, d2c, epsilon=1):
        """
        One step fast gradient sign method
        """
        self.encoder = encoder
        self.decoder = decoder
        self.d2c = d2c
        self.loss_fn = nn.CrossEntropyLoss()
        self.epsilon = epsilon
        
    def perturb(self, x, y):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        z_var = to_var(mu, requires_grad=True)
        y_var = to_var(y)
        batch_size = y.size(0)
        y1 = idx2onehot(y, n=10)
        y1 = y1.view(batch_size,10,1,1)
        y2 = torch.cat([y1,y1],dim = 2)
        y3 = torch.cat([y2,y2],dim = 3)
        z2 = torch.cat([z_var,y3],dim = 1)

        scores = self.d2c(z2)
        loss = self.loss_fn(scores, y_var)
        loss.backward()
        grad_sign = z_var.grad.data.sign()

        z = mu + self.epsilon * grad_sign
        x_adv = self.decoder(torch.cat([z,y3],dim = 1))
        return x_adv

class FGSMAttack(object):
    def __init__(self, model=None, epsilon = 8/255):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, epsilon=8/255):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilon is not None:
            self.epsilon = epsilon
        size = X_nat.size()
        X = np.copy(X_nat)

        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        y_var = to_var(torch.LongTensor(y))

        scores = self.model(X_var)
        loss = self.loss_fn(scores, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += self.epsilon * grad_sign
        if size[1] == 3:
            X = np.clip(X, -1, 1)
        else:
            X = np.clip(X, 0, 1)

        return X


class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.2, k=40, a=0.01, 
        random_start=True):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, epsilon = 0.25):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        X_nat = X_nat.cpu().numpy()
        if self.rand:
            X = X_nat + np.random.uniform(-self.epsilon, self.epsilon,
                X_nat.shape).astype('float32')
        else:
            X = np.copy(X_nat)

        for i in range(self.k):
            X_var = to_var(torch.from_numpy(X), requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()

            X += self.a * np.sign(grad)

            X = np.clip(X, X_nat - self.epsilon, X_nat + self.epsilon)
            X = np.clip(X, 0, 1) # ensure valid pixel range

        return X

class LtwoPGDattack(object):
    def __init__(self, model=None, epsilon=2, k=40, a=0.5, 
        random_start=True):

        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, epsilon = 0.25):
        
        X = X_nat
        size = X_nat.size()
        batch_size = size[0]
        X_nat =  X_nat.view(batch_size,-1)

        if epsilon is not None:
            self.epsilon = epsilon

        for i in range(self.k):
            X_var = to_var(X, requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad.data
            grad = grad.view(batch_size,-1)
            norm_grad = torch.norm(grad, dim = 1).view(-1,1)
            grad = torch.div(grad,norm_grad)
            X = X.view(batch_size, -1)
            X_temp = X + self.a * grad
            dis = X_temp - X_nat
            eta = torch.norm(dis, dim = 1)
            eta = eta.view(-1,1)
            eta1 = torch.clamp(eta,0, self.epsilon)

            X = X_nat + eta1 * torch.div(dis,eta)
            X = X.view(size)
            if size[1] == 1:
                X = torch.clamp(X, 0, 1) # ensure valid pixel range
            else:
                X = torch.clamp(X, -1, 1)
        X_adv = X.numpy()
        return X_adv

class GaussAttack(object):
    def __init__(self, epsilon = 2):
        self.epsilon = epsilon

    def perturb(self, x, y, epsilon = 2):

        if epsilon is not None:
            self.epsilon = epsilon

        size = x.size()
        batch_size = size[0]
        vecx = x.view(batch_size,-1)
        direc = torch.randn(vecx.size())
        norm_direc = torch.norm(direc,dim = 1)
        norm_direc = norm_direc.view(-1,1)

        x_adv = vecx + self.epsilon * torch.div(direc, norm_direc)
        if size[1] == 3:
            x_adv = torch.clamp(x_adv,-1,1)
        else:
            x_adv = torch.clamp(x_adv,0,1)
        x_adv = x_adv.view(size)
        x_adv = np.copy(x_adv)

        return x_adv


class PCAAttack(object):
    def __init__(self, epsilon = 2, Thres = 0.5):
        self.epsilon = epsilon
        self.Thres = Thres
    def perturb(self, x, y, epsilon = 2):

        if epsilon is not None:
            epsilon = self.epsilon

        size = x.size()
        batch_size = size[0]
        vecx = x.view(batch_size,-1)
        No_feature = vecx.size()[1]
        vecxmean = torch.mean(vecx,dim = 0)
        K = vecx - vecxmean
        K = K.numpy()
        U,S,Vt = np.linalg.svd(K)
        # for i in range(No_feature):
        #     if S[i]<=self.Thres:
        #         break

        direc = Vt[-200:-1,:]
        direc = torch.from_numpy(direc)
        direc = torch.mean(direc, dim = 0)
        norm_direc = torch.norm(direc)

        x_adv = vecx + self.epsilon * torch.div(direc, norm_direc)
        if size[1] == 3:
            x_adv = torch.clamp(x_adv,-1,1)
        else:
            x_adv = torch.clamp(x_adv,0,1)
        x_adv = x_adv.view(size)
        x_adv = np.copy(x_adv)

        return x_adv