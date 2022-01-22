print("started")
import sys
fsock = open('out.log', 'w', 1)
# sys.stdout = fsock
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torchvision.datasets import MNIST,CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tqdm import tqdm
from torchvision.utils import save_image
from torch.autograd import Variable
from utee import selector
import matplotlib.pyplot as plt
import tqdm
import math
import pickle
import logging

from reformer_model import *
from classifier_model import *
from adversaries import *
from utils import *

dataset_path = '~/datasets'
cuda = True
batch_size = 16
CALC_MU_SIGMA = False
epochs = 50

MODEL_REFINER_SAVE_PATH = 'denoise_batch.pt'
model_raw = ResNet18()
model_weights = torch.load('basic_training')['net']
model_weights_new = dict()
for key in model_raw.state_dict().keys() :
  model_weights_new[key] = model_weights['module.'+key]
model_raw.load_state_dict(model_weights_new)
DEVICE = torch.device("cuda" if cuda else "cpu")

transform = transforms.Compose([
        transforms.ToTensor(),
])

train_dataset_raw = CIFAR10(dataset_path, transform=transform, train=True, download=True)
test_dataset_raw  = CIFAR10(dataset_path, transform=transform, train=False, download=True)
print("Length of test dataset is ", len(test_dataset_raw))
print("Length of train dataset is ", len(train_dataset_raw))

kwargs = {} 
train_loader = DataLoader(dataset=train_dataset_raw, batch_size=batch_size, shuffle=True, **kwargs)
test_loader  = DataLoader(dataset=test_dataset_raw,  batch_size=batch_size, shuffle=True,  **kwargs)

# initialize the NN
model_reformer = REDNet30()
model_reformer.cuda()
print("Reformer model architechture :-") 
print(model_reformer)
model_reformer = torch.load(MODEL_REFINER_SAVE_PATH)
model_raw.cuda()
model_raw.eval()
model_reformer.eval()
np.random.seed(1)
torch.random.manual_seed(1)
    
if CALC_MU_SIGMA :
    np.random.seed(1)
    torch.random.manual_seed(1)
    model_raw.cuda()
    model_raw.eval()
    model_reformer.eval()
    types = ['normal']          
    data_loader = DataLoader(
        dataset=test_dataset_raw, batch_size=batch_size, shuffle=True)

    methods = ['simple','adap_dist','adap_custom']
    recons_errors = {}
    no_of_incorrects = {}
    no_of_corrects = {}

    for t in types :
      recons_errors[t] = []
      no_of_incorrects[t] = 0
      no_of_corrects[t] = 0

    epsilon = 0.0314
    epsilon_step = 0.00784
    random_pert_amt = 0.0314

    epsilon1 = 500/255
    epsilon_step1 = 60/255
    sigma = 0.5
    no_of_iters = 7
    no_of_iters1 = 12
    epsilon_random = 2/255
    device = 'cuda:0'
    rec_errors = {}

    for t in types :
      rec_errors[t] = []
      rec_errors["before_"+t] = []

    for iteration, (x, y) in tqdm.tqdm(enumerate(data_loader)):
        print("")
        x, y = x.to(device), y.to(device)
        if iteration > 100 :
            break
        for t in types :
            if t == 'normal' :
                x_type = x
            if t == 'random' :
                x_type = random(x,epsilon)
            if t == 'fgsm' :
                x_type = fgsm(model_raw,x,y,epsilon)
            if t == 'fgsm-L2' :
                x_type = fgsm_L2(model_raw,x,y,epsilon1)
            if t == 'R-fgsm' :
                x_type = R_fgsm(model_raw,x,y,epsilon,epsilon_random)
            if t == 'R-fgsm-L2' :
                x_type = R_fgsm_L2(model_raw,x,y,epsilon1,epsilon_random)
            if t == 'BIM' :
                x_type = BIM(model_raw,x,y,epsilon,epsilon_step,no_of_iters)
            if t == 'BIM-L2' :
                x_type = BIM_L2(model_raw,x,y,epsilon1,epsilon_step1,no_of_iters1)
            if t == 'CW' :
                rands = torch.randint(0,9,(x.shape[0],)).to(device)
                x_type = CW(model_raw,x,y,epsilon,3*epsilon_step,no_of_iters,2,(rands<y)*rands + (rands>=y)*(rands+1)) 
            if t == 'S-BIM' :
                rands = torch.randint(0,9,(x.shape[0],)).to(device)
                x_type = S_BIM(model_raw,models,x,(rands<y)*rands + (rands>=y)*(rands+1),epsilon,sigma,epsilon_step,no_of_iters)
            
            rec_errors["before_"+t].append(rec_error(x_type,model_reformer).cpu())
            x_type = refine_BIM(model_reformer,x_type,16,12/255,1/255) #model_reformer(x_type)
            rec_errors[t].append(rec_error(x_type,model_reformer).cpu())
            y_type = torch.argmax(model_raw(x_type),1)
            no_of_incorrects[t] += torch.sum((y_type!=y))
            no_of_corrects[t] += torch.sum((y_type==y))
            print("Corrects ",t," : ", torch.sum((y_type==y)))
            
    for t in types :
        print(t, ": ", 100*float(no_of_corrects[t])/float(no_of_corrects[t]+no_of_incorrects[t]))
        rec_errors[t] = torch.cat(rec_errors[t],dim=0)
        rec_errors['before_'+t] = torch.cat(rec_errors['before_'+t],dim=0)

    rec_errors['before_normal'] = (rec_errors['before_normal']>20)*3 + (rec_errors['before_normal']<=20)*rec_errors['before_normal']
    mu = torch.mean(rec_errors['before_normal'])
    sigma = math.sqrt(torch.var(rec_errors['before_normal']-mu)) 
    dist = {}
    dist['mu'] = mu
    dist['sigma'] = sigma
    pickle.dump(dist,open('mu_sigma.pkl','wb'))

dist = pickle.load(open('mu_sigma.pkl','rb'))
mu,sigma = dist['mu'], dist['sigma']
print("mu and sigma : ", mu,sigma)

types = ['normal','random','fgsm','R-fgsm','BIM']          
data_loader = DataLoader(
    dataset=test_dataset_raw, batch_size=batch_size, shuffle=True)

methods = ['simple','adap_dist','adap_custom','adap_prob_individual','kl-divergence']
method = 'adap_prob_individual'
random_pert_amt = 0.0314
recons_errors = {}
no_of_incorrects = {}
no_of_corrects = {}
for t in types :
  recons_errors[t] = []
  no_of_incorrects[t] = 0
  no_of_corrects[t] = 0
# model.eval()
epsilon = 0.0314
epsilon_step = 0.00784

epsilon1 = 500/255
epsilon_step1 = 60/255
sigma = 0.5
no_of_iters = 7
no_of_iters1 = 12
epsilon_random = 2/255
device = 'cuda:0'
rec_errors = {}

for t in types :
  rec_errors[t] = []
  rec_errors["before_"+t] = []
  rec_errors["cae_"+t] = []

for iteration, (x, y) in tqdm.tqdm(enumerate(data_loader)):
    print("")
    x, y = x.to(device), y.to(device)
    if iteration > 100 :
      break
    for t in types :
        if t == 'normal' :
          x_type = x
        if t == 'random' :
          x_type = random(x,epsilon)
        if t == 'fgsm' :
          x_type = fgsm(model_raw,x,y,epsilon)
        if t == 'fgsm-L2' :
          x_type = fgsm_L2(model_raw,x,y,epsilon1)
        if t == 'R-fgsm' :
          x_type = R_fgsm(model_raw,x,y,epsilon,epsilon_random)
        if t == 'R-fgsm-L2' :
          x_type = R_fgsm_L2(model_raw,x,y,epsilon1,epsilon_random)
        if t == 'BIM' :
          x_type = BIM(model_raw,x,y,epsilon,epsilon_step,no_of_iters)
        if t == 'BIM_random' :
          x_type = random(BIM(model_raw,x,y,epsilon,epsilon_step,no_of_iters),random_pert_amt/2)  
        if t == 'BIM-L2' :
          x_type = BIM_L2(model_raw,x,y,epsilon1,epsilon_step1,no_of_iters1)
        if t == 'CW' :
          rands = torch.randint(0,9,(x.shape[0],)).to(device)
          x_type = CW(model_raw,x,y,epsilon,3*epsilon_step,no_of_iters,2,(rands<y)*rands + (rands>=y)*(rands+1)) 
        if t == 'S-BIM' :
          rands = torch.randint(0,9,(x.shape[0],)).to(device)
          x_type = S_BIM(model_raw,models,x,(rands<y)*rands + (rands>=y)*(rands+1),epsilon,sigma,epsilon_step,no_of_iters)
        
        rec_errors["before_"+t].append(rec_error(x_type,model_reformer).cpu())
        
        if method == 'adap_prob_individual' :
          x_type = refine_BIM_prob_individual(model_reformer,x_type,13,0.7/255,2/255,mu,sigma) #model_reformer(x_type)
        elif method == 'simple':
          x_type = refine_BIM(model_reformer,x_type,12,12/255,1/255) #model_reformer(x_type)
        elif method == 'adap_dist' :
          x_type = refine_BIM_biased_cont(model_reformer,x_type,16,12/255,1/255,mu,sigma) #model_reformer(x_type)
        elif method == 'kl-divergence' :
          x_type = KL_divergence(model_reformer,x_type,16,12/255,2/255,mu,sigma) #model_reformer(x_type)

        rec_errors[t].append(rec_error(x_type,model_reformer).cpu())
        y_type = torch.argmax(model_raw(x_type),1)
        # if t == 'BIM' :
        #   for i in range(16) :
        #     if y_type[i] != y[i] :
        #       rec_errors['cae_BIM'].append(get_recon_cae(x_type[i:(i+1)],y_type[i:(i+1)]))
        # if t == 'normal' :
        #   rec_errors['cae_normal'].append(get_recon_cae(x_type,y_type))
        no_of_incorrects[t] += torch.sum((y_type!=y))
        no_of_corrects[t] += torch.sum((y_type==y))
        print("Corrects",t," : ", torch.sum((y_type==y)))
        # stdout.flush()
        # logger.debug("Corrects "+t+" : "+str(float(torch.sum((y_type==y)))))
        
for t in types :
    print(t, ": ", float(no_of_corrects[t])/float(no_of_corrects[t]+no_of_incorrects[t]))
    rec_errors[t] = torch.cat(rec_errors[t],dim=0)
    rec_errors['before_'+t] = torch.cat(rec_errors['before_'+t],dim=0)

rec_errors['cae_BIM'] = torch.cat(rec_errors['cae_BIM'],dim=0)
rec_errors['cae_normal'] = torch.cat(rec_errors['cae_normal'],dim=0)

fsock.close()
