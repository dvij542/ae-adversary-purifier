import math
import torch
from adversaries import *
import torch.nn as nn
import torch.nn.functional as F

def rec_error(x,model_reformer) :
  x_recon = clip(model_reformer(x),0.,1.)
  return torch.sum((x_recon.detach().cpu().view(-1,32*32*3) - x.detach().cpu().view(-1,32*32*3))**2,axis=1)

def get_same_index(target, label):
    label_indices = []
    for i in range(len(target)):
        if target[i] == label:
            label_indices.append(i)
    return label_indices

def refine_BIM(model,X,no_of_steps,epsilon,epsilon_step) :
  Xn = X.clone()
  delta = torch.zeros_like(X, requires_grad=True)
  optimizer = torch.optim.Adam([delta],lr=epsilon_step)
  for i in range(no_of_steps) :
    model.eval()
    # print(len(model(Xn+delta)))
    X_recon = model(Xn+delta)
    optimizer.zero_grad()
    loss = torch.sum((Xn+delta-X_recon)**2)
    loss.backward()
    optimizer.step()
    # delta = clip(delta,-0.1,0.1)
    # print(torch.mean(torch.abs(delta.grad.detach())))
    # Xn = Xn.clone() + epsilon_step * delta.grad.detach()
  # diff = Xn - X
  return Xn + delta 

def KL_divergence(model,X,no_of_steps,epsilon,epsilon_step,mu,sigma) :
  Xn = X.clone()
  delta = torch.zeros_like(X, requires_grad=True)
  model.eval()
  X_recon = model(Xn+delta)
  recon_dist = torch.sum((Xn+delta-X_recon)**2)/Xn.shape[0]
  prob = torch.exp(-(recon_dist-mu)**2/sigma**2) 
  # mu, sigma = torch.tensor(mu), torch.tensor(sigma)
  optimizer = torch.optim.Adam([delta],lr=epsilon_step)
  optimizer.zero_grad()
  for i in range(no_of_steps) :
    model.eval()
    X_recon = model(Xn+delta)
    optimizer.zero_grad()
    recon_dist = torch.sum((Xn+delta-X_recon)**2,axis=[1,2,3])
    mu_d = torch.mean(recon_dist)
    sigma_d = torch.var(recon_dist)**(1/2)
    # loss = torch.log(sigma) - torch.log(sigma_d) + ((sigma_d)**2 + (mu-mu_d)**2)/(2*sigma**2)
    loss = ((mu-mu_d)**2)/(sigma**2)
    loss.backward()
    optimizer.step()

  return Xn + delta 

def refine_BIM_biased(model,X,no_of_steps,epsilon,epsilon_step,mu,sigma) :
  Xn = X.clone()
  delta = torch.zeros_like(X, requires_grad=True)
  slope = 1/(2*sigma)
  model.eval()
  X_recon = model(Xn+delta)
  recon_dist = torch.sum((Xn+delta-X_recon)**2)/Xn.shape[0]
  print(torch.sigmoid((recon_dist-mu)/sigma))
  
  optimizer = torch.optim.Adam([delta],lr=epsilon_step*torch.sigmoid((recon_dist-mu)/sigma))
  optimizer.zero_grad()
  for i in range(no_of_steps) :
    model.eval()
    # print(len(model(Xn+delta)))
    X_recon = model(Xn+delta)
    optimizer.zero_grad()
    recon_dist = torch.sum((Xn+delta-X_recon)**2)
    # loss = torch.sigmoid(-(mu-sigma)*slope+slope*recon_dist)*recon_dist
    loss = recon_dist
    loss.backward()
    optimizer.step()
    # delta = clip(delta,-0.1,0.1)
    # print(torch.mean(torch.abs(delta.grad.detach())))
    # Xn = Xn.clone() + epsilon_step * delta.grad.detach()
  # diff = Xn - X
  return Xn + delta 

def refine_BIM_prob(model,X,no_of_steps,epsilon,epsilon_step,mu,sigma) :
  Xn = X.clone()
  delta = torch.zeros_like(X, requires_grad=True)
  slope = 1/(2*sigma)
  model.eval()
  X_recon = model(Xn+delta)
  recon_dist = torch.sum((Xn+delta-X_recon)**2)/Xn.shape[0]
  prob = torch.exp(-(recon_dist-mu)**2/sigma**2) 
  # print(prob)
  
  optimizer = torch.optim.Adam([delta],lr=epsilon_step)
  optimizer.zero_grad()
  for i in range(no_of_steps) :
    model.eval()
    # print(len(model(Xn+delta)))
    X_recon = model(Xn+delta)
    optimizer.zero_grad()
    recon_dist = torch.sum((Xn+delta-X_recon)**2)/Xn.shape[0]
    # loss = torch.sigmoid(-(mu-sigma)*slope+slope*recon_dist)*recon_dist
    prob = -(recon_dist-mu)**2/sigma**2 
    loss = -torch.sum(prob)
    loss.backward()
    optimizer.step()
    # delta = clip(delta,-0.1,0.1)
    # print(torch.mean(torch.abs(delta.grad.detach())))
    # Xn = Xn.clone() + epsilon_step * delta.grad.detach()
  # diff = Xn - X
  return Xn + delta 

def refine_BIM_prob_individual(model,X,no_of_steps,epsilon,epsilon_step,mu,sigma) :
  Xn = X.clone()
  delta = torch.zeros_like(X, requires_grad=True)
  model.eval()
  X_recon = model(Xn+delta)
  recon_dist = torch.sum((Xn+delta-X_recon)**2)/Xn.shape[0]
  prob = torch.exp(-(recon_dist-mu)**2/sigma**2) 
  
  optimizer = torch.optim.Adam([delta],lr=epsilon_step)
  optimizer.zero_grad()
  # model.eval()
  for i in range(no_of_steps) :
    X_recon = model(Xn+delta)
    optimizer.zero_grad()
    recon_dist = torch.sum((Xn+delta-X_recon)**2,axis=[1,2,3],keepdim=True)
    Xn = random(Xn,epsilon*(clip(((recon_dist.detach()-mu)/(2*sigma))**2,0,4)))
    prob = -((recon_dist>mu)*(recon_dist-mu))**2/sigma**2 
    loss = -torch.sum(prob)
    loss.backward()
    optimizer.step()

  return Xn + delta 

def refine_BIM_biased_cont(model,X,no_of_steps,epsilon,epsilon_step,mu,sigma) :
  Xn = X.clone()
  delta = torch.zeros_like(X, requires_grad=True)
  slope = 1/(2*sigma)
  model.eval()
  X_recon = model(Xn+delta)
  recon_dist = torch.sum((Xn+delta-X_recon)**2)/Xn.shape[0]
  print(torch.sigmoid((recon_dist-mu)/sigma))
  
  optimizer = torch.optim.Adam([delta],lr=epsilon_step)
  optimizer.zero_grad()
  for i in range(no_of_steps) :
    model.eval()
    # print(len(model(Xn+delta)))
    X_recon = model(Xn+delta)
    optimizer.zero_grad()
    recon_dist = torch.sum((Xn+delta-X_recon)**2)
    # loss = torch.sigmoid(-(mu-sigma)*slope+slope*recon_dist)*recon_dist
    loss = recon_dist*torch.sigmoid((recon_dist-mu)/sigma)
    loss.backward()
    optimizer.step()
    # delta = clip(delta,-0.1,0.1)
    # print(torch.mean(torch.abs(delta.grad.detach())))
    # Xn = Xn.clone() + epsilon_step * delta.grad.detach()
  # diff = Xn - X
  return Xn + delta 

def refine_BIM_L2(model,X,no_of_steps,epsilon,epsilon_step) :
  Xn = X.clone()
  for i in range(no_of_steps) :
    delta = torch.zeros_like(X, requires_grad=True)
    X_recon,_,_,_ = model(Xn+delta)
    loss = -torch.mean((Xn+delta-X_recon)**2)
    loss.backward()
    Xn = Xn.clone() + epsilon_step * delta.grad.detach()/torch.norm(torch.norm((delta.grad.detach()),dim=2,keepdim=True),dim=3,keepdim=True)
  diff = Xn - X
  factor = clip(torch.norm(torch.norm((diff.detach()),dim=2,keepdim=True),\
              dim=3,keepdim=True),0,epsilon)\
              / torch.norm(torch.norm((diff.detach()),dim=2,keepdim=True),\
              dim=3,keepdim=True)
  return X + diff*factor
  # return X + clip(diff,-epsilon,epsilon) 

def refine_BIM_ulta(model,X,no_of_steps,epsilon,epsilon_step) :
  Xn = X.clone()
  for i in range(no_of_steps) :
    delta = torch.zeros_like(X, requires_grad=True)
    X_recon = torch.sigmoid(model(Xn+delta))
    loss = -torch.mean((Xn+delta-X_recon)**2)
    loss.backward()
    Xn = Xn.clone() + epsilon_step * delta.grad.detach().sign()
  diff = Xn - X
  
  return X + clip(diff,-epsilon,epsilon) 

