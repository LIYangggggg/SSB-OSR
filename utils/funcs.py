import numpy as np
from pprint import pprint
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import Counter
from copy import deepcopy


def test_predict_MSP(model, test_loader):

    model.eval()
    id_preds = []       # Store class preds
    osr_preds = []      # Stores OSR preds

    # First extract all features
    for images, _, _, _ in tqdm(test_loader):

        images = images.cuda()
        logits = model(images) 
        sftmax = torch.nn.functional.softmax(logits, dim=-1)

        id_preds.extend(sftmax.argmax(dim=-1).detach().cpu().numpy())
        osr_preds.extend(sftmax.max(dim=-1)[0].detach().cpu().numpy())
        
    id_preds = np.array(id_preds)
    osr_preds = np.array(osr_preds)
    
    return id_preds, osr_preds

def test_predict_MSP_RP(model, test_loader, targets):

    model.eval()
    id_preds = []       # Store class preds
    osr_preds = []      # Stores OSR preds
    targets = torch.tensor(targets).cuda()
    targets = targets.unsqueeze(0)

    # First extract all features
    for images, _, _, _ in tqdm(test_loader):

        images = images.cuda()

        # Get logits
        logits = model(images)
        sftmax = torch.nn.functional.softmax(logits, dim=-1)
        
        sim = sftmax - targets
        conf, _ = torch.max(sim, dim=-1)

        id_preds.extend(sftmax.argmax(dim=-1).detach().cpu().numpy())
        osr_preds.extend(conf.data.detach().cpu().numpy())
        
    id_preds = np.array(id_preds)
    osr_preds = np.array(osr_preds)
    
    return id_preds, osr_preds

def test_predict_Energy(model, test_loader, T=1):
    model.eval()
    id_preds = []       # Store class preds
    energy_preds = []      # Stores OSR preds

    # First extract all features
    for images, _, _, _ in tqdm(test_loader):

        images = images.cuda()

        # Get logits
        logits = model(images)
        sftmax = torch.nn.functional.softmax(logits, dim=-1)

        id_preds.extend(sftmax.argmax(dim=-1).detach().cpu().numpy())
        energy_preds.extend(((T * torch.logsumexp(logits / T, dim=1))).detach().cpu().numpy())
        
    id_preds = np.array(id_preds)
    energy_preds = np.array(energy_preds)
    
    return id_preds, energy_preds

def test_predict_Energy_RW(model, test_loader, targets, T=1):
    model.eval()
    id_preds = []       # Store class preds
    energy_preds = []      # Stores OSR preds
    targets = torch.tensor(targets).cuda()
    targets = targets.unsqueeze(0)

    for images, _, _, _ in tqdm(test_loader):

        images = images.cuda()

        logits = model(images)
        sftmax = torch.nn.functional.softmax(logits, dim=-1)
        energy_score = T * torch.logsumexp(logits / T, dim=1)
        
        sim = -sftmax * targets
        sim = sim.sum(1) / (torch.norm(sftmax, dim=1) * torch.norm(targets, dim=1))
        sim = sim + 1

        energy_score = energy_score * sim
        id_preds.extend(sftmax.argmax(dim=-1).detach().cpu().numpy())
        energy_preds.extend(energy_score.detach().cpu().numpy())
        
    id_preds = np.array(id_preds)
    energy_preds = np.array(energy_preds)
    
    return id_preds, energy_preds

def test_predict_ODIN(model, test_loader, epsilon=0.0, T=1000):
    """
    Get class predictions and Energy Softmax Score for all instances in loader
    """

    model.eval()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    id_preds = []       # Store class preds
    odin_preds = []      # Stores OSR preds

    # First extract all features
    for b, (images, _, _, _) in enumerate(tqdm(test_loader)):
        inputs = Variable(images.cuda(), requires_grad=True)
        outputs = model(inputs)
        # Get logits
        sftmax = torch.nn.functional.softmax(outputs, dim=-1)
        
        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / T

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -epsilon, gradient)
        outputs = model(Variable(tempInputs))      
        outputs = outputs / T  
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        
        # save 
        id_preds.extend(sftmax.argmax(dim=-1).detach().cpu().numpy())
        odin_preds.extend(np.max(nnOutputs, axis=1))
        
    id_preds = np.array(id_preds)
    odin_preds = np.array(odin_preds)
    
    return id_preds, odin_preds

def test_predict_ODIN_RW(model, test_loader, targets, epsilon=0.0, T=1000):
    """
    Get class predictions and ODIN_RW Score for all instances in loader
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    id_preds = []       # Store class preds
    odin_preds = []      # Stores OSR preds
    targets = np.expand_dims(targets, axis=0)

    for b, (images, _, _, _) in enumerate(tqdm(test_loader)):
        inputs = Variable(images.cuda(), requires_grad=True)
        outputs = model(inputs)
        # Get logits
        sftmax = torch.nn.functional.softmax(outputs, dim=-1)
        
        id_preds.extend(sftmax.argmax(dim=-1).detach().cpu().numpy())
        
        sftmax = sftmax.data.cpu().numpy()
        sim = -sftmax * targets
        sim = sim.sum(axis=1) / (np.linalg.norm(sftmax, axis=-1) * np.linalg.norm(targets, axis=-1))
        sim = np.expand_dims(sim, axis=1)
        sim = sim + 1
        
        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / T

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -epsilon, gradient)
        outputs = model(Variable(tempInputs))      
        outputs = outputs / T  
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu().numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        
        nnOutputs = sim * nnOutputs
        odin_preds.extend(np.max(nnOutputs, axis=1))
        
    id_preds = np.array(id_preds)
    odin_preds = np.array(odin_preds)
    
    return id_preds, odin_preds

def test_predict_GradNorm(model, test_loader, T=1, num_classes=1000):
    """
    Get class predictions and Grad Norm Score for all instances in loader
    """

    model.eval()
    id_preds = []       # Store class preds
    gradnorm_preds = []      # Stores OSR preds
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

    # First extract all features
    for b,(images, _, _, _) in enumerate(tqdm(test_loader)):
        inputs = Variable(images.cuda(), requires_grad=True)
        model.zero_grad()
        # Get logits
        outputs = model(inputs)
        targets = torch.ones((inputs.shape[0], num_classes)).cuda()
        outputs = outputs / T
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))
        
        loss.backward()
        
        layer_grad = model.module.head.weight.grad.data
        layer_grad_norm = torch.sum(torch.abs(layer_grad)).detach().cpu().numpy()
        gradnorm_preds.append(layer_grad_norm)
        
        sftmax = torch.nn.functional.softmax(outputs, dim=-1)
        id_preds.append(sftmax.argmax(dim=-1).detach().cpu().numpy())
        
        
    id_preds = np.array(id_preds)
    gradnorm_preds = np.array(gradnorm_preds)
    
    return id_preds, gradnorm_preds

def test_predict_GradNorm_RP(model, test_loader, targets, T=1, num_classes=1000):
    """
    Get class predictions and Grad Norm Score for all instances in loader
    """

    model.eval()
    id_preds = []       # Store class preds
    gradnorm_preds = []      # Stores OSR preds
    targets = torch.tensor(targets).cuda()
    targets = targets.unsqueeze(0)
    feat_model = deepcopy(model)
    feat_model.module.head = nn.Sequential()

    # First extract all features
    for b,(images, _, _, _) in enumerate(tqdm(test_loader)):
        inputs = Variable(images.cuda(), requires_grad=True)
        # Get logits
        features = feat_model(inputs)
        outputs = model.module.head.forward(features)
        U = torch.norm(features, p=1, dim=1)
        out_softmax = torch.nn.functional.softmax(outputs, dim=1)
        V = torch.norm((targets - out_softmax), p=1, dim=1)
        S = U * V / 768 / num_classes
        
        id_preds.extend(out_softmax.argmax(dim=-1).detach().cpu().numpy())
        gradnorm_preds.extend(S.detach().cpu().numpy())
        
    id_preds = np.array(id_preds)
    gradnorm_preds = np.array(gradnorm_preds)
    
    return id_preds, gradnorm_preds
