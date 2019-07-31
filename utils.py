import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def forward_and_loss(model, x, y, loss_fn, pad_token):
    out, hidden_state = model(x)
    loss = loss_fn(out.view(-1, out.size(-1)), y.view(-1), ignore_index=pad_token)
    return out, loss

def train_model(model, optimizer, train_iter, pad_token, weight_path=None, device=None):
    total_loss = 0.0
    
    model.train()
    
    with tqdm(total=len(train_iter)) as pbar:
        for x, y in train_iter: 
            if device is not None and device.type=='cuda':
                x = x.cuda()
                y = y.cuda()
                
            optimizer.zero_grad()
            _, loss = forward_and_loss(model, x, y, F.cross_entropy, pad_token=pad_token)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.update(1)
            pbar.set_description("%-10s = %.6f  " % ('loss', total_loss))
            
    # Save model
    if weight_path is not None:
        state = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict()
        }
        
        torch.save(state, weight_path)
    
    return total_loss

def evaluate_model(model, val_iter, pad_token, device=None):
    model.eval()
    with torch.no_grad(), tqdm(total=len(val_iter)) as pbar:
        total_loss = 0.0

        for x, y in val_iter:
            if device is not None and device.type=='cuda':
                x = x.cuda()
                y = y.cuda()

            _, loss = forward_and_loss(model, x, y, F.cross_entropy, pad_token=pad_token)
            
            total_loss += loss.item()
            
            pbar.update(1)
            pbar.set_description("%-10s = %.6f  " % ('val_loss', total_loss))
    
    return total_loss