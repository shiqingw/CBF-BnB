import time
import torch
from .utils import format_time, get_grad_l2_norm
import numpy as np

def train_nn_mse_loss_no_test(model, optimizer, scheduler, num_epochs, train_dataloader, best_loc, device, threshold=1e-6):
    loss_monitor = np.zeros(num_epochs, dtype=np.float32)
    grad_norm_monitor = np.zeros(num_epochs, dtype=np.float32)
    model.to(device)
    best_epoch_loss = float('inf')
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_grad_norm = 0
        epoch_start_time = time.time()
        for batch_idx, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = torch.nn.functional.mse_loss(out, y)
            loss.backward()
            grad_norm = get_grad_l2_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                epoch_grad_norm += grad_norm
                epoch_loss += loss.detach().cpu().numpy()
        epoch_end_time = time.time()
        epoch_loss = epoch_loss/(batch_idx+1)
        epoch_grad_norm = epoch_grad_norm/(batch_idx+1)
        if epoch % 5 == 0:
            print("Epoch: {:03d} | Loss: {:.4E} | Grad Norm: {:.4E} | Time: {}".format(
                epoch+1,
                epoch_loss,
                epoch_grad_norm,
                format_time(epoch_end_time - epoch_start_time)))
        loss_monitor[epoch] = epoch_loss
        grad_norm_monitor[epoch] = epoch_grad_norm
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            torch.save(model.state_dict(), best_loc)
            print("> Save model at epoch {:03d} with loss {:.4E}".format(epoch+1, best_epoch_loss))
        scheduler.step()
        if best_epoch_loss < threshold:
            print("> Threshold reached. Stop training.")
            break
    
    end_time = time.time()
    print("Total time: {}".format(format_time(end_time - start_time)))
    
    return loss_monitor, grad_norm_monitor

def pretrain_lyapunov_nn(lyapunov_nn, optimizer, scheduler, num_epochs, train_dataloader, boundary_state_torch, positive_weight, positive_margin, 
                         boundary_weight, boundary_margin, forward_inv_weight, lip_lyapunov_nn, stability_cutoff_radius,
                         best_loc, device, threshold=1e-6):
    loss_monitor = np.zeros(num_epochs, dtype=np.float32)
    grad_norm_monitor = np.zeros(num_epochs, dtype=np.float32)
    lyapunov_nn.to(device)
    best_epoch_loss = float('inf')
    start_time = time.time()
    for epoch in range(num_epochs):
        lyapunov_nn.train()
        epoch_loss = 0
        epoch_grad_norm = 0
        epoch_start_time = time.time()
        for batch_idx, (x,) in enumerate(train_dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            V = lyapunov_nn(x)

            loss_positive = positive_weight * torch.max(-V + positive_margin, torch.zeros_like(V)).mean()

            V_boundary = lyapunov_nn(boundary_state_torch.to(device))
            loss_forward_inv = forward_inv_weight * torch.std(V_boundary)

            loss_boundary = boundary_weight * torch.max(-V_boundary + lip_lyapunov_nn*stability_cutoff_radius + boundary_margin, torch.zeros_like(V_boundary)).mean()

            loss = loss_positive + loss_forward_inv + loss_boundary

            loss.backward()
            grad_norm = get_grad_l2_norm(lyapunov_nn)
            torch.nn.utils.clip_grad_norm_(lyapunov_nn.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                epoch_grad_norm += grad_norm
                epoch_loss += loss.detach().cpu().numpy()
        epoch_end_time = time.time()
        epoch_loss = epoch_loss/(batch_idx+1)
        epoch_grad_norm = epoch_grad_norm/(batch_idx+1)
        if epoch % 5 == 0:
            print("Epoch: {:03d} | Loss: {:.4E} | Grad Norm: {:.4E} | Time: {}".format(
                epoch+1,
                epoch_loss,
                epoch_grad_norm,
                format_time(epoch_end_time - epoch_start_time)))
        loss_monitor[epoch] = epoch_loss
        grad_norm_monitor[epoch] = epoch_grad_norm
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            torch.save(lyapunov_nn.state_dict(), best_loc)
            print("> Save model at epoch {:03d} with loss {:.4E}".format(epoch+1, best_epoch_loss))
        scheduler.step()
        if best_epoch_loss < threshold:
            print("> Threshold reached. Stop training.")
            break
    
    end_time = time.time()
    print("Total time: {}".format(format_time(end_time - start_time)))
    
    return loss_monitor, grad_norm_monitor