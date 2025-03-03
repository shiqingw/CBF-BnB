import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from cores.lip_nn.models import LipschitzNetwork
from cores.utils.utils import seed_everything, save_nn_weights, save_dict, get_grad_l2_norm, format_time
from cores.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from cores.utils.draw_utils import draw_curve, draw_multiple_curves
from cores.utils.draw_utils import draw_safe_set_and_feasibility_contour_given_labels

def label_rectangles(A, B, a, b, c):
    """
    A:  (N, 3) array of [x, y, theta] for each rectangle
    B:  (D, 2) array of test points
    a, b, c: scalars that define the half-widths of the rectangle
             in its local frame, as per your definitions.

    c_1 = [x,y]^T + R(\theta) [a,b]^T
    c_2 = [x,y]^T + R(\theta) [a,-b]^T
    c_3 = [x,y]^T + R(\theta) [-c,-b]^T
    c_4 = [x,y]^T + R(\theta) [-c,b]^T
    R(\theta)=[cos(\theta), -sin(\theta);
                sin(\theta), cos(\theta)].

    Returns:
        labels: (N,) array of +1 or -1 for each row of A.
    """
    # Unpack columns from A
    x = A[:, 0]          # shape (N,)
    y = A[:, 1]          # shape (N,)
    theta = A[:, 2]      # shape (N,)

    # Precompute cos(theta), sin(theta) for each rectangle
    cos_t = np.cos(theta)  # shape (N,)
    sin_t = np.sin(theta)  # shape (N,)

    # We want to compute p' = R(-theta)*(p - [x, y]).
    # If p = (X, Y) and rectangle center is (x, y), then
    #   p' = [ cos(theta),  sin(theta)] * [X - x]
    #        [-sin(theta),  cos(theta)]   [Y - y]
    #
    # Notice that R(-theta) = R(theta)^T = [[ cos_t, sin_t],
    #                                       [-sin_t, cos_t]]
    #
    # Let's broadcast B so that for each i we have all B in one shot.

    # Shape transformations:
    #   B[:, 0] is shape (D,)
    #   x is shape (N,)
    # We want to create something of shape (N, D):
    #   x_diff[i, j] = B[j, 0] - x[i], etc.

    x_diff = B[:, 0][None, :] - x[:, None]  # shape (N, D)
    y_diff = B[:, 1][None, :] - y[:, None]  # shape (N, D)

    # Now apply R(-theta):
    # p'_x =  cos_t[i]*(x_diff[i, j]) + sin_t[i]*(y_diff[i, j])
    # p'_y = -sin_t[i]*(x_diff[i, j]) + cos_t[i]*(y_diff[i, j])
    #
    # We make cos_t, sin_t broadcast along the second dimension:

    p_prime_x = cos_t[:, None] * x_diff + sin_t[:, None] * y_diff
    p_prime_y = -sin_t[:, None] * x_diff + cos_t[:, None] * y_diff

    # Check if points lie within the rectangle in local coords.
    # The rectangle in local coords is: x' in [-c, a], y' in [-b, b].
    # inside[i, j] = True if sample i contains point j inside.

    inside = (
        (p_prime_x >= -c) & (p_prime_x <= a) &
        (p_prime_y >= -b) & (p_prime_y <= b)
    )
    # For each rectangle i, we check if ANY point j is inside.
    intersects_any = np.any(inside, axis=1)  # shape (N,)

    # Label: +1 if no intersection, -1 if intersects
    labels = np.where(intersects_any, -1, +1)

    return labels

def train_cbf(cbf_config, nn_input_bias, train_config, states_np, labels_np, system, 
              control_upper_bound, control_lower_bound, results_dir, step_counter, config, device):
    # Build CBF network
    print("> Building cbf neural network ...")
    cbf_alpha = cbf_config["alpha"]
    cbf_in_features = cbf_config["in_features"]
    cbf_out_features = cbf_config["out_features"]
    cbf_gamma = cbf_config["lipschitz_constant"]
    cbf_activations = [cbf_config["activations"]]*(cbf_config["num_layers"]-1)
    cbf_widths = [cbf_in_features]+[cbf_config["width_each_layer"]]*(cbf_config["num_layers"]-1)+[cbf_out_features]
    cbf_zero_at_zero = bool(cbf_config["zero_at_zero"])
    cbf_input_bias = np.array(nn_input_bias, dtype=config.np_dtype)
    cbf_input_transform = 1.0/np.array(cbf_config["input_transform_to_inverse"], dtype=config.np_dtype)
    cbf_nn = LipschitzNetwork(in_features=cbf_in_features, 
                                out_features=cbf_out_features,
                                gamma=cbf_gamma,
                                activations=cbf_activations,
                                widths=cbf_widths,
                                zero_at_zero=cbf_zero_at_zero,
                                input_bias=cbf_input_bias,
                                input_transform=cbf_input_transform,
                                dtype=config.pt_dtype,
                                random_psi=False,
                                trainable_psi=False)
    # summary(cbf_nn, input_size=(1, cbf_in_features), dtypes=[config.pt_dtype])
    # save_nn_weights(cbf_nn, f"{results_dir}/cbf_weights_init.pt")
    # lip_cbf_nn = cbf_nn.get_l2_lipschitz_bound()
    # print("==> Lipschitz constant of CBF network: {:.6f}".format(lip_cbf_nn))
    cbf_nn.to(device)

    # Trainset    
    train_state = torch.tensor(states_np, dtype=config.pt_dtype)
    train_label = torch.tensor(labels_np, dtype=config.pt_dtype)
    train_dataset = torch.utils.data.TensorDataset(train_state, train_label)
    batch_size = train_config["batch_size"]
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Define optimizer, learning rate scheduler, loss function, and loss monitor
    num_epochs = train_config["num_epochs"]
    optimizer = torch.optim.Adam([
        {'params': cbf_nn.parameters(), 'lr': train_config['cbf_lr'], 'weight_decay': train_config['cbf_wd']}
        ])
    scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, 
                                              max_lr=[train_config['cbf_lr']], 
                                              min_lr=[0.0], 
                                              first_cycle_steps=num_epochs, 
                                              warmup_steps=train_config["warmup_steps"])
    
    # Start training
    print("> Start training ...")
    safe_set_weight = train_config["safe_set_weight"]
    unsafe_set_weight = train_config["unsafe_set_weight"]
    feasibility_weight = train_config["feasibility_weight"]
    safe_set_margin = train_config["safe_set_margin"]
    unsafe_set_margin = train_config["unsafe_set_margin"]
    feasibility_margin = train_config["feasibility_margin"]
    train_loss_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    safe_set_loss_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    unsafe_set_loss_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    feasibility_loss_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    train_cbf_grad_norm_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    best_epoch_train_loss = float('inf')
    cbf_best_loss_loc = f"{results_dir}/cbf_weights_best_loss_{step_counter:06d}.pt"
    best_epoch = None
    cbf_nn.to(device)
    system.to(device)
    start_time = time.time()
    for epoch in range(num_epochs):
        # Train
        cbf_nn.train()

        epoch_train_loss = 0
        epoch_safe_set_loss = 0
        epoch_unsafe_set_loss = 0
        epoch_feasibility_loss = 0
        epoch_train_cbf_grad_norm = 0
        epoch_train_start_time = time.time()
        for batch_idx, (x, label) in enumerate(train_dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            h, h_dx = cbf_nn.forward_with_jacobian(x) # (batch_size, 1), (batch_size, 1, state_dim)
            h = h.squeeze(1) # (batch_size,)
            h_dx = h_dx.squeeze(1) # (batch_size, state_dim)

            # Safe and unsafe indices
            safe_set_idx = label > 0
            unsafe_set_idx = label < 0

            # Loss for safe set (label = 1)
            h_safe_set = h[safe_set_idx] # (batch_size,)
            loss_safe_set = safe_set_weight * torch.max(-h_safe_set + safe_set_margin, torch.zeros_like(h_safe_set)).mean()
            
            # Loss for unsafe set (label = -1)
            h_unsafe_set = h[unsafe_set_idx] # (batch_size,)
            loss_unsafe_set = unsafe_set_weight * torch.max(h_unsafe_set + unsafe_set_margin, torch.zeros_like(h_unsafe_set)).mean()

            # Loss for feasibility
            x_safe = x[safe_set_idx] # (safe_size, state_dim)
            h_dx_safe = h_dx[safe_set_idx] # (safe_size, state_dim)
            drift_safe = system.get_drift(x_safe) # (safe_size, state_dim)
            actuation_safe = system.get_actuation(x_safe) # (safe_size, state_dim, control_dim)
            h_dx_f_safe = (h_dx_safe * drift_safe).sum(dim=1) # (safe_size,)
            h_dx_g_safe = torch.bmm(h_dx_safe.unsqueeze(1), actuation_safe).squeeze(1) # (safe_size, control_dim)
            h_dx_g_safe_pos = torch.max(h_dx_g_safe, torch.zeros_like(h_dx_g_safe)) # (safe_size, control_dim)
            h_dx_g_safe_neg = torch.min(h_dx_g_safe, torch.zeros_like(h_dx_g_safe)) # (safe_size, control_dim)
            tmp = h_dx_f_safe # (safe_size,)
            tmp += (h_dx_g_safe_pos * control_upper_bound).sum(dim=1) # (safe_size,)
            tmp += (h_dx_g_safe_neg * control_lower_bound).sum(dim=1) # (safe_size,)
            tmp += cbf_alpha * h_safe_set # (safe_size,)

            loss_feasibility = feasibility_weight * torch.max(-tmp + feasibility_margin, torch.zeros_like(tmp)).mean()

            # Total loss
            loss = loss_safe_set + loss_unsafe_set + loss_feasibility
            loss.backward()
            cbf_grad_norm = get_grad_l2_norm(cbf_nn)
            torch.nn.utils.clip_grad_norm_(cbf_nn.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                epoch_train_loss += loss.detach().cpu().numpy()
                epoch_safe_set_loss += loss_safe_set.detach().cpu().numpy()
                epoch_unsafe_set_loss += loss_unsafe_set.detach().cpu().numpy()
                epoch_feasibility_loss += loss_feasibility.detach().cpu().numpy()
                epoch_train_cbf_grad_norm += cbf_grad_norm
                
        epoch_train_end_time = time.time()
        epoch_train_loss = epoch_train_loss/(batch_idx+1)
        epoch_safe_set_loss = epoch_safe_set_loss/(batch_idx+1)
        epoch_unsafe_set_loss = epoch_unsafe_set_loss/(batch_idx+1)
        epoch_feasibility_loss = epoch_feasibility_loss/(batch_idx+1)
        epoch_train_cbf_grad_norm = epoch_train_cbf_grad_norm/(batch_idx+1)
        
        if epoch % 5 == 0:
            print("Epoch: {:03d} | Train Loss: {:.4E} | CBF GN: {:.4E} | Time: {}".format(
                epoch+1,
                epoch_train_loss, 
                epoch_train_cbf_grad_norm,
                format_time(epoch_train_end_time - epoch_train_start_time)))
        train_loss_monitor[epoch] = epoch_train_loss
        safe_set_loss_monitor[epoch] = epoch_safe_set_loss
        unsafe_set_loss_monitor[epoch] = epoch_unsafe_set_loss
        feasibility_loss_monitor[epoch] = epoch_feasibility_loss
        train_cbf_grad_norm_monitor[epoch] = epoch_train_cbf_grad_norm

        # Save best model
        if epoch_train_loss < best_epoch_train_loss:
            best_epoch_train_loss = epoch_train_loss
            best_epoch = epoch
            save_nn_weights(cbf_nn, cbf_best_loss_loc)

        scheduler.step()
    
    end_time = time.time()
    print("> Total time: {}".format(format_time(end_time - start_time)))

    def feasibility_fun(x):
        h, h_dx = cbf_nn.forward_with_jacobian(x) # (batch_size, 1), (batch_size, 1, state_dim)
        h = h.squeeze(1) # (batch_size,)
        h_dx = h_dx.squeeze(1) # (batch_size, state_dim)
        drift = system.get_drift(x) # (batch_size, state_dim)
        actuation = system.get_actuation(x) # (batch_size, state_dim, control_dim)
        h_dx_f = (h_dx * drift).sum(dim=1) # (batch_size,)
        h_dx_g = torch.bmm(h_dx.unsqueeze(1), actuation).squeeze(1) # (batch_size, control_dim)
        h_dx_g_pos = torch.max(h_dx_g, torch.zeros_like(h_dx_g)) # (batch_size, control_dim)
        h_dx_g_neg = torch.min(h_dx_g, torch.zeros_like(h_dx_g)) # (batch_size, control_dim)
        tmp = h_dx_f
        tmp += (h_dx_g_pos * control_upper_bound).sum(dim=1)
        tmp += (h_dx_g_neg * control_lower_bound).sum(dim=1)
        tmp += cbf_alpha * h
        return tmp
    
    # Calculate statistics
    num_true_safe = np.sum(labels_np > 0)
    num_true_unsafe = np.sum(labels_np < 0)
    cbf_torch = cbf_nn.forward(train_state).detach().cpu().squeeze(1) # (num_samples,)
    cbf_np = cbf_torch.numpy()
    num_pred_safe = np.sum(cbf_np[labels_np > 0] >= 0)
    num_pred_unsafe = np.sum(cbf_np[labels_np < 0] < 0)
    
    feasibility_on_pred_safe = feasibility_fun(train_state[cbf_torch >= 0]).detach().cpu().numpy() # (num_samples,)
    feasibility_on_pred_safe = np.sum(feasibility_on_pred_safe >= 0)
    print(f"> Safe percentage: {num_pred_safe/num_true_safe:.4f}")
    print(f"> Unsafe percentage: {num_pred_unsafe/num_true_unsafe:.4f}")
    print(f"> Feasibility on pred safe: {feasibility_on_pred_safe/num_pred_safe:.4f}")

    draw_curve(data=train_loss_monitor,
               ylabel="Train Loss", 
               savepath=f"{results_dir}/00_train_train_loss_{step_counter:06d}.png", 
               dpi=100)
    draw_multiple_curves(data_list=[safe_set_loss_monitor, unsafe_set_loss_monitor, feasibility_loss_monitor],
                         label_list=["Safe set loss", "Unsafe set loss", "Feasibility loss"],
                         savepath=f"{results_dir}/00_train_train_loss_decomp_{step_counter:06d}.png",
                         dpi=100)
    draw_curve(data=train_cbf_grad_norm_monitor,
               ylabel="CBF train grad norm",
               savepath=f"{results_dir}/00_train_cbf_grad_norm_{step_counter:06d}.png",
               dpi=100)
    
    # Plots
    print("> Visualizing the CBF function ...")
    state_lower_bound = np.min(states_np, axis=0)
    state_upper_bound = np.max(states_np, axis=0)
        
    print("> Visualizing the feasibility condition ...")
    pairwise_idx = [(0,1), (0,2), (1,2)]
    state_labels = [r"$x$", r"$y$", r"$\theta$"]
    state_names = ["x", "y", "theta"]
    for (x_idx, y_idx) in pairwise_idx:
        draw_safe_set_and_feasibility_contour_given_labels(cbf_nn=cbf_nn,
                                        test_func=feasibility_fun,
                                        state_lower_bound=state_lower_bound, 
                                        state_upper_bound=state_upper_bound,
                                        mesh_size=400, 
                                        x_state_idx=x_idx,
                                        y_state_idx=y_idx,
                                        x_label=state_labels[x_idx],
                                        y_label=state_labels[y_idx],
                                        states_np=states_np,
                                        labels_np=labels_np,
                                        savepath=f"{results_dir}/00_cond_safe_and_feasibility_contour_{state_names[x_idx]}_{state_names[y_idx]}_{step_counter:06d}.pdf", 
                                        dpi=100, 
                                        device=device, 
                                        pt_dtype=config.pt_dtype)

    return cbf_nn