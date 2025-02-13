import json
import sys
import os
import argparse
import shutil
import torch
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from torchinfo import summary
from torch.utils.data import DataLoader
import time

from cores.dynamical_systems.create_system import get_system
from cores.lip_nn.models import LipschitzNetwork
from cores.utils.utils import seed_everything, save_nn_weights, save_dict, get_grad_l2_norm, format_time
from cores.utils.config import Configuration
from cores.utils.draw_utils import draw_curve, draw_multiple_curves
from cores.utils.draw_utils import draw_safe_set_contour, draw_feasibility_condition_contour
from cores.cosine_annealing_warmup import CosineAnnealingWarmupRestarts

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=1, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    args = parser.parse_args()

    # Create result directory
    print("==> Creating result directory ...")
    exp_num = args.exp_num
    results_dir = "{}/eg2_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    shutil.copy(test_settings_path, results_dir)

    # Load test settings
    print("==> Loading test settings ...")
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Decide torch device
    print("==> Deciding torch device ...")
    config = Configuration()
    user_device = args.device
    if user_device != "None":
        device = torch.device(user_device)
    else:
        device = config.device
    cpu = torch.device("cpu")
    print('==> torch device: ', device)

    # Seed everything
    print("==> Seeding everything ...")
    seed = test_settings["seed"]
    seed_everything(seed)

    # Build dynamical system
    print("==> Building dynamical system ...")
    system_name = test_settings["true_system_name"]
    system = get_system(system_name=system_name, 
                        dtype=config.pt_dtype)
    save_nn_weights(system, f"{results_dir}/system_params.pt")

    # Build CBF network
    print("==> Building cbf neural network ...")
    cbf_config = test_settings["cbf_nn_config"]
    cbf_alpha = cbf_config["alpha"]
    cbf_in_features = cbf_config["in_features"]
    cbf_out_features = cbf_config["out_features"]
    cbf_gamma = cbf_config["lipschitz_constant"]
    cbf_activations = [cbf_config["activations"]]*(cbf_config["num_layers"]-1)
    cbf_widths = [cbf_in_features]+[cbf_config["width_each_layer"]]*(cbf_config["num_layers"]-1)+[cbf_out_features]
    cbf_zero_at_zero = bool(cbf_config["zero_at_zero"])
    cbf_input_bias = np.array(cbf_config["input_bias"], dtype=config.np_dtype)
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
    summary(cbf_nn, input_size=(1, cbf_in_features), dtypes=[config.pt_dtype])
    save_nn_weights(cbf_nn, f"{results_dir}/cbf_weights_init.pt")
    lip_cbf_nn = cbf_nn.get_l2_lipschitz_bound()
    print("==> Lipschitz constant of CBF network: {:.6f}".format(lip_cbf_nn))
    cbf_nn.to(device)

    # Control config
    control_config = test_settings["control_config"]
    control_lower_bound = torch.tensor(control_config["lower_bound"], dtype=config.pt_dtype, device=device)
    control_upper_bound = torch.tensor(control_config["upper_bound"], dtype=config.pt_dtype, device=device)

    # Disturbance config
    disturbance_config = test_settings["disturbance_config"]
    disturbance_channel_matrix = torch.tensor(disturbance_config["channel_matrix"], dtype=config.pt_dtype, device=device)
    disturbance_lower_bound = torch.tensor(disturbance_config["lower_bound"], dtype=config.pt_dtype, device=device)
    disturbance_upper_bound = torch.tensor(disturbance_config["upper_bound"], dtype=config.pt_dtype, device=device)

    # The state space
    dataset_config = test_settings["dataset_config"]
    state_lower_bound = np.array(dataset_config["state_lower_bound"], dtype=config.np_dtype)
    state_upper_bound = np.array(dataset_config["state_upper_bound"], dtype=config.np_dtype)
    mesh_size = dataset_config["mesh_size"]
    state_dim = system.state_dim
    meshgrid = np.meshgrid(*[np.linspace(state_lower_bound[i], state_upper_bound[i], mesh_size[i]) for i in range(state_dim)])
    state_np = np.concatenate([meshgrid[i].reshape(-1, 1) for i in range(state_dim)], axis=1)
    del meshgrid
    print("==> Amount of training data: ", state_np.shape[0])
    
    # The safe set
    unsafe_set_config = test_settings["unsafe_set_config"]
    unsafe_set_lower_bound = np.array(unsafe_set_config["unsafe_set_lower_bound"], dtype=config.np_dtype)
    unsafe_set_upper_bound = np.array(unsafe_set_config["unsafe_set_upper_bound"], dtype=config.np_dtype)
    is_unsafe = np.all((state_np >= unsafe_set_lower_bound) & (state_np <= unsafe_set_upper_bound), axis=1)
    label_np = np.where(is_unsafe, -1, 1)
    del is_unsafe

    # Create training and test data loader
    print("==> Creating training data ...")
    train_config = test_settings["train_config"]
    train_state = torch.tensor(state_np, dtype=config.pt_dtype)
    train_label = torch.tensor(label_np, dtype=config.pt_dtype)
    train_dataset = torch.utils.data.TensorDataset(train_state, train_label)
    batch_size = train_config["batch_size"]
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = train_dataloader

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
    print("==> Start training ...")
    safe_set_weight = train_config["safe_set_weight"]
    unsafe_set_weight = train_config["unsafe_set_weight"]
    feasibility_weight = train_config["feasibility_weight"]
    unsafe_set_margin = train_config["unsafe_set_margin"]
    feasibility_margin = train_config["feasibility_margin"]
    train_loss_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    safe_set_loss_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    unsafe_set_loss_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    feasibility_loss_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    test_loss_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    train_cbf_grad_norm_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    best_epoch_test_loss = float('inf')
    cbf_best_loss_loc = f"{results_dir}/cbf_weights_best_loss.pt"
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
            loss_safe_set = safe_set_weight * torch.max(-h_safe_set, torch.zeros_like(h_safe_set)).mean()
            
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

            # Add disturbance
            h_dx_G_safe = torch.matmul(h_dx_safe, disturbance_channel_matrix) # (safe_size, disturbance_dim)
            h_dx_G_safe_pos = torch.max(h_dx_G_safe, torch.zeros_like(h_dx_G_safe)) # (safe_size, disturbance_dim)
            h_dx_G_safe_neg = torch.min(h_dx_G_safe, torch.zeros_like(h_dx_G_safe)) # (safe_size, disturbance_dim)
            tmp += (h_dx_G_safe_pos * disturbance_lower_bound).sum(dim=1) # (safe_size,)
            tmp += (h_dx_G_safe_neg * disturbance_upper_bound).sum(dim=1) # (safe_size,)
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
        
        # Test
        cbf_nn.eval()
        epoch_test_loss = 0
        epoch_test_start_time = time.time()

        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(test_dataloader):
                x = x.to(device)
                h, h_dx = cbf_nn.forward_with_jacobian(x) # (batch_size, 1), (batch_size, 1, state_dim)
                h = h.squeeze(1) # (batch_size,)
                h_dx = h_dx.squeeze(1) # (batch_size, state_dim)

                # Safe and unsafe indices
                safe_set_idx = label > 0
                unsafe_set_idx = label < 0

                # Loss for safe set (label = 1)
                h_safe_set = h[safe_set_idx] # (batch_size,)
                loss_safe_set = safe_set_weight * torch.max(-h_safe_set, torch.zeros_like(h_safe_set)).mean()
                
                # Loss for unsafe set (label = -1)
                h_unsafe_set = h[unsafe_set_idx] # (batch_size,)
                loss_unsafe_set = unsafe_set_weight * torch.max(h_unsafe_set, torch.zeros_like(h_unsafe_set)).mean()

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

                # Add disturbance
                h_dx_G_safe = torch.matmul(h_dx_safe, disturbance_channel_matrix) # (safe_size, disturbance_dim)
                h_dx_G_safe_pos = torch.max(h_dx_G_safe, torch.zeros_like(h_dx_G_safe)) # (safe_size, disturbance_dim)
                h_dx_G_safe_neg = torch.min(h_dx_G_safe, torch.zeros_like(h_dx_G_safe)) # (safe_size, disturbance_dim)
                tmp += (h_dx_G_safe_pos * disturbance_lower_bound).sum(dim=1) # (safe_size,)
                tmp += (h_dx_G_safe_neg * disturbance_upper_bound).sum(dim=1) # (safe_size,)
                loss_feasibility = feasibility_weight * torch.max(-tmp, torch.zeros_like(tmp)).mean()

                # Total loss
                loss = loss_safe_set + loss_unsafe_set + loss_feasibility
                epoch_test_loss += loss.detach().cpu().numpy()
                
        epoch_test_end_time = time.time()
        epoch_test_loss = epoch_test_loss/(batch_idx+1)
        print("Epoch: {:03d} | Test Loss: {:.4E} | Time: {}".format(epoch+1,
                    epoch_test_loss, format_time(epoch_test_end_time - epoch_test_start_time)))
        test_loss_monitor[epoch] = epoch_test_loss

        # Save the model if the test loss is the best
        if epoch_test_loss < best_epoch_test_loss:
            best_epoch_test_loss = epoch_test_loss
            torch.save(cbf_nn.state_dict(), cbf_best_loss_loc)
            print("> Save at epoch {:03d} | Test loss {:.4E}".format(epoch+1, best_epoch_test_loss))

        scheduler.step()
    
    end_time = time.time()
    print("Total time: {}".format(format_time(end_time - start_time)))

    del train_state, train_dataset, train_dataloader, test_dataloader, label_np, state_np

    # Visualize training loss
    print("==> Visualizing training and test losses ...")
    draw_curve(data=train_loss_monitor,
               ylabel="Train Loss", 
               savepath=f"{results_dir}/00_train_train_loss.png", 
               dpi=100)
    draw_multiple_curves(data_list=[safe_set_loss_monitor, unsafe_set_loss_monitor, feasibility_loss_monitor],
                         label_list=["Safe set loss", "Unsafe set loss", "Feasibility loss"],
                         savepath=f"{results_dir}/00_train_train_loss_decomp.png",
                         dpi=100)
    draw_curve(data=train_cbf_grad_norm_monitor,
               ylabel="CBF train grad norm",
               savepath=f"{results_dir}/00_train_cbf_grad_norm.png",
               dpi=100)
    draw_curve(data=test_loss_monitor,
               ylabel="Test Loss",
               savepath=f"{results_dir}/00_train_test_loss.png",
               dpi=100)

    # Load the best weights
    cbf_nn.load_state_dict(torch.load(cbf_best_loss_loc, weights_only=True, map_location=device))
    cbf_nn.eval()

    # Compute the smoothness constants
    print("==> Computing the smoothness constants ...")
    lip_cbf_nn = cbf_nn.get_l2_lipschitz_bound()
    hess_bound_cbf_nn = cbf_nn.get_l2_hessian_bound()
    print("> Lipschitz constant of CBF network: {:.6f}".format(lip_cbf_nn))
    print("> Hessian bound of CBF network: {:.6f}".format(hess_bound_cbf_nn))

    # Save the training results
    print("==> Saving the training results ...")
    train_results = {
        "time": end_time - start_time,
        "train_loss": train_loss_monitor,
        "safe_set_loss": safe_set_loss_monitor,
        "unsafe_set_loss": unsafe_set_loss_monitor,
        "feasibility_loss": feasibility_loss_monitor,
        "test_loss": test_loss_monitor,
        "train_cbf_grad_norm": train_cbf_grad_norm_monitor,
        "lip_cbf_nn": lip_cbf_nn,
        "hess_bound_cbf_nn": hess_bound_cbf_nn
    }
    save_dict(train_results, f"{results_dir}/train_results.pkl")

    # Check the CBF function and feasibility condition
    print("==> Visualizing the CBF function and feasibility condition ...")
    post_mesh_size = dataset_config["post_mesh_size"]
    meshgrid = np.meshgrid(*[np.linspace(state_lower_bound[i], state_upper_bound[i], post_mesh_size[i]) for i in range(state_dim)])
    state_flatten_np = np.concatenate([meshgrid[i].reshape(-1, 1) for i in range(state_dim)], axis=1)
    is_unsafe = np.all((state_flatten_np >= unsafe_set_lower_bound) & (state_flatten_np <= unsafe_set_upper_bound), axis=1)
    label_np = np.where(is_unsafe, -1, 1)
    del meshgrid, is_unsafe
    state_flatten_torch = torch.tensor(state_flatten_np, dtype=config.pt_dtype)
    dataset = torch.utils.data.TensorDataset(state_flatten_torch)
    batch_size = 512
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    h_flatten_torch = torch.zeros((state_flatten_np.shape[0],), dtype=config.pt_dtype)
    feasibility_flatten_torch = torch.zeros((state_flatten_np.shape[0],), dtype=config.pt_dtype)
    for (batch_idx, (state_batch,)) in enumerate(dataloader):
        state_batch = state_batch.to(device)
        h_batch, h_dx_batch = cbf_nn.forward_with_jacobian(state_batch)
        h_batch = h_batch.squeeze(1)
        h_dx_batch = h_dx_batch.squeeze(1)
        h_flatten_torch[batch_idx*batch_size:min((batch_idx+1)*batch_size, h_flatten_torch.shape[0])] = h_batch.detach().cpu()
        drift_batch = system.get_drift(state_batch) # (batch_size, state_dim)
        actuation_batch = system.get_actuation(state_batch) # (batch_size, state_dim, control_dim)
        h_dx_f_batch = (h_dx_batch * drift_batch).sum(dim=1) # (batch_size,)
        h_dx_g_batch = torch.bmm(h_dx_batch.unsqueeze(1), actuation_batch).squeeze(1) # (batch_size, control_dim)
        h_dx_g_pos_batch = torch.max(h_dx_g_batch, torch.zeros_like(h_dx_g_batch)) # (batch_size, control_dim)
        h_dx_g_neg_batch = torch.min(h_dx_g_batch, torch.zeros_like(h_dx_g_batch)) # (batch_size, control_dim)
        tmp = h_dx_f_batch
        tmp += (h_dx_g_pos_batch * control_upper_bound).sum(dim=1)
        tmp += (h_dx_g_neg_batch * control_lower_bound).sum(dim=1)
        tmp += cbf_alpha * h_batch
        h_dx_G_batch = torch.matmul(h_dx_batch, disturbance_channel_matrix) # (batch_size, disturbance_dim)
        h_dx_G_pos_batch = torch.max(h_dx_G_batch, torch.zeros_like(h_dx_G_batch)) # (batch_size, disturbance_dim)
        h_dx_G_neg_batch = torch.min(h_dx_G_batch, torch.zeros_like(h_dx_G_batch)) # (batch_size, disturbance_dim)
        tmp += (h_dx_G_pos_batch * disturbance_lower_bound).sum(dim=1)
        tmp += (h_dx_G_neg_batch * disturbance_upper_bound).sum(dim=1)
        feasibility_flatten_torch[batch_idx*batch_size:min((batch_idx+1)*batch_size, feasibility_flatten_torch.shape[0])] = tmp.detach().cpu()

    del state_flatten_torch, dataset, dataloader

    print("==> Checking the safe and unsafe set ...")
    h_flatten_np = h_flatten_torch.detach().cpu().numpy()
    h_flatten_np = h_flatten_np.squeeze()
    
    # Calculate the percentage of h_flatten_np >= 0 within label_np > 0
    safe_set_idx = label_np > 0
    safe_set_count = np.sum(safe_set_idx)
    safe_set_percentage = np.sum(h_flatten_np[safe_set_idx] >= 0) / safe_set_count
    print(f"> Safe set percentage: {safe_set_percentage:.4f}")

    # Calculate the percentage of h_flatten_np < 0 within label_np < 0
    unsafe_set_idx = label_np < 0
    unsafe_set_count = np.sum(unsafe_set_idx)
    unsafe_set_percentage = np.sum(h_flatten_np[unsafe_set_idx] < 0) / unsafe_set_count
    print(f"> Unsafe set percentage: {unsafe_set_percentage:.4f}")

    # Calculate the percentage of feasibility_flatten_torch >= 0 within h_flatten_np >= 0
    feasibility_flatten_np = feasibility_flatten_torch.detach().cpu().numpy()
    feasibility_flatten_np = feasibility_flatten_np.squeeze()
    feasibility_idx = h_flatten_np >= 0
    feasibility_count = np.sum(feasibility_idx)
    feasibility_percentage = np.sum(feasibility_flatten_np[feasibility_idx] >= 0) / feasibility_count
    print(f"> Feasibility percentage: {feasibility_percentage:.4f}")

    del state_flatten_np, h_flatten_np, feasibility_flatten_np, safe_set_idx, unsafe_set_idx, feasibility_idx

    def is_safe_fun(x):
        is_unsafe = np.all((x >= unsafe_set_lower_bound) & (x <= unsafe_set_upper_bound), axis=1)
        label_np = np.where(is_unsafe, -1, 1)
        return label_np
    
    # Plots
    print("==> Visualizing the CBF function and stability condition ...")
    pairwise_idx = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    state_labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]
    state_names = ["x", "y", "vx", "vy"]
    for (x_idx, y_idx) in pairwise_idx:
        draw_safe_set_contour(cbf_nn=cbf_nn,
                            state_lower_bound=state_lower_bound, 
                            state_upper_bound=state_upper_bound, 
                            mesh_size=400,
                            x_state_idx=x_idx,
                            y_state_idx=y_idx,
                            x_label=state_labels[x_idx],
                            y_label=state_labels[y_idx],
                            is_safe_func=is_safe_fun,
                            particular_level=None,
                            savepath=f"{results_dir}/00_cond_safe_contour_{state_names[x_idx]}_{state_names[y_idx]}.png", 
                            dpi=100,
                            device=device, 
                            pt_dtype=config.pt_dtype)

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
        h_dx_G = torch.matmul(h_dx, disturbance_channel_matrix) # (safe_size, disturbance_dim)
        h_dx_G_pos = torch.max(h_dx_G, torch.zeros_like(h_dx_G)) # (safe_size, disturbance_dim)
        h_dx_G_neg = torch.min(h_dx_G, torch.zeros_like(h_dx_G)) # (safe_size, disturbance_dim)
        tmp += (h_dx_G_pos * disturbance_lower_bound).sum(dim=1)
        tmp += (h_dx_G_neg * disturbance_upper_bound).sum(dim=1)
        return tmp
    
    print("==> Visualizing the feasibility condition ...")
    pairwise_idx = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    state_labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]
    state_names = ["x", "y", "vx", "vy"]
    for (x_idx, y_idx) in pairwise_idx:
        draw_feasibility_condition_contour(model=feasibility_fun,
                                        state_lower_bound=state_lower_bound, 
                                        state_upper_bound=state_upper_bound,
                                        mesh_size=400, 
                                        x_state_idx=x_idx,
                                        y_state_idx=y_idx,
                                        x_label=state_labels[x_idx],
                                        y_label=state_labels[y_idx],
                                        is_safe_func=is_safe_fun,
                                        savepath=f"{results_dir}/00_cond_feasibility_contour_{state_names[x_idx]}_{state_names[y_idx]}.png", 
                                        dpi=100, 
                                        device=device, 
                                        pt_dtype=config.pt_dtype)

    print("==> Done!")