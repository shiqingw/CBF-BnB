import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from cores.dynamical_systems.create_system import get_system
from cores.lip_nn.models import LipschitzNetwork
from cores.utils.config import Configuration
from cores.mesh.mesh_tools import decompose_hyperrectangle

import numpy as np
import matplotlib.pyplot as plt

def draw_safe_set(exp_num):
    print("==> Exp_num:", exp_num)
    results_dir = "{}/eg2_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    save_dir = "{}/eg2_results/{:03d}".format(str(Path(__file__).parent.parent), 0)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load test settings
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Decide torch device
    print("==> Deciding torch device ...")
    config = Configuration()
    device = config.device
    print('==> torch device: ', device)

    # Build dynamical system
    print("==> Building dynamical system ...")
    system_name = test_settings["true_system_name"]
    system = get_system(system_name=system_name, 
                        dtype=config.pt_dtype).to(device)
    system.load_state_dict(torch.load(f"{results_dir}/system_params.pt", weights_only=True, map_location=device))
    system.eval()

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
    cbf_nn.load_state_dict(torch.load(f"{results_dir}/cbf_weights_best_loss.pt", weights_only=True, map_location=device))
    cbf_nn.to(device)
    cbf_nn.eval()

    # The state space
    dataset_config = test_settings["dataset_config"]
    state_lower_bound = torch.tensor(dataset_config["state_lower_bound"], dtype=config.pt_dtype)
    state_upper_bound = torch.tensor(dataset_config["state_upper_bound"], dtype=config.pt_dtype)

    # Control config
    control_config = test_settings["control_config"]
    control_lower_bound = torch.tensor(control_config["lower_bound"], dtype=config.pt_dtype, device=device)
    control_upper_bound = torch.tensor(control_config["upper_bound"], dtype=config.pt_dtype, device=device)

    # Disturbance config
    disturbance_config = test_settings["disturbance_config"]
    disturbance_channel_matrix = torch.tensor(disturbance_config["channel_matrix"], dtype=config.pt_dtype, device=device)
    disturbance_lower_bound = torch.tensor(disturbance_config["lower_bound"], dtype=config.pt_dtype, device=device)
    disturbance_upper_bound = torch.tensor(disturbance_config["upper_bound"], dtype=config.pt_dtype, device=device)

    # Test function
    def test_func(x):
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
    
    # The safe set
    unsafe_set_config = test_settings["unsafe_set_config"]
    unsafe_set_lower_bound = torch.tensor(unsafe_set_config["unsafe_set_lower_bound"], dtype=config.pt_dtype)
    unsafe_set_upper_bound = torch.tensor(unsafe_set_config["unsafe_set_upper_bound"], dtype=config.pt_dtype)
    X_unsafe_lb = unsafe_set_lower_bound.unsqueeze(0).detach().cpu().numpy()
    X_unsafe_ub = unsafe_set_upper_bound.unsqueeze(0).detach().cpu().numpy()
    
    # Matplotlib settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})
    fontsize = 50
    ticksize = 50
    level_fontsize = 35
    legend_fontsize = 40

    pairwise_idx = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    state_labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]
    state_names = ["x", "y", "vx", "vy"]
    mesh_size = 100
    for (x_idx, y_idx) in pairwise_idx:
        x_np = np.linspace(state_lower_bound[x_idx], state_upper_bound[x_idx], mesh_size)
        y_np = np.linspace(state_lower_bound[y_idx], state_upper_bound[y_idx], mesh_size)
        X_np, Y_np = np.meshgrid(x_np, y_np)
        X_flatten_np = X_np.reshape(-1, 1)
        Y_flatten_np = Y_np.reshape(-1, 1)
        state_flatten_np = np.zeros((X_flatten_np.shape[0], len(state_lower_bound)), dtype=np.float32)
        state_flatten_np[:, x_idx] = X_flatten_np[:, 0]
        state_flatten_np[:, y_idx] = Y_flatten_np[:, 0]
        state_torch = torch.tensor(state_flatten_np, dtype=config.pt_dtype)
        dataset = torch.utils.data.TensorDataset(state_torch)
        batch_size = 512
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        feasibility_torch = torch.zeros((state_torch.shape[0],), dtype=config.pt_dtype)
        cbf_torch = torch.zeros((state_torch.shape[0],), dtype=config.pt_dtype)
        for (batch_idx, (state_batch,)) in enumerate(dataloader):
            state_batch = state_batch.to(device)
            feasibility_batch = test_func(state_batch).detach().cpu()
            feasibility_torch[batch_idx * batch_size:min((batch_idx + 1) * batch_size, feasibility_torch.shape[0])] = feasibility_batch
            cbf_batch = cbf_nn(state_batch).squeeze().detach().cpu()
            cbf_torch[batch_idx * batch_size:min((batch_idx + 1) * batch_size, cbf_torch.shape[0])] = cbf_batch
        feasibility_flatten_np = feasibility_torch.detach().cpu().numpy()
        feasibility_np = feasibility_flatten_np.reshape(mesh_size, mesh_size)
        cbf_flatten_np = cbf_torch.detach().cpu().numpy()
        cbf_np = cbf_flatten_np.reshape(mesh_size, mesh_size)

        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111)
        lengend_handles = []

        CS_cbf = ax.contour(X_np, Y_np, cbf_np, 
                            levels=np.linspace(np.min(cbf_np), np.max(cbf_np), 10), 
                            colors='black', linestyles='solid')
        ax.clabel(CS_cbf, inline=True, fontsize=level_fontsize)
        CS_safety_zero = ax.contour(X_np, Y_np, cbf_np, levels=[0], 
                                colors='tab:blue', linewidths=6, linestyles='solid')
        if np.max(cbf_np) > 0:
            ax.contourf(X_np, Y_np, cbf_np, levels=[0, np.max(cbf_np)], colors=['tab:blue'], alpha=0.5)

            safe_set_face_rgba = mcolors.to_rgba('tab:blue', alpha=0.5)
            legend_safe_set = mpatches.Patch(facecolor=safe_set_face_rgba, edgecolor='tab:blue', linewidth=6, label=r'$h_{\theta}(x) \geq 0$')
            lengend_handles.append(legend_safe_set)

        CS_feasibility_zero = ax.contour(X_np, Y_np, feasibility_np, levels=[0], 
                                colors='tab:red', linewidths=6, linestyles='solid')
        if np.min(feasibility_np) < 0:
            ax.contourf(X_np, Y_np, feasibility_np, levels=[np.min(feasibility_np), 0], colors=['tab:red'], alpha=0.5)
            infeasibility_face_rgba = mcolors.to_rgba('tab:red', alpha=0.5)
            legend_infeasibility = mpatches.Patch(facecolor=infeasibility_face_rgba, edgecolor='tab:red', linewidth=6, label=r'$\tilde{H}(x) < 0$')
            lengend_handles.append(legend_infeasibility)
        
        # Plot unsafe region
        for k in range(X_unsafe_lb.shape[0]):
            intersect = True
            x_unsafe_lb = X_unsafe_lb[k]
            x_unsafe_ub = X_unsafe_ub[k]
            for dim in range(X_unsafe_lb.shape[1]):
                if dim != x_idx and dim != y_idx:
                    if not (x_unsafe_lb[dim] <= 0 <= x_unsafe_ub[dim]):
                        intersect = False
                        break
            if intersect:
                rect = plt.Rectangle((x_unsafe_lb[x_idx], x_unsafe_lb[y_idx]), 
                                    x_unsafe_ub[x_idx]-x_unsafe_lb[x_idx], x_unsafe_ub[y_idx]-x_unsafe_lb[y_idx], 
                                    edgecolor=None, facecolor='gray', alpha=0.5)
                ax.add_patch(rect)

        # Set labels and formatting
        ax.set_xlabel(state_labels[x_idx], fontsize=fontsize)
        ax.set_ylabel(state_labels[y_idx], fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=ticksize)

        inadmissible_face_rgba = mcolors.to_rgba('gray', alpha=0.5)
        legend_inadmissible = mpatches.Patch(facecolor=inadmissible_face_rgba, edgecolor=None, label=r'$\mathcal{X}_{\mathrm{in}}$')
        lengend_handles.append(legend_inadmissible)
        ax.legend(handles=lengend_handles,
                  loc='upper right',
                  fontsize=legend_fontsize)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/cond_feasibility_contour_{state_names[x_idx]}_{state_names[y_idx]}_{exp_num:03d}.pdf", dpi=200)
        plt.close()


if __name__ == "__main__":
    exp_nums = list(range(12, 13))
    for exp_num in exp_nums:
        draw_safe_set(exp_num)
