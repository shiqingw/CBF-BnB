import matplotlib.pyplot as plt
import numpy as np
import torch

def draw_curve(data, ylabel, savepath, dpi):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi, frameon=True)
    ax.plot(np.arange(len(data)), data, linewidth=1)
    ax.set_xlabel("epochs", fontsize=20)
    ax.set_ylabel(ylabel.lower(), fontsize=20)
    ax.set_title(ylabel, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20, grid_linewidth=10)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.close()

def draw_multiple_curves(data_list, label_list, savepath, dpi):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi, frameon=True)
    for data, label in zip(data_list, label_list):
        ax.plot(np.arange(len(data)), data, linewidth=1, label=label)
    ax.set_xlabel("epochs", fontsize=20)
    ax.legend(fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20, grid_linewidth=10)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.close()

def draw_2d_scatter(train_data, test_data, xlabel, ylabel, title, savepath, dpi):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi, frameon=True)
    if train_data is not None:
        ax.scatter(train_data[:, 0], train_data[:, 1], s=1, c='tab:blue', label='train')
    if test_data is not None:
        ax.scatter(test_data[:, 0], test_data[:, 1], s=1, c='tab:orange', label='test')
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20, grid_linewidth=10)
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.close()

def draw_state_space(state_np, x_state_idx, y_state_idx, x_label, y_label, positive_cutoff_radius, stability_cutoff_radius, savepath, dpi):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    fontsize = 50
    ticksize = 25
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100, frameon=True)
    ax.scatter(state_np[:, x_state_idx], state_np[:, y_state_idx], s=1, c='tab:blue')
    circle_positive = plt.Circle((0, 0), positive_cutoff_radius, color='tab:orange', fill=False)
    ax.add_artist(circle_positive)
    circle_stability = plt.Circle((0, 0), stability_cutoff_radius, color='tab:red', fill=False)
    ax.add_artist(circle_stability)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.close()

def draw_safe_set_contour(cbf_nn, state_lower_bound, state_upper_bound, mesh_size, x_state_idx, y_state_idx, x_label, y_label, is_safe_func,
                            particular_level, savepath, dpi, device, pt_dtype):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    x_np = np.linspace(state_lower_bound[x_state_idx], state_upper_bound[x_state_idx], mesh_size)
    y_np = np.linspace(state_lower_bound[y_state_idx], state_upper_bound[y_state_idx], mesh_size)
    X_np, Y_np = np.meshgrid(x_np, y_np)
    X_flatten_np = X_np.reshape(-1, 1)
    Y_flatten_np = Y_np.reshape(-1, 1)
    state_flatten_np = np.zeros((X_flatten_np.shape[0], len(state_lower_bound)), dtype=np.float32)
    state_flatten_np[:, x_state_idx] = X_flatten_np[:, 0]
    state_flatten_np[:, y_state_idx] = Y_flatten_np[:, 0]
    is_safe_func_np = is_safe_func(state_flatten_np)
    state_torch = torch.tensor(state_flatten_np, dtype=pt_dtype)
    dataset = torch.utils.data.TensorDataset(state_torch)
    batch_size = 512
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    h_torch = torch.zeros((state_torch.shape[0], 1), dtype=pt_dtype)
    for (batch_idx, (state_batch,)) in enumerate(dataloader):
        state_batch = state_batch.to(device)
        h_batch = cbf_nn(state_batch).detach().cpu()
        h_torch[batch_idx * batch_size:min((batch_idx + 1) * batch_size, h_torch.shape[0])] = h_batch
    h_flatten_np = h_torch.detach().cpu().numpy()
    h_np = h_flatten_np.reshape(mesh_size, mesh_size)

    fontsize = 50
    ticksize = 25
    level_fontsize = 35
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    # ax.contourf(X_np, Y_np, h_np, levels=100, cmap='viridis')
    CS_all = ax.contour(X_np, Y_np, h_np, levels=np.linspace(np.min(h_np), np.max(h_np), 10), colors='black')
    ax.clabel(CS_all, inline=True, fontsize=level_fontsize)  # Add labels to contour lines
    CS_zero = ax.contour(X_np, Y_np, h_np, levels=[0], colors='tab:red')
    ax.clabel(CS_zero, inline=True, fontsize=level_fontsize)
    if particular_level is not None:
        CS_particular = ax.contour(X_np, Y_np, h_np, levels=[particular_level], colors='yellow')
        ax.clabel(CS_particular, inline=True, fontsize=level_fontsize)
    
    # Draw the safe set
    safe_idx = is_safe_func_np >= 0
    ax.scatter(X_flatten_np[safe_idx], Y_flatten_np[safe_idx], s=1, c='tab:blue', label='safe set')
    unsafe_idx = is_safe_func_np < 0
    ax.scatter(X_flatten_np[unsafe_idx], Y_flatten_np[unsafe_idx], s=1, c='tab:orange', label='unsafe set')

    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.close()

def draw_feasibility_condition_contour(model, state_lower_bound, state_upper_bound, mesh_size, x_state_idx, y_state_idx, x_label, y_label, is_safe_func,
                                    savepath, dpi, device, pt_dtype):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    x_np = np.linspace(state_lower_bound[x_state_idx], state_upper_bound[x_state_idx], mesh_size)
    y_np = np.linspace(state_lower_bound[y_state_idx], state_upper_bound[y_state_idx], mesh_size)
    X_np, Y_np = np.meshgrid(x_np, y_np)
    X_flatten_np = X_np.reshape(-1, 1)
    Y_flatten_np = Y_np.reshape(-1, 1)
    state_flatten_np = np.zeros((X_flatten_np.shape[0], len(state_lower_bound)), dtype=np.float32)
    state_flatten_np[:, x_state_idx] = X_flatten_np[:, 0]
    state_flatten_np[:, y_state_idx] = Y_flatten_np[:, 0]
    is_safe_func_np = is_safe_func(state_flatten_np)
    state_torch = torch.tensor(state_flatten_np, dtype=pt_dtype)
    dataset = torch.utils.data.TensorDataset(state_torch)
    batch_size = 512
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    feasibility_torch = torch.zeros((state_torch.shape[0],), dtype=pt_dtype)
    for (batch_idx, (state_batch,)) in enumerate(dataloader):
        state_batch = state_batch.to(device)
        feasibility_batch = model(state_batch).detach().cpu()
        feasibility_torch[batch_idx * batch_size:min((batch_idx + 1) * batch_size, feasibility_torch.shape[0])] = feasibility_batch
    feasibility_flatten_np = feasibility_torch.detach().cpu().numpy()
    feasibility_np = feasibility_flatten_np.reshape(mesh_size, mesh_size)

    fontsize = 50
    ticksize = 25
    level_fontsize = 35
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    # ax.contourf(X_np, Y_np, feasibility_np, levels=100, cmap='viridis')
    CS_all = ax.contour(X_np, Y_np, feasibility_np, levels=np.linspace(np.min(feasibility_np), np.max(feasibility_np), 10), colors='black')
    ax.clabel(CS_all, inline=True, fontsize=level_fontsize)  # Add labels to contour lines
    CS_zero = ax.contour(X_np, Y_np, feasibility_np, levels=[0], colors='tab:red')
    ax.clabel(CS_zero, inline=True, fontsize=level_fontsize)

    # Draw the safe set
    safe_idx = is_safe_func_np >= 0
    ax.scatter(X_flatten_np[safe_idx], Y_flatten_np[safe_idx], s=1, c='tab:blue', label='safe set')
    unsafe_idx = is_safe_func_np < 0
    ax.scatter(X_flatten_np[unsafe_idx], Y_flatten_np[unsafe_idx], s=1, c='tab:orange', label='unsafe set')
    
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.close()

def draw_safe_set_contour_given_labels(cbf_nn, state_lower_bound, state_upper_bound, mesh_size, x_state_idx, y_state_idx, x_label, y_label, states_np, labels_np,
                            particular_level, savepath, dpi, device, pt_dtype):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    x_np = np.linspace(state_lower_bound[x_state_idx], state_upper_bound[x_state_idx], mesh_size)
    y_np = np.linspace(state_lower_bound[y_state_idx], state_upper_bound[y_state_idx], mesh_size)
    X_np, Y_np = np.meshgrid(x_np, y_np)
    X_flatten_np = X_np.reshape(-1, 1)
    Y_flatten_np = Y_np.reshape(-1, 1)
    state_flatten_np = np.zeros((X_flatten_np.shape[0], len(state_lower_bound)), dtype=np.float32)
    state_flatten_np[:, x_state_idx] = X_flatten_np[:, 0]
    state_flatten_np[:, y_state_idx] = Y_flatten_np[:, 0]
   
    state_torch = torch.tensor(state_flatten_np, dtype=pt_dtype)
    dataset = torch.utils.data.TensorDataset(state_torch)
    batch_size = 512
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    h_torch = torch.zeros((state_torch.shape[0], 1), dtype=pt_dtype)
    for (batch_idx, (state_batch,)) in enumerate(dataloader):
        state_batch = state_batch.to(device)
        h_batch = cbf_nn(state_batch).detach().cpu()
        h_torch[batch_idx * batch_size:min((batch_idx + 1) * batch_size, h_torch.shape[0])] = h_batch
    h_flatten_np = h_torch.detach().cpu().numpy()
    h_np = h_flatten_np.reshape(mesh_size, mesh_size)

    fontsize = 50
    ticksize = 25
    level_fontsize = 35
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    ax.contourf(X_np, Y_np, h_np, levels=[0, np.max(h_np)], colors=['tab:blue'], alpha=0.5)
    CS_all = ax.contour(X_np, Y_np, h_np, levels=np.linspace(np.min(h_np), np.max(h_np), 10), colors='black')
    ax.clabel(CS_all, inline=True, fontsize=level_fontsize)  # Add labels to contour lines
    CS_zero = ax.contour(X_np, Y_np, h_np, levels=[0], colors='tab:blue', linewidths=3)
    ax.clabel(CS_zero, inline=True, fontsize=level_fontsize)
    if particular_level is not None:
        CS_particular = ax.contour(X_np, Y_np, h_np, levels=[particular_level], colors='yellow')
        ax.clabel(CS_particular, inline=True, fontsize=level_fontsize)
    
    # Draw the safe set
    # filter states_np based on the distance to the 2d subspace
    projection = np.zeros_like(states_np)
    projection[:, x_state_idx] = states_np[:, x_state_idx]
    projection[:, y_state_idx] = states_np[:, y_state_idx]
    distance = np.linalg.norm(states_np - projection, axis=1)
    safe_to_draw = (distance < 0.1) & (labels_np == 1)
    unsafe_to_draw = (distance < 0.1) & (labels_np == -1)
    ax.scatter(states_np[safe_to_draw, x_state_idx], states_np[safe_to_draw, y_state_idx], s=5, c='tab:blue', label='safe set')
    ax.scatter(states_np[unsafe_to_draw, x_state_idx], states_np[unsafe_to_draw, y_state_idx], s=5, c='tab:orange', label='unsafe set')

    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.close()

def draw_feasibility_condition_contour_given_labels(model, state_lower_bound, state_upper_bound, mesh_size, x_state_idx, y_state_idx, x_label, y_label, states_np, labels_np,
                                    savepath, dpi, device, pt_dtype):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    x_np = np.linspace(state_lower_bound[x_state_idx], state_upper_bound[x_state_idx], mesh_size)
    y_np = np.linspace(state_lower_bound[y_state_idx], state_upper_bound[y_state_idx], mesh_size)
    X_np, Y_np = np.meshgrid(x_np, y_np)
    X_flatten_np = X_np.reshape(-1, 1)
    Y_flatten_np = Y_np.reshape(-1, 1)
    state_flatten_np = np.zeros((X_flatten_np.shape[0], len(state_lower_bound)), dtype=np.float32)
    state_flatten_np[:, x_state_idx] = X_flatten_np[:, 0]
    state_flatten_np[:, y_state_idx] = Y_flatten_np[:, 0]
    
    state_torch = torch.tensor(state_flatten_np, dtype=pt_dtype)
    dataset = torch.utils.data.TensorDataset(state_torch)
    batch_size = 512
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    feasibility_torch = torch.zeros((state_torch.shape[0],), dtype=pt_dtype)
    for (batch_idx, (state_batch,)) in enumerate(dataloader):
        state_batch = state_batch.to(device)
        feasibility_batch = model(state_batch).detach().cpu()
        feasibility_torch[batch_idx * batch_size:min((batch_idx + 1) * batch_size, feasibility_torch.shape[0])] = feasibility_batch
    feasibility_flatten_np = feasibility_torch.detach().cpu().numpy()
    feasibility_np = feasibility_flatten_np.reshape(mesh_size, mesh_size)

    fontsize = 50
    ticksize = 25
    level_fontsize = 35
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    ax.contourf(X_np, Y_np, feasibility_np, levels=[np.min(feasibility_np), 0], colors=['tab:red'], alpha=0.5)
    CS_all = ax.contour(X_np, Y_np, feasibility_np, levels=np.linspace(np.min(feasibility_np), np.max(feasibility_np), 10), colors='black')
    ax.clabel(CS_all, inline=True, fontsize=level_fontsize)  # Add labels to contour lines
    CS_zero = ax.contour(X_np, Y_np, feasibility_np, levels=[0], colors='tab:red', linewidths=3)
    ax.clabel(CS_zero, inline=True, fontsize=level_fontsize)

    # Draw the safe set
    # filter states_np based on the distance to the 2d subspace
    projection = np.zeros_like(states_np)
    projection[:, x_state_idx] = states_np[:, x_state_idx]
    projection[:, y_state_idx] = states_np[:, y_state_idx]
    distance = np.linalg.norm(states_np - projection, axis=1)
    safe_to_draw = (distance < 0.1) & (labels_np == 1)
    unsafe_to_draw = (distance < 0.1) & (labels_np == -1)
    ax.scatter(states_np[safe_to_draw, x_state_idx], states_np[safe_to_draw, y_state_idx], s=5, c='tab:blue', label='safe set')
    ax.scatter(states_np[unsafe_to_draw, x_state_idx], states_np[unsafe_to_draw, y_state_idx], s=5, c='tab:orange', label='unsafe set')

    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.close()

def draw_safe_set_and_feasibility_contour_given_labels(cbf_nn, test_func, state_lower_bound, state_upper_bound, mesh_size, x_state_idx, y_state_idx, x_label, y_label, states_np, labels_np,
                            savepath, dpi, device, pt_dtype):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    x_np = np.linspace(state_lower_bound[x_state_idx], state_upper_bound[x_state_idx], mesh_size)
    y_np = np.linspace(state_lower_bound[y_state_idx], state_upper_bound[y_state_idx], mesh_size)
    X_np, Y_np = np.meshgrid(x_np, y_np)
    X_flatten_np = X_np.reshape(-1, 1)
    Y_flatten_np = Y_np.reshape(-1, 1)
    state_flatten_np = np.zeros((X_flatten_np.shape[0], len(state_lower_bound)), dtype=np.float32)
    state_flatten_np[:, x_state_idx] = X_flatten_np[:, 0]
    state_flatten_np[:, y_state_idx] = Y_flatten_np[:, 0]
   
    state_torch = torch.tensor(state_flatten_np, dtype=pt_dtype)
    dataset = torch.utils.data.TensorDataset(state_torch)
    batch_size = 512
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    h_torch = torch.zeros((state_torch.shape[0], 1), dtype=pt_dtype)
    feasibility_torch = torch.zeros((state_torch.shape[0],), dtype=pt_dtype)
    for (batch_idx, (state_batch,)) in enumerate(dataloader):
        state_batch = state_batch.to(device)
        h_batch = cbf_nn(state_batch).detach().cpu()
        h_torch[batch_idx * batch_size:min((batch_idx + 1) * batch_size, h_torch.shape[0])] = h_batch
        feasibility_batch = test_func(state_batch).detach().cpu()
        feasibility_torch[batch_idx * batch_size:min((batch_idx + 1) * batch_size, feasibility_torch.shape[0])] = feasibility_batch
    h_flatten_np = h_torch.detach().cpu().numpy()
    h_np = h_flatten_np.reshape(mesh_size, mesh_size)
    feasibility_flatten_np = feasibility_torch.detach().cpu().numpy()
    feasibility_np = feasibility_flatten_np.reshape(mesh_size, mesh_size)

    fontsize = 50
    ticksize = 25
    level_fontsize = 35
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    ax.contourf(X_np, Y_np, h_np, levels=[0, np.max(h_np)], colors=['tab:blue'], alpha=0.5)
    CS_all = ax.contour(X_np, Y_np, h_np, levels=np.linspace(np.min(h_np), np.max(h_np), 10), colors='black')
    ax.clabel(CS_all, inline=True, fontsize=level_fontsize)  # Add labels to contour lines
    CS_zero = ax.contour(X_np, Y_np, h_np, levels=[0], colors='tab:blue', linewidths=3)
    ax.clabel(CS_zero, inline=True, fontsize=level_fontsize)
    
    ax.contourf(X_np, Y_np, feasibility_np, levels=[np.min(feasibility_np), 0], colors=['tab:red'], alpha=0.5)
    CS_zero = ax.contour(X_np, Y_np, feasibility_np, levels=[0], colors='tab:red', linewidths=3)

    # Draw the safe set
    # filter states_np based on the distance to the 2d subspace
    projection = np.zeros_like(states_np)
    projection[:, x_state_idx] = states_np[:, x_state_idx]
    projection[:, y_state_idx] = states_np[:, y_state_idx]
    distance = np.linalg.norm(states_np - projection, axis=1)
    safe_to_draw = (distance < 0.1) & (labels_np == 1)
    unsafe_to_draw = (distance < 0.1) & (labels_np == -1)
    ax.scatter(states_np[safe_to_draw, x_state_idx], states_np[safe_to_draw, y_state_idx], s=5, c='tab:blue', label='safe set')
    ax.scatter(states_np[unsafe_to_draw, x_state_idx], states_np[unsafe_to_draw, y_state_idx], s=5, c='tab:orange', label='unsafe set')

    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.close()