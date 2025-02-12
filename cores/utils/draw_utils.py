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

def draw_2d_lyapunov_contour(lyapunov_nn, state_lower_bound, state_upper_bound, mesh_size, savepath, dpi, device, pt_dtype):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})
    x_np = np.linspace(state_lower_bound[0], state_upper_bound[0], mesh_size)
    y_np = np.linspace(state_lower_bound[1], state_upper_bound[1], mesh_size)
    X_np, Y_np = np.meshgrid(x_np, y_np)
    X_flatten_np = X_np.reshape(-1, 1)
    Y_flatten_np = Y_np.reshape(-1, 1)
    XY_flatten_np = np.concatenate([X_flatten_np, Y_flatten_np], axis=1)
    XY_torch = torch.tensor(XY_flatten_np, dtype=pt_dtype)
    dataset = torch.utils.data.TensorDataset(XY_torch)
    batch_size = 1000
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    V_torch = torch.zeros((XY_torch.shape[0], 1), dtype=pt_dtype)
    for (batch_idx, (XY_batch,)) in enumerate(dataloader):
        XY_batch = XY_batch.to(device)
        V_batch = lyapunov_nn(XY_batch).detach().cpu()
        V_torch[batch_idx * batch_size:min((batch_idx + 1) * batch_size, V_torch.shape[0])] = V_batch
    V_flatten_np = V_torch.detach().cpu().numpy()
    V_np = V_flatten_np.reshape(mesh_size, mesh_size)
    V_boundary_np = np.concatenate([V_np[0, :], V_np[-1, :], V_np[:, 0], V_np[:, -1]], axis=0)
    V_min_np = np.min(V_boundary_np)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fontsize = 20
    ticksize = 15
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    ax.contourf(X_np, Y_np, V_np, levels=100, cmap='viridis')
    CS = ax.contour(X_np, Y_np, V_np, levels=np.linspace(0, np.max(V_np), 10), colors='black')
    ax.clabel(CS, inline=True, fontsize=10)  # Add labels to contour lines
    CS = ax.contour(X_np, Y_np, V_np, levels=[0], colors='tab:red')
    ax.clabel(CS, inline=True, fontsize=10)
    CS_min = ax.contour(X_np, Y_np, V_np, levels=[V_min_np], colors='tab:orange')
    ax.clabel(CS_min, inline=True, fontsize=10)  # Add label to V_min_np contour line
    ax.set_xlabel(r"$x_1$", fontsize=fontsize)
    ax.set_ylabel(r"$x_2$", fontsize=fontsize)
    ax.set_title("Lyapunov Function", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.close()

def draw_lyapunov_contour(lyapunov_nn, state_lower_bound, state_upper_bound, mesh_size, x_state_idx, y_state_idx, x_label, y_label, savepath, dpi, device, pt_dtype):
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
    batch_size = 1000
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    V_torch = torch.zeros((state_torch.shape[0], 1), dtype=pt_dtype)
    for (batch_idx, (state_batch,)) in enumerate(dataloader):
        state_batch = state_batch.to(device)
        V_batch = lyapunov_nn(state_batch).detach().cpu()
        V_torch[batch_idx * batch_size:min((batch_idx + 1) * batch_size, V_torch.shape[0])] = V_batch
    V_flatten_np = V_torch.detach().cpu().numpy()
    V_np = V_flatten_np.reshape(mesh_size, mesh_size)
    V_boundary_np = np.concatenate([V_np[0, :], V_np[-1, :], V_np[:, 0], V_np[:, -1]], axis=0)
    V_min_np = np.min(V_boundary_np)

    fontsize = 50
    ticksize = 25
    level_fontsize = 35
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    ax.contourf(X_np, Y_np, V_np, levels=100, cmap='viridis')
    CS_all = ax.contour(X_np, Y_np, V_np, levels=np.linspace(0, np.max(V_np), 10), colors='black')
    ax.clabel(CS_all, inline=True, fontsize=level_fontsize)  # Add labels to contour lines
    CS_zero = ax.contour(X_np, Y_np, V_np, levels=[0], colors='tab:red')
    ax.clabel(CS_zero, inline=True, fontsize=level_fontsize)
    CS_min = ax.contour(X_np, Y_np, V_np, levels=[V_min_np], colors='tab:orange')
    ax.clabel(CS_min, inline=True, fontsize=level_fontsize)  # Add label to V_min_np contour line
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
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

def draw_positive_condition_contour(lyapunov_nn, state_lower_bound, state_upper_bound, mesh_size, x_state_idx, y_state_idx, x_label, y_label, positive_cutoff_radius,
                                    stability_cutoff_radius, particular_level, savepath, dpi, device, pt_dtype):
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
    V_torch = torch.zeros((state_torch.shape[0], 1), dtype=pt_dtype)
    for (batch_idx, (state_batch,)) in enumerate(dataloader):
        state_batch = state_batch.to(device)
        V_batch = lyapunov_nn(state_batch).detach().cpu()
        V_torch[batch_idx * batch_size:min((batch_idx + 1) * batch_size, V_torch.shape[0])] = V_batch
    V_flatten_np = V_torch.detach().cpu().numpy()
    V_np = V_flatten_np.reshape(mesh_size, mesh_size)
    V_boundary_np = np.concatenate([V_np[0, :], V_np[-1, :], V_np[:, 0], V_np[:, -1]], axis=0)
    V_min_np = np.min(V_boundary_np)

    fontsize = 50
    ticksize = 25
    level_fontsize = 35
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    ax.contourf(X_np, Y_np, V_np, levels=100, cmap='viridis')
    CS_all = ax.contour(X_np, Y_np, V_np, levels=np.linspace(0, np.max(V_np), 10), colors='black')
    ax.clabel(CS_all, inline=True, fontsize=level_fontsize)  # Add labels to contour lines
    CS_zero = ax.contour(X_np, Y_np, V_np, levels=[0], colors='tab:red')
    ax.clabel(CS_zero, inline=True, fontsize=level_fontsize)
    CS_min = ax.contour(X_np, Y_np, V_np, levels=[V_min_np], colors='tab:orange')
    ax.clabel(CS_min, inline=True, fontsize=level_fontsize)  # Add label to V_min_np contour line
    if particular_level is not None:
        CS_particular = ax.contour(X_np, Y_np, V_np, levels=[particular_level], colors='yellow')
        ax.clabel(CS_particular, inline=True, fontsize=level_fontsize)
    postive_circle = plt.Circle((0, 0), positive_cutoff_radius, color='tab:red', fill=False)
    ax.add_artist(postive_circle)
    stability_circle = plt.Circle((0, 0), stability_cutoff_radius, color='tab:orange', fill=False)
    ax.add_artist(stability_circle)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.close()

def draw_stability_condition_contour(model, state_lower_bound, state_upper_bound, mesh_size, x_state_idx, y_state_idx, x_label, y_label, stability_cutoff_radius,
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
    stability_torch = torch.zeros((state_torch.shape[0], 1), dtype=pt_dtype)
    for (batch_idx, (state_batch,)) in enumerate(dataloader):
        state_batch = state_batch.to(device)
        stability_batch = model(state_batch).detach().cpu()
        stability_torch[batch_idx * batch_size:min((batch_idx + 1) * batch_size, stability_torch.shape[0])] = stability_batch
    stability_flatten_np = stability_torch.detach().cpu().numpy()
    stability_np = stability_flatten_np.reshape(mesh_size, mesh_size)

    fontsize = 50
    ticksize = 25
    level_fontsize = 35
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    ax.contourf(X_np, Y_np, stability_np, levels=100, cmap='viridis')
    CS_all = ax.contour(X_np, Y_np, stability_np, levels=np.linspace(np.min(stability_np), 0, 10), colors='black')
    ax.clabel(CS_all, inline=True, fontsize=level_fontsize)  # Add labels to contour lines
    CS_zero = ax.contour(X_np, Y_np, stability_np, levels=[0], colors='tab:red')
    ax.clabel(CS_zero, inline=True, fontsize=level_fontsize)
    circle = plt.Circle((0, 0), stability_cutoff_radius, color='tab:red', fill=False)
    ax.add_artist(circle)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
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
    feasibility_torch = torch.zeros((state_torch.shape[0], 1), dtype=pt_dtype)
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