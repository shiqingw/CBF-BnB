import json
import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import time
import multiprocessing
from datetime import datetime
from dreal import Variable, abs, sin, sqrt, Expression, Config, logical_and, logical_not, logical_imply, CheckSatisfiability

from cores.utils.dreal_utils import get_dreal_lipschitz_exp
from cores.dynamical_systems.create_system import get_system
from cores.lip_nn.models import LipschitzNetwork
from cores.utils.utils import seed_everything, format_time, save_dict, load_dict
from cores.utils.config import Configuration

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=1, type=int, help='test case number')
    parser.add_argument('--dreal_precision', default=1e-3, type=float, help='dReal precision')
    args = parser.parse_args()
    dreal_precision = args.dreal_precision

    # Create result directory
    print("==> Creating result directory ...")
    exp_num = args.exp_num
    results_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg1_results_keep/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(results_dir, exp_num)
    if not os.path.exists(test_settings_path):
        test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)
    
    # Load test settings
    print("==> Loading test settings ...")
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Decide torch device
    config = Configuration()
    device = torch.device("cpu")
    print('==> torch device: ', device)

    # Seed everything
    print("==> Seeding everything ...")
    seed = test_settings["seed"]
    seed_everything(seed)

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
    _ = cbf_nn(torch.zeros((1, cbf_in_features), dtype=config.pt_dtype, device=device))

    # The state space
    dataset_config = test_settings["dataset_config"]
    state_lower_bound = torch.tensor(dataset_config["state_lower_bound"], dtype=config.pt_dtype)
    state_upper_bound = torch.tensor(dataset_config["state_upper_bound"], dtype=config.pt_dtype)

    # Control config
    control_config = test_settings["control_config"]
    control_lower_bound = torch.tensor(control_config["lower_bound"], dtype=config.pt_dtype, device=device)
    control_upper_bound = torch.tensor(control_config["upper_bound"], dtype=config.pt_dtype, device=device)
    assert torch.allclose(control_lower_bound + control_upper_bound, torch.zeros_like(control_lower_bound))

    # Disturbance config
    disturbance_config = test_settings["disturbance_config"]
    disturbance_channel_matrix = torch.tensor(disturbance_config["channel_matrix"], dtype=config.pt_dtype, device=device)
    disturbance_lower_bound = torch.tensor(disturbance_config["lower_bound"], dtype=config.pt_dtype, device=device)
    disturbance_upper_bound = torch.tensor(disturbance_config["upper_bound"], dtype=config.pt_dtype, device=device)
    assert torch.allclose(disturbance_lower_bound + disturbance_upper_bound, torch.zeros_like(disturbance_lower_bound))
    disturbance_elementwise_upper_bound = torch.maximum(torch.abs(disturbance_lower_bound), torch.abs(disturbance_upper_bound)) 

    # The state space
    dataset_config = test_settings["dataset_config"]
    state_lower_bound = np.array(dataset_config["state_lower_bound"], dtype=config.np_dtype)
    state_upper_bound = np.array(dataset_config["state_upper_bound"], dtype=config.np_dtype)

    # Control config
    control_config = test_settings["control_config"]
    control_lower_bound = torch.tensor(control_config["lower_bound"], dtype=config.pt_dtype, device=device)
    control_upper_bound = torch.tensor(control_config["upper_bound"], dtype=config.pt_dtype, device=device)

    # Disturbance config
    disturbance_config = test_settings["disturbance_config"]
    disturbance_channel_matrix = torch.tensor(disturbance_config["channel_matrix"], dtype=config.pt_dtype, device=device)
    disturbance_lower_bound = torch.tensor(disturbance_config["lower_bound"], dtype=config.pt_dtype, device=device)
    disturbance_upper_bound = torch.tensor(disturbance_config["upper_bound"], dtype=config.pt_dtype, device=device)
    disturbance_elementwise_upper_bound = torch.maximum(torch.abs(disturbance_lower_bound), torch.abs(disturbance_upper_bound)) 

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

    # Define variables
    x1 = Variable("x1")
    x2 = Variable("x2")
    vars_ = np.array([x1, x2])
    print("==> dReal variables: ", vars_)
    print("==> dReal for CBF function")
    cbf = get_dreal_lipschitz_exp(vars_, cbf_nn, dtype=config.pt_dtype, device=device)

    # System dynamics
    print("==> dReal for stability condition")
    mass = system.mass.detach().cpu().numpy().item()
    length = system.length.detach().cpu().numpy().item()
    viscous_friction = system.viscous_friction.detach().cpu().numpy().item()
    gravity = system.gravity.detach().cpu().numpy().item()
    inertia = mass * length**2
    drift = np.array([x2, gravity/length * sin(x1) - viscous_friction / inertia * x2])
    h_dx = np.array([cbf.Differentiate(x1), cbf.Differentiate(x2)])
    feasibility_condition = np.dot(drift, h_dx)
    actuation = np.array([[0.0], [1.0/inertia]]) # (state_dim, control_dim)
    h_dx_g = np.dot(h_dx, actuation) # (1, control_dim)
    h_dx_g_abs = np.abs(h_dx_g)
    feasibility_condition += np.dot(h_dx_g_abs, control_upper_bound.cpu().numpy().squeeze())
    feasibility_condition += cbf_alpha * cbf
    h_dx_G = np.dot(h_dx, disturbance_channel_matrix.cpu().numpy().squeeze())
    h_dx_G_abs = np.abs(h_dx_G)
    feasibility_condition += np.dot(h_dx_G_abs, disturbance_elementwise_upper_bound.cpu().numpy().squeeze())

    # Test dReal expression
    print("> Checking consistency for feasibility condition ...")
    N = 10
    test_input = 5*(torch.rand((10, cbf_in_features), dtype=config.pt_dtype, device=device)-0.5)
    for i in range(N):
        env = {x1: test_input[i, 0].item(), x2: test_input[i, 1].item()}
        t1 = time.time()
        dreal_value = feasibility_condition.Evaluate(env)
        t2 = time.time()
        pytorch_value = test_func(test_input[i].unsqueeze(0)).detach().cpu().numpy().squeeze()
        if np.linalg.norm(np.array(dreal_value)-pytorch_value) > 1e-5:
            print(f"> Test input {i+1}: {test_input[i]}")
            print(f"> dReal value: {dreal_value} | Time used: {format_time(t2-t1)}")
            print(f"> PyTorch value: {pytorch_value}")
            print(f"> Difference: {np.linalg.norm(np.array(dreal_value)-pytorch_value)}")

    # Solve the stability problem
    print("==> Verifying with dReal ...")
    dreal_config = Config()
    dreal_config.use_polytope_in_forall = True
    dreal_config.use_local_optimization = True
    dreal_config.precision = dreal_precision
    dreal_config.number_of_jobs = min(4, multiprocessing.cpu_count())
    print(f"> dReal precision: {dreal_config.precision:.1E}")
    print(f"> dReal number of jobs: {dreal_config.number_of_jobs}")

    bound = logical_and(x1 >= state_lower_bound[0],
                        x1 <= state_upper_bound[0],
                        x2 >= state_lower_bound[1],
                        x2 <= state_upper_bound[1],
                        cbf >= 0)
    condition = logical_not(logical_imply(bound, feasibility_condition>=0))

    print("> Start checking")
    success = True
    false_positive = False
    start_time = time.time()
    print("> Start time:", datetime.fromtimestamp(start_time))
    result = CheckSatisfiability(condition, dreal_config)
    stop_time = time.time()
    print("> Stop time:", datetime.fromtimestamp(stop_time))
    print("> Time used:", format_time(stop_time-start_time))
    print("> Result:")
    print(result)

    if result:
        CE = []
        for i in range(result.size()):
            CE.append(result[i].mid())
        CE_torch = torch.tensor(CE, dtype=config.pt_dtype, device=device)
        if CE_torch.dim != 2:
            CE_torch = CE_torch.unsqueeze(0)
        print("> Counterexample:", CE_torch)
        print("> Counterexample value:", test_func(CE_torch))
        if torch.any(test_func(CE_torch) > 0):
            print("> Found valid counterexample(s)!")
            success = False
            false_positive = False
        else:
            print("> False positive!")
            success = False
            false_positive = True
    
    print("> Success:", success)
    print("> False positive:", false_positive)

    # Save the result
    checing_result = {
        "success": success,
        "false_positive": false_positive,
        "time": stop_time-start_time,
        "precision": dreal_config.precision,
        "number_of_jobs": dreal_config.number_of_jobs,
    }
    if result:
        checing_result["counter_example"] = CE_torch.cpu().numpy()
        checing_result["counter_example_value"] = test_func(CE_torch).cpu().numpy()

    save_dict(checing_result, f"{results_dir}/00_dreal_feasibility_result_{dreal_config.precision:.1E}.pkl")

    print("==> Done!")