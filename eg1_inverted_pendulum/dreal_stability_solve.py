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
from dreal import Variable, sin, sqrt, Expression, Config, logical_and, logical_not, logical_imply, CheckSatisfiability

from cores.utils.dreal_utils import get_dreal_lyapunov_exp, get_dreal_controller_exp
from cores.dynamical_systems.create_system import get_system
from cores.lip_nn.models import LipschitzNetwork, ControllerNetwork
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

    # Build Lyapunov network
    print("==> Building Lyapunov neural network ...")
    lyapunov_config = test_settings["lyapunov_nn_config"]
    lyapunov_in_features = lyapunov_config["in_features"]
    lyapunov_out_features = lyapunov_config["out_features"]
    lyapunov_gamma = lyapunov_config["lipschitz_constant"]
    lyapunov_activations = [lyapunov_config["activations"]]*(lyapunov_config["num_layers"]-1)
    lyapunov_widths = [lyapunov_in_features]+[lyapunov_config["width_each_layer"]]*(lyapunov_config["num_layers"]-1)+[lyapunov_out_features]
    lyapunov_zero_at_zero = bool(lyapunov_config["zero_at_zero"])
    lyapunov_input_bias = np.array(lyapunov_config["input_bias"], dtype=config.np_dtype)
    lyapunov_input_transform = 1.0/np.array(lyapunov_config["input_transform_to_inverse"], dtype=config.np_dtype)
    lyapunov_nn = LipschitzNetwork(in_features=lyapunov_in_features,
                                   out_features=lyapunov_out_features,
                                   gamma=lyapunov_gamma,
                                   activations=lyapunov_activations,
                                   widths=lyapunov_widths,
                                   zero_at_zero=lyapunov_zero_at_zero,
                                   input_bias=lyapunov_input_bias,
                                   input_transform=lyapunov_input_transform,
                                   dtype=config.pt_dtype,
                                   random_psi=False,
                                   trainable_psi=False)
    lyapunov_nn.load_state_dict(torch.load(f"{results_dir}/lyapunov_weights_best_forward_invariant.pt", weights_only=True, map_location=device))
    lyapunov_nn.to(device)
    lyapunov_nn.eval()
    _ = lyapunov_nn(torch.zeros((1, lyapunov_in_features), dtype=config.pt_dtype, device=device))

    # Build controller network
    print("==> Building controller neural network ...")
    controller_config = test_settings["controller_nn_config"]
    controller_in_features = controller_config["in_features"]
    controller_out_features = controller_config["out_features"]
    controller_gamma = controller_config["lipschitz_constant"]
    controller_activations = [controller_config["activations"]]*(controller_config["num_layers"]-1)
    controller_widths = [controller_in_features]+[controller_config["width_each_layer"]]*(controller_config["num_layers"]-1)+[controller_out_features]
    controller_zero_at_zero = bool(controller_config["zero_at_zero"])
    controller_input_bias = np.array(controller_config["input_bias"], dtype=config.np_dtype)
    controller_input_transform = 1.0/np.array(controller_config["input_transform_to_inverse"], dtype=config.np_dtype)
    controller_lower_bound = controller_config["lower_bound"]
    controller_upper_bound = controller_config["upper_bound"]
    controller_nn = ControllerNetwork(in_features=controller_in_features, 
                                    out_features=controller_out_features,
                                    gamma=controller_gamma,
                                    activations=controller_activations,
                                    widths=controller_widths,
                                    zero_at_zero=controller_zero_at_zero,
                                    input_bias=controller_input_bias,
                                    input_transform=controller_input_transform,
                                    lower_bound=controller_lower_bound,
                                    upper_bound=controller_upper_bound,
                                    dtype=config.pt_dtype,
                                    random_psi=False,
                                    trainable_psi=True)
    controller_nn.load_state_dict(torch.load(f"{results_dir}/controller_weights_best_forward_invariant.pt", weights_only=True, map_location=device))
    controller_nn.to(device)
    controller_nn.eval()
    _ = controller_nn(torch.zeros((1, controller_in_features), dtype=config.pt_dtype, device=device))

    # The state space
    dataset_config = test_settings["dataset_config"]
    state_lower_bound = np.array(dataset_config["state_lower_bound"], dtype=config.np_dtype)
    state_upper_bound = np.array(dataset_config["state_upper_bound"], dtype=config.np_dtype)

    # Stability config
    stability_config = test_settings["stability_config"]
    stability_cutoff_radius = stability_config["cutoff_radius"]
    beta = stability_config["beta"]
    disturbance_channel = stability_config["disturbance_channel"]
    disturbance_channel = torch.tensor(disturbance_channel, dtype=config.pt_dtype).to(device)
    d0 = stability_config["d0"]
    d1 = stability_config["d1"]
    d1_state = stability_config["d1_state"]
    d1_state = torch.tensor(d1_state, dtype=config.pt_dtype).to(device)

    # Load forward invariant information
    roa_info = load_dict(f"{results_dir}/forward_invariant_results.pkl")
    sublevel_value = roa_info["lyapunov_value"]
    print("==> Forward invariant sublevel value: {:.6f}".format(sublevel_value))

    # The stability condition
    def test_func(x):
        u = controller_nn(x)
        dx = system(x, u)
        _, V_dx = lyapunov_nn.forward_with_jacobian(x)
        norm_x = torch.linalg.norm(x, ord=2, dim=1).unsqueeze(-1) # (batch_size,1)
        dV = torch.bmm(V_dx, dx.unsqueeze(-1)).squeeze(-1) # (batch_size,1)
        V_dx_G = torch.matmul(V_dx, disturbance_channel.to(device)).squeeze(-2) # (batch_size,disturbance_dim)
        norm_V_dx_G = torch.linalg.norm(V_dx_G, ord=2, dim=1).unsqueeze(-1) # (batch_size,1)
        d1_norm = torch.linalg.norm(torch.matmul(x, d1_state.to(device).T), ord=2, dim=1).unsqueeze(-1) # (batch_size,1)
        stability_condition = dV + norm_V_dx_G * (d0 + d1*d1_norm) + beta*norm_x
        return stability_condition

    # Define variables
    x1 = Variable("x1")
    x2 = Variable("x2")
    vars_ = np.array([x1, x2])
    print("==> dReal variables: ", vars_)
    print("==> dReal for lyapunov function")
    V = get_dreal_lyapunov_exp(vars_, lyapunov_nn, dtype=config.pt_dtype, device=device)
    print("==> dReal for controller")
    control = get_dreal_controller_exp(vars_, controller_nn, dtype=config.pt_dtype, device=device)

    # System dynamics
    print("==> dReal for stability condition")
    mass = system.mass.detach().cpu().numpy().item()
    length = system.length.detach().cpu().numpy().item()
    viscous_friction = system.viscous_friction.detach().cpu().numpy().item()
    gravity = system.gravity.detach().cpu().numpy().item()
    inertia = mass * length**2
    x2_dot = gravity/length * sin(x1) + control / inertia - viscous_friction / inertia * x2
    state_derivative = np.array([x2, x2_dot])

    grad_V = np.array([V.Differentiate(x1), V.Differentiate(x2)])
    stability_condition = np.dot(state_derivative, grad_V)
    norm_grad_V_disturbance_channel = sqrt(np.sum(np.power(grad_V*disturbance_channel.cpu().numpy().squeeze(), 2)))
    norm_x = sqrt(x1*x1 + x2*x2)
    d1_norm = np.dot(vars_, d1_state.cpu().numpy().squeeze())
    d1_norm = sqrt(np.sum(np.power(d1_norm, 2)))
    stability_condition += (d0 + d1*d1_norm)*norm_grad_V_disturbance_channel
    stability_condition += beta*norm_x

    # Test dReal expression
    print("> Checking consistency for stability condition ...")
    N = 10
    test_input = 5*(torch.rand((10, lyapunov_in_features), dtype=config.pt_dtype, device=device)-0.5)
    for i in range(N):
        env = {x1: test_input[i, 0].item(), x2: test_input[i, 1].item()}
        t1 = time.time()
        dreal_value = stability_condition.Evaluate(env)
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

    bound = logical_and(sqrt(x1*x1 + x2*x2)>= stability_cutoff_radius,
                        x1 >= state_lower_bound[0],
                        x1 <= state_upper_bound[0],
                        x2 >= state_lower_bound[1],
                        x2 <= state_upper_bound[1],
                        V <= sublevel_value)
    condition = logical_not(logical_imply(bound, stability_condition<=0))

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
        checing_result["counter_example_value"] = lyapunov_nn(CE_torch).cpu().numpy()

    save_dict(checing_result, f"{results_dir}/00_dreal_stability_result_{dreal_config.precision:.1E}.pkl")

    print("==> Done!")