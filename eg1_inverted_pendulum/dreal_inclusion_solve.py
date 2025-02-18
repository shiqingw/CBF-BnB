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
from dreal import Variable, sqrt, Expression, Config, logical_and, logical_not, logical_imply, CheckSatisfiability

from cores.utils.dreal_utils import get_dreal_lipschitz_exp
from cores.lip_nn.models import LipschitzNetwork
from cores.utils.utils import seed_everything, format_time, save_dict
from cores.utils.config import Configuration
from cores.mesh.mesh_tools import decompose_hyperrectangle

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

    # Build CBF network
    print("==> Building CBF neural network ...")
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

    # The safe set
    safe_set_config = test_settings["safe_set_config"]
    safe_set_lower_bound = torch.tensor(safe_set_config["safe_set_lower_bound"], dtype=config.pt_dtype)
    safe_set_upper_bound = torch.tensor(safe_set_config["safe_set_upper_bound"], dtype=config.pt_dtype)

    # Unsafe hyperrectangles
    X_lb, X_ub = decompose_hyperrectangle(state_lower_bound, state_upper_bound, 
                                          safe_set_lower_bound, safe_set_upper_bound)
    n_subregions = X_lb.shape[0]
    print(f"==> Number of subregions: {n_subregions}")

    # Define variables
    x1 = Variable("x1")
    x2 = Variable("x2")
    vars_ = np.array([x1, x2])
    print("==> dReal variables: ", vars_)
    print("==> dReal for CBF function")
    cbf = get_dreal_lipschitz_exp(vars_, cbf_nn, dtype=config.pt_dtype, device=device)
    
    # Solve the positivity problem
    print("==> Verifying with dReal ...")
    dreal_config = Config()
    dreal_config.use_polytope_in_forall = True
    dreal_config.use_local_optimization = True
    dreal_config.precision = dreal_precision
    dreal_config.number_of_jobs = min(1, multiprocessing.cpu_count())
    print(f"> dReal precision: {dreal_config.precision:.1E}")
    print(f"> dReal number of jobs: {dreal_config.number_of_jobs}")

    time_checking = 0.0
    success = True

    for jj in range(n_subregions):
        if success == False:
            break

        subregion_lb, subregion_ub = X_lb[jj], X_ub[jj]
        subregion_lb_np = subregion_lb.detach().cpu().numpy()
        subregion_ub_np = subregion_ub.detach().cpu().numpy()
        print(f"==> Subregion {(jj+1):010d} / {n_subregions:010d}: ")
        print(f"==> [{subregion_lb_np}, {subregion_ub_np}]")


        bound = logical_and(x1 >= subregion_lb_np[0],
                            x1 <= subregion_ub_np[0],
                            x2 >= subregion_lb_np[1],
                            x2 <= subregion_ub_np[1])
        condition = logical_not(logical_imply(bound, cbf<=0))

        false_positive = False
        start_time = time.time()
        print("> Start time:", datetime.fromtimestamp(start_time))
        result = CheckSatisfiability(condition, dreal_config)
        stop_time = time.time()
        print("> Stop time:", datetime.fromtimestamp(stop_time))
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
            print("> Counterexample value:", cbf_nn(CE_torch))
            if torch.any(cbf_nn(CE_torch) < 0):
                print("> Found valid counterexample(s)!")
                success = False
                false_positive = False
            else:
                print("> False positive!")
                success = False
                false_positive = True

        time_checking += stop_time-start_time
        print(f"> Time for subregion {(jj+1):010d}: {format_time(stop_time-start_time)} = {stop_time-start_time} s")
        print(f"> Total time up to subregion {(jj+1):010d}: {format_time(time_checking)} = {time_checking} s")
        print("")
    
    # Save the result
    checing_result = {
        "success": success,
        "false_positive": false_positive,
        "time": time_checking,
        "precision": dreal_config.precision,
        "number_of_jobs": dreal_config.number_of_jobs,
    }

    save_dict(checing_result, f"{results_dir}/00_dreal_inclusion_result_{dreal_config.precision:.1E}.pkl")
    print("==> Success:", success)
    print("==> Done!")

