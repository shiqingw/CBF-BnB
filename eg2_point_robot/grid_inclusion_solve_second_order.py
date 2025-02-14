import json
import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import time

from cores.lip_nn.models import LipschitzNetwork
from cores.utils.utils import format_time, save_dict
from cores.utils.config import Configuration
from cores.mesh.inclusion_adaptive_mesh_second_order import InclusionAdaptiveMeshSecondOrder
from cores.mesh.mesh_tools import decompose_hyperrectangle

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=1, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    parser.add_argument('--precision', default=1e-3, type=float, help='precision')
    args = parser.parse_args()

    # Create result directory
    print("==> Creating result directory ...")
    exp_num = args.exp_num
    results_dir = "{}/eg2_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg2_results_keep/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings_{:03d}.json".format(results_dir, exp_num)
    if not os.path.exists(test_settings_path):
        test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)
    
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
    print('==> torch device: ', device)

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
    lip_cbf_nn = cbf_nn.get_l2_lipschitz_bound()
    hess_bound_cbf_nn = cbf_nn.get_l2_hessian_bound()
    third_order_elementwise_bound_cbf_nn = cbf_nn.get_l2_elementwise_third_order_bound()
    print("==> Lipschitz constant of CBF network: {:.6f}".format(lip_cbf_nn))
    print("==> Hessian bound of CBF network: {:.6f}".format(hess_bound_cbf_nn))
    print("==> Third order elementwise bound of CBF network: {:.6f}".format(third_order_elementwise_bound_cbf_nn))

    # The state space
    dataset_config = test_settings["dataset_config"]
    state_lower_bound = torch.tensor(dataset_config["state_lower_bound"], dtype=config.pt_dtype)
    state_upper_bound = torch.tensor(dataset_config["state_upper_bound"], dtype=config.pt_dtype)

    # The safe set
    unsafe_set_config = test_settings["unsafe_set_config"]
    unsafe_set_lower_bound = torch.tensor(unsafe_set_config["unsafe_set_lower_bound"], dtype=config.pt_dtype)
    unsafe_set_upper_bound = torch.tensor(unsafe_set_config["unsafe_set_upper_bound"], dtype=config.pt_dtype)

    # Unsafe hyperrectangles
    X_lb = unsafe_set_lower_bound.unsqueeze(0)
    X_ub = unsafe_set_upper_bound.unsqueeze(0)
    n_subregions = X_lb.shape[0]
    print(f"==> Number of subregions: {n_subregions}")
    
    # Iterate over all subregions
    success = None
    time_checking = 0.0
    precision = args.precision

    for jj in range(n_subregions):
        if success == False:
            break

        subregion_lb, subregion_ub = X_lb[jj], X_ub[jj]
        print(f"==> Subregion {(jj+1):010d} / {n_subregions:010d}: ")
        print(f"==> [{subregion_lb.detach().cpu().numpy()}, {subregion_ub.detach().cpu().numpy()}]")

        # Adaptive mesh
        mesh = InclusionAdaptiveMeshSecondOrder(f=cbf_nn,
                                    lip_f_l2=lip_cbf_nn,
                                    hess_f_l2=hess_bound_cbf_nn,
                                    thrid_order_elementwise_l2=third_order_elementwise_bound_cbf_nn,
                                    x_lb=subregion_lb.unsqueeze(0),
                                    x_ub=subregion_ub.unsqueeze(0),
                                    batch_size=2**14, #16384
                                    dtype=config.pt_dtype,
                                    device=device)

        start_time = time.time()
        for i in range(50):

            f_lb, f_ub = mesh.bound() # shape (N,) and (N,)
            if (len(f_lb) == 0) or (len(f_ub) == 0):
                success = True
                print("> Mesh returns no regions. Done!")
                break

            f_lb_max = f_lb.max()
            f_ub_max = f_ub.max()
            print(f"> Iteration {i:02d} | Bound: [{f_lb_max.item():.6f}, {f_ub_max.item():.6f}] | l2 radius: {mesh.mesh_radius_l2:.2E} | Size: {len(f_lb):.2E} | Time: {format_time(time.time()-start_time)}")

            if torch.any(f_lb > 0.0):
                not_included_idx = (f_lb > 0.0)
                not_included_regions_lb, not_included_regions_ub = mesh.regions[not_included_idx]
                not_included_region_mid = (not_included_regions_lb + not_included_regions_ub) / 2.0
                not_included_values = f_lb[not_included_idx]
                print("> Not included region: ", not_included_region_mid.detach().cpu().numpy()[0])
                print("> Not included value: ", not_included_values.detach().cpu().numpy()[0])

                # double check
                f_lb_double_check = cbf_nn(not_included_region_mid.to(device)).detach().cpu()
                print("> Double check: ", f_lb_double_check)
                break

            if f_ub_max <= 0.0:
                success = True
                print("> No region to refine. Done!")
                break

            if f_ub_max - f_lb_max <= precision:
                success = True
                print("> Early stopping. Done!")
                break

            refine_idx = (f_ub > 0.0)
            mesh.refine(refine_idx)

        end_time = time.time()
        time_checking += end_time - start_time
        print(f"> Time for subregion {(jj+1):010d}: {format_time(end_time-start_time)}")
        print(f"> Total time up to subregion {(jj+1):010d}: {format_time(time_checking)} = {time_checking} s")

    print(f"> Total time excluding drawing: {format_time(time_checking)} = {time_checking} s")
    print("> Success:", success)
    checking_result = {
        "success": success,
        "time": time_checking,
        "precision": precision
    }
    platform = config.platform
    save_dict(checking_result, f"{results_dir}/00_grid_inclusion_result_{platform}_{device}_{precision:.1E}.pkl")

    print("==> Done!")

