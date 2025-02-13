import json
import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import time

from cores.dynamical_systems.create_system import get_system
from cores.lip_nn.models import LipschitzNetwork, ControllerNetwork
from cores.utils.utils import format_time, load_dict, save_dict
from cores.utils.config import Configuration
from cores.mesh.feasibility_adaptive_mesh import FeasibilityAdaptiveMesh
from cores.mesh.mesh_tools import split_nd_rectangle_at_index

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=1, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    parser.add_argument('--precision', default=1e-3, type=float, help='precision')
    parser.add_argument('--draw', action='store_true', help='draw the mesh')
    args = parser.parse_args()

    # Create result directory
    print("==> Creating result directory ...")
    exp_num = args.exp_num
    results_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg1_results_keep/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
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

    # Adaptive mesh
    n_pieces_each_dim = torch.tensor([1, 1], dtype=torch.long)
    n_subregions = n_pieces_each_dim.prod().item()
    print(f"==> Number of subregions: {n_subregions}")
    
    # Iterate over all subregions
    success = None
    time_checking = 0.0
    precision = args.precision

    for jj in range(n_subregions):
        if success == False:
            break
        
        subregion_lb, subregion_ub = split_nd_rectangle_at_index(state_lower_bound, state_upper_bound, n_pieces_each_dim, jj, config.pt_dtype)
        print(f"==> Subregion {(jj+1):010d} / {n_subregions:010d}: ")
        print(f"==> [{subregion_lb.detach().cpu().numpy()}, {subregion_ub.detach().cpu().numpy()}]")

        # Calulate the lipchitz bound of the test function
        subregion_lb_np = subregion_lb.cpu().numpy()
        subregion_ub_np = subregion_ub.cpu().numpy()
        subregion_l2_bound = np.linalg.norm(np.maximum(np.abs(subregion_lb_np), np.abs(subregion_ub_np)), ord=2)
        subregion_l2_radius = np.linalg.norm(subregion_ub_np - subregion_lb_np, ord=2)/2.0
        u_lb = control_lower_bound.cpu().numpy()
        u_ub = control_upper_bound.cpu().numpy()
        F_l1_bound = system.get_f_l1_bound(x_lb=subregion_lb_np, x_ub=subregion_ub_np, u_lb=u_lb, u_ub=u_ub)
        F_l2_bound = system.get_f_l2_bound(x_lb=subregion_lb_np, x_ub=subregion_ub_np, u_lb=u_lb, u_ub=u_ub)
        F_dx_l2_bound = system.get_f_dx_l2_bound(x_lb=subregion_lb_np, x_ub=subregion_ub_np, u_lb=u_lb, u_ub=u_ub)
        F_dxdx_elementwise_l2_bound = system.get_f_dxdx_elementwise_l2_bound(x_lb=subregion_lb_np, x_ub=subregion_ub_np, u_lb=u_lb, u_ub=u_ub)
        norm_disturbance_channel_l2 = np.linalg.norm(disturbance_channel_matrix.cpu().numpy(), ord=2)
        norm_disturbance_l2_bound = np.linalg.norm(disturbance_elementwise_upper_bound.cpu().numpy(), ord=2)

        lip_zero_order_l2 = F_l2_bound * hess_bound_cbf_nn
        lip_zero_order_l2 += lip_cbf_nn * F_dx_l2_bound
        lip_zero_order_l2 += cbf_alpha * lip_cbf_nn
        lip_zero_order_l2 += hess_bound_cbf_nn * norm_disturbance_channel_l2 * norm_disturbance_l2_bound

        print("==> Lipschitz constant of the zero order test function: {:.6f}".format(lip_zero_order_l2))

        mesh = FeasibilityAdaptiveMesh(
            cbf_nn=cbf_nn,
            lip_h_l2=lip_cbf_nn,
            hess_h_l2=hess_bound_cbf_nn, 
            thrid_order_h_elementwise_l2_bound=third_order_elementwise_bound_cbf_nn, 
            control_lower_bound=control_lower_bound,
            control_upper_bound=control_upper_bound,
            system=system,
            F_l1_bound=F_l1_bound,
            F_dx_l2_bound=F_dx_l2_bound, 
            F_dxdx_elementwise_l2_bound=F_dxdx_elementwise_l2_bound,
            disturbance_elementwise_upper_bound=disturbance_elementwise_upper_bound,
            disturbance_channel=disturbance_channel_matrix,
            cbf_alpha=cbf_alpha,
            lip_zero_order_l2=lip_zero_order_l2,
            state_dim=system.state_dim,
            x_lb=subregion_lb.unsqueeze(0),
            x_ub=subregion_ub.unsqueeze(0), 
            batch_size=2**14, # 16384
            dtype=config.pt_dtype,
            device=device)
        
        start_time = time.time()
        for i in range(50):

            f_lb, f_ub = mesh.bound() # shape (N,) and (N,)
            if (len(f_lb) == 0) or (len(f_ub) == 0):
                success = True
                print("> Mesh returns no regions. Done!")
                break

            f_ub_min = f_ub.min()
            f_lb_min = f_lb.min()

            print(f"> Iteration {i:02d} | Bound: [{f_lb_min.item():.6f}, {f_ub_min.item():.6f}] | l2 radius: {mesh.mesh_radius_l2:.2E} | Size: {len(f_lb):.2E} | Time: {format_time(time.time()-start_time)}")

            if torch.any(f_ub < 0.0):
                negative_idx = (f_ub < 0.0)
                negative_regions_lb, negative_regions_ub = mesh.regions[negative_idx]
                negative_regions_mid = (negative_regions_lb + negative_regions_ub) / 2.0
                negative_values = f_ub[negative_idx]
                print("> Negative region: ", negative_regions_mid.detach().cpu().numpy()[0])
                print("> Negative value: ", negative_values.detach().cpu().numpy()[0])

                # double check
                f_ub_double_check = test_func(negative_regions_mid.to(device)).detach().cpu()
                h_value = cbf_nn.forward(negative_regions_mid.to(device)).detach().cpu()
                print("> Double check: ", f_ub_double_check)
                print("> CBF value: ", h_value)
                break

            # Plot mesh
            if args.draw:
                raise NotImplementedError("Drawing is not implemented yet.")

            if f_lb_min >= 0:
                success = True
                print("> No region to refine. Done!")
                break

            if f_ub_min - f_lb_min <= precision:
                success = True
                print("> Early stopping. Done!")
                break
            
            refine_idx = (f_ub > 0)
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
    save_dict(checking_result, f"{results_dir}/00_grid_feasibility_result_{platform}_{device}_{precision:.1E}.pkl")

    print("==> Done!")

