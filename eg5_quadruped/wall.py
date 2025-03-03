import json
import sys
import os
import argparse
import shutil
import torch
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
from scipy.spatial.transform import Rotation
import scipy.sparse as sparse
import osqp

from Go2Py.sim.mujoco import Go2Sim
from cores.dynamical_systems.create_system import get_system
from cores.lip_nn.models import LipschitzNetwork
from cores.utils.utils import seed_everything, save_nn_weights, save_dict, get_grad_l2_norm, format_time
from cores.utils.config import Configuration
from cores.utils.draw_utils import draw_curve, draw_multiple_curves
from cores.utils.draw_utils import draw_safe_set_contour, draw_feasibility_condition_contour
from cores.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from cores.utils.online_utils import label_rectangles, train_cbf
from cores.utils.osqp_utils import init_osqp

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=1, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    args = parser.parse_args()

    # Create result directory
    print("==> Creating result directory ...")
    exp_num = args.exp_num
    results_dir = "{}/eg5_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
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

    # Control config
    control_config = test_settings["control_config"]
    control_lower_bound = torch.tensor(control_config["lower_bound"], dtype=config.pt_dtype, device=device)
    control_upper_bound = torch.tensor(control_config["upper_bound"], dtype=config.pt_dtype, device=device)

    # robot.model.hfield_data is 300x300
    # Define the height map
    print("==> Launching the simulator ...")
    print("> Define the height map ...")
    height_map = np.zeros((300, 300), dtype=np.float32)
    height_map[:, 250:] = 2.6
    robot = Go2Sim(mode='highlevel',
                   height_map=height_map,
                   moving_viewer_camera=True)
    robot.standUpReset()

    # Collision config
    collision_config = test_settings["collision_config"]
    collision_front_offset = collision_config["front_offset"]
    collision_left_right_offset = collision_config["left_right_offset"]
    collision_rear_offset = collision_config["rear_offset"]
    collision_offset_in_body = np.array([[collision_front_offset, collision_left_right_offset, 0],
                                         [collision_front_offset, -collision_left_right_offset, 0],
                                         [-collision_rear_offset, -collision_left_right_offset, 0],
                                         [-collision_rear_offset, collision_left_right_offset, 0]],
                                         dtype=config.np_dtype)

    # Simulation settings
    sumulator_config = test_settings["simulator_config"]
    simulator_dt = sumulator_config["simulator_dt"]
    T = sumulator_config["total_time"]
    horizon = int(T/simulator_dt)
    print("> T:", T)
    print("> horizon:", horizon)
    lidar_dt = sumulator_config["lidar_dt"] # 0.09° (5 Hz), 0.18° (10 Hz), 0.36° (20 Hz)
    lidar_step = int(lidar_dt/simulator_dt)
    lidar_nray = sumulator_config["lidar_nray"]
    print("> simulator_dt:", simulator_dt)
    print("> lidar_dt:", lidar_dt)
    print("> lidar_step:", lidar_step)
    cbf_training_dt = sumulator_config["cbf_training_dt"]
    cbf_training_step = int(cbf_training_dt/simulator_dt)
    print("> cbf_training_dt:", cbf_training_dt)
    print("> cbf_training_step:", cbf_training_step)

    # CBF config
    cbf_config = test_settings["cbf_nn_config"]

    # Training config
    train_config = test_settings["train_config"]

    # Define OSQP problem
    print("==> Define OSQP problem")
    n_vars = system.control_dim
    n_cbf = 1
    n_in = n_vars + n_cbf
    cbf_qp = init_osqp(n_v=n_vars, n_in=n_in, config=config)

    # Visualize the collision corners and lidar points
    id_geom_offset = 0
    collision_corner_id_geom_offset = id_geom_offset
    point_cloud_id_geom_offset = id_geom_offset + len(collision_offset_in_body)
    flag_cbf = False

    for step_counter in range(0, horizon):
        loop_start_time = time.time()
    
        pos, quat = robot.getPose()
        R_b_to_w = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        rpy = Rotation.from_matrix(R_b_to_w).as_euler('xyz', degrees=False)
        theta_2d = np.arctan2(R_b_to_w[1,0], R_b_to_w[0,0])
        R_2d_to_3d = np.array([[np.cos(theta_2d), -np.sin(theta_2d), 0],
                               [np.sin(theta_2d), np.cos(theta_2d), 0],
                               [0, 0, 1]], dtype=config.np_dtype)
        collision_corners_in_world = pos + collision_offset_in_body @ R_2d_to_3d.T
        for corner_ind in range(len(collision_corners_in_world)):
            robot.add_visual_capsule(point1=collision_corners_in_world[corner_ind],
                                     point2=collision_corners_in_world[(corner_ind+1)%len(collision_corners_in_world)], 
                                     radius=0.01, 
                                     rgba=np.array([0,1,0,1]), 
                                     id_geom_offset=collision_corner_id_geom_offset+corner_ind, 
                                     limit_num=False)

        if (step_counter % lidar_step == 0) and abs(rpy[1]) <= 0.1:
            pcd_in_body = robot.getLaserScan(nray=lidar_nray, max_range=30.0)
            pcd_directions_in_world = pcd_in_body @ R_b_to_w.T
            pcd_in_world = pos + pcd_directions_in_world
            pcd_xy_in_world = pcd_in_world[:, :2]
            for point_ind in range(len(pcd_in_world)):
                robot.add_visual_ellipsoid(np.array([0.01, 0.01, 0.01], dtype=config.np_dtype),
                                    pcd_in_world[point_ind],
                                    np.eye(3).astype(config.np_dtype),
                                    np.array([1,0,0,1]),
                                    id_geom_offset=point_cloud_id_geom_offset+point_ind)
            if (flag_cbf == False) and (step_counter % cbf_training_step == 0):
                
                distance_samples = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=config.np_dtype)
                xyz_states_np = pos + pcd_directions_in_world[:, None, :] * distance_samples[None, :, None]
                xy_states_np = xyz_states_np[:, :, :2]
                xy_states_np = xy_states_np.reshape(-1, 2)
                n_theta_samples = 11
                theta_samples = np.linspace(-np.pi, np.pi, n_theta_samples)
                xy_states_repeated_np = np.repeat(xy_states_np, n_theta_samples, axis=0)
                theta_tiled_np = np.tile(theta_samples, xy_states_np.shape[0]).reshape(-1, 1)
                states_np = np.concatenate([xy_states_repeated_np, theta_tiled_np], axis=1)
                labels_np = label_rectangles(states_np, pcd_xy_in_world,
                                            collision_front_offset, collision_left_right_offset, collision_rear_offset)

                # Train CBF
                cbf_nn = train_cbf(cbf_config=cbf_config,
                          nn_input_bias=[pos[0], pos[1], 0],
                          train_config=train_config,
                          states_np=states_np, 
                          labels_np=labels_np, 
                          system=system, 
                          control_lower_bound=control_lower_bound,
                          control_upper_bound=control_upper_bound, 
                          results_dir=results_dir,
                          step_counter=step_counter,
                          config=config,
                          device=device)
            
                flag_cbf = True
            
        C = np.zeros((n_in, n_vars), dtype=config.np_dtype)
        lb = np.zeros(n_in, dtype=config.np_dtype)
        ub = np.zeros(n_in, dtype=config.np_dtype)
        x_np = np.array([pos[0], pos[1], theta_2d], dtype=config.np_dtype)
        drift_np = np.zeros(system.state_dim, dtype=config.np_dtype)
        actuation_np = np.eye(system.control_dim, dtype=config.np_dtype)
        h_torch, h_dx_torch = cbf_nn.forward_with_jacobian(torch.tensor(x_np, dtype=config.pt_dtype, device=device).unsqueeze(0))
        h_np = h_torch.detach().cpu().numpy().squeeze(0)
        h_dx_np = h_dx_torch.detach().cpu().numpy().squeeze(0)
        C[:n_vars, :n_vars] = np.eye(n_vars, dtype=config.np_dtype)
        lb[:n_vars] = control_lower_bound.cpu().numpy()
        ub[:n_vars] = control_upper_bound.cpu().numpy()
        C[n_vars:, :n_vars] = h_dx_np @ actuation_np
        lb[n_vars:] = -cbf_config["alpha"] * h_np + h_dx_np @ drift_np
        ub[n_vars:] = np.inf
        u_nominal = np.array([0.4, 0, 0.0], dtype=config.np_dtype)
        g = -u_nominal

        data = C.flatten()
        rows, cols = np.indices(C.shape)
        row_indices = rows.flatten()
        col_indices = cols.flatten()
        Ax = sparse.csc_matrix((data, (row_indices, col_indices)), shape=C.shape)
        cbf_qp.update(q=g, l=lb, u=ub, Ax=Ax.data)
        results = cbf_qp.solve()
        u_safe = results.x
        print(u_safe)

        robot.stepHighlevel(u_safe[0], u_safe[1], u_safe[2], step_height=0, kp=[2, 0.5, 0.5], ki=[0.02, 0.01, 0.01])
        
        time.sleep(max(0, simulator_dt - (time.time()-loop_start_time)))
