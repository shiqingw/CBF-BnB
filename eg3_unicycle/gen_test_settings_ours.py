import sys
import os
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def generate_json_script(filename, entry):
    data_str = f"""{{
    "seed": {entry["random_seed"]},
    "true_system_name": "Unicycle",
    "cbf_nn_config": {{
        "in_features": 3,
        "out_features": 1,
        "lipschitz_constant": {entry["lipschitz_constant"]},
        "activations": "tanh",
        "num_layers": {entry["num_layer"]},
        "width_each_layer": 256,
        "input_bias": [
            0.0,
            0.0,
            0.0
        ],
        "input_transform_to_inverse": [
            1.0,
            1.0,
            1.0
        ],
        "zero_at_zero": 0,
        "alpha": {entry["cbf_alpha"]}
    }},

    "control_config": {{
        "lower_bound": [-1.0, -1.0],
        "upper_bound": [1.0, 1.0]
    }},

   "disturbance_config": {{
        "channel_matrix": [[0.0], [0.0], [0.0]],
        "lower_bound": [-0.0],
        "upper_bound": [0.0]
    }},

    "dataset_config": {{
        "state_lower_bound": [-2.0, -2.0, -1.57],
        "state_upper_bound": [2.0, 2.0, 1.57],
        "mesh_size": [40, 40, 30],
        "post_mesh_size": [80, 80, 60]
    }},

    "unsafe_set_config": {{
        "unsafe_set_lower_bound": [-0.5, -2, -1.57],
        "unsafe_set_upper_bound": [0.5, 0.0, 1.57]
    }},

    "train_config": {{
        "num_epochs": 100,
        "warmup_steps": 30,
        "batch_size": 512,
        "cbf_lr": 1e-3,
        "cbf_wd": 1e-5,
        "safe_set_weight": 0.0025,
        "unsafe_set_weight": 1.0,
        "feasibility_weight": 1.0,
        "safe_set_margin": 0.02,
        "unsafe_set_margin": 0.2,
        "feasibility_margin": 0.25
    }}
}}"""
    with open(filename, 'w') as file:
        file.write(data_str)


num_layers = [2]
num_layers.sort()
lipschitz_constants = [4.0]
lipschitz_constants.sort()
cbf_alphas = [5e-1]
cbf_alphas.sort()
random_seeds = [0, 100, 200, 300]
random_seeds.sort()

# iterate over all combinations of gammas, train_ratios, and random_seeds and generate a test_settings file for each
data = []
for num_layer in num_layers:
    for lipschitz_constant in lipschitz_constants:
        for cbf_alpha in cbf_alphas:
            for random_seed in random_seeds:
                data.append({
                    "num_layer": num_layer,
                    "lipschitz_constant": lipschitz_constant,
                    "cbf_alpha": cbf_alpha,
                    "random_seed": random_seed
                })

start = 17
exp_nums = range(start, start+len(data))
for i in range(len(data)):
    entry = data[i]
    exp_num = exp_nums[i]
    filename = os.path.join(str(Path(__file__).parent.parent), 'eg3_unicycle', 'test_settings', f"test_settings_{exp_num:03}.json")
    generate_json_script(filename, entry)