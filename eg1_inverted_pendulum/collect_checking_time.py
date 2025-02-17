import os
import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from cores.utils.utils import load_dict

def diagnosis(exp_num):
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg1_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings_{:03d}.json".format(results_dir, exp_num)
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    all_info = []
    # Step 1
    num_layers = test_settings["cbf_nn_config"]["num_layers"]
    all_info.append(num_layers)

    # Step 2a
    filepath = f"{results_dir}/00_dreal_inclusion_result_1.0E-03.pkl"
    if os.path.exists(filepath):
        results = load_dict(filepath)
        all_info.append(int(results["success"]))
        all_info.append(results["time"])
    else:
        all_info.append(0)
        all_info.append(0)

    # Step 2b
    filepath = f"{results_dir}/00_grid_inclusion_result_linux_cpu_1.0E-03.pkl"
    if os.path.exists(filepath):
        results = load_dict(filepath)
        all_info.append(int(results["success"]))
        all_info.append(results["time"])
    else:
        all_info.append(0)
        all_info.append(0)

    # Step 2c
    filepath = f"{results_dir}/00_grid_inclusion_result_linux_cuda_1.0E-03.pkl"
    if os.path.exists(filepath):
        results = load_dict(filepath)
        all_info.append(int(results["success"]))
        all_info.append(results["time"])
    else:
        all_info.append(0)
        all_info.append(0)

    # Step 3a
    filepath = f"{results_dir}/00_dreal_feasibility_result_1.0E-03.pkl"
    if os.path.exists(filepath):
        results = load_dict(filepath)
        all_info.append(int(results["success"]))
        all_info.append(results["time"])
    else:
        all_info.append(0)
        all_info.append(0)

    # Step 3b
    filepath = f"{results_dir}/00_grid_feasibility_result_linux_cpu_1.0E-03.pkl"
    if os.path.exists(filepath):
        results = load_dict(filepath)
        all_info.append(int(results["success"]))
        all_info.append(results["time"])
    else:
        all_info.append(0)
        all_info.append(0)

    # Step 3c
    filepath = f"{results_dir}/00_grid_feasibility_result_linux_cuda_1.0E-03.pkl"
    if os.path.exists(filepath):
        results = load_dict(filepath)
        all_info.append(int(results["success"]))
        all_info.append(results["time"])
    else:
        all_info.append(0)
        all_info.append(0)
    
    # Step 5
    seed = test_settings["seed"]
    all_info.append(seed)

    # Step 6
    all_info.append(exp_num)
    
    return all_info

if __name__ == "__main__":
    # save to a txt file with separator that can be directly copy pasted to excel-
    with open("text.txt", "w") as file:
        exp_nums = list(range(1, 13))
        for exp_num in exp_nums:
            try:
                out = diagnosis(exp_num)
                print("#############################################")
                for ii in out:
                    file.write(f"{ii}\t")
                file.write(f"\n")
            except:
                file.write(f"\n")