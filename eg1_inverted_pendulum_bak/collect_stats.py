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

    # Step 2
    cbf_alpha = test_settings["cbf_nn_config"]["alpha"]
    all_info.append(cbf_alpha)

    # Step 3
    feasibility_results = load_dict(f"{results_dir}/feasibility_results.pkl")
    all_info.append(int(feasibility_results["success"]))
    all_info.append(feasibility_results["safe_set_percentage"])

    # Step 4
    train_info = load_dict(f"{results_dir}/train_results.pkl")
    all_info.append(train_info["hess_bound_cbf_nn"])

    # Step 5
    all_info.append(int(feasibility_results["safety_success"]))
    all_info.append(int(feasibility_results["feasibility_success"]))

    # Step 6
    all_info.append(train_info["time"])

    # Step 7
    all_info.append(int(feasibility_results["best_epoch"]))

    # Step 8
    seed = test_settings["seed"]
    all_info.append(seed)

    # Step 9
    all_info.append(exp_num)
    
    return all_info

if __name__ == "__main__":
    # save to a txt file with separator that can be directly copy pasted to excel-
    with open("text.txt", "w") as file:
        for exp_num in range(1, 17):
            try:
                out = diagnosis(exp_num)
                print("#############################################")
                for ii in out:
                    file.write(f"{ii}\t")
                file.write(f"\n")
            except:
                file.write(f"\n")