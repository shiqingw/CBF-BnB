import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def generate_sh_script(filename, exp_nums, device):
    with open(filename, "w") as file:
        for exp_num in exp_nums:
            command1 = f"mkdir eg1_results/{exp_num:03}\n"
            command2 = f"python -u eg1_point_robot/train.py --exp_num {exp_num} --device {device} > eg2_results/{exp_num:03}/output.out\n"
            file.write(command1)
            file.write(command1)
            file.write(command2)

exp_nums = list(range(1, 13))
device = "cuda"
file = os.path.join(str(Path(__file__).parent.parent), f"run_cuda_train_one.sh")

generate_sh_script(file, exp_nums, device)