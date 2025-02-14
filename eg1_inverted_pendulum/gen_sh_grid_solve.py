import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def generate_sh_script_grid_positive_solve(filename, exp_nums, device, precision):
    with open(filename, "w") as file:
        for exp_num in exp_nums:
            command1 = f"mkdir eg1_results/{exp_num:03d}\n"
            command2 = f"python -u eg1_inverted_pendulum/grid_inclusion_solve_second_order.py " + \
                f"--exp_num {exp_num} " +\
                f"--device {device} " +\
                f"--precision {precision:.1E} " +\
                f"> eg1_results/{exp_num:03d}/output_grid_inclusion_solve_{device}_{precision:.1E}.out\n"
            file.write(command1)
            file.write(command2)

def generate_sh_script_grid_stability_solve(filename, exp_nums, device, precision):
    with open(filename, "w") as file:
        for exp_num in exp_nums:
            command1 = f"mkdir eg1_results/{exp_num:03d}\n"
            command2 = f"python -u eg1_inverted_pendulum/grid_feasibility_solve_first_order.py " + \
                f"--exp_num {exp_num} " +\
                f"--device {device} " +\
                f"--precision {precision:.1E} " +\
                f"> eg1_results/{exp_num:03d}/output_grid_feasibility_solve_{device}_{precision:.1E}.out\n"
            file.write(command1)
            file.write(command2)

exp_nums = list(range(1, 13))
device = "cpu"
precision = 1.0E-3

file = os.path.join(str(Path(__file__).parent.parent), f"run_grid_inclusion_solve.sh")
generate_sh_script_grid_positive_solve(file, exp_nums, device, precision)

file = os.path.join(str(Path(__file__).parent.parent), f"run_grid_feasibility_solve.sh")
generate_sh_script_grid_stability_solve(file, exp_nums, device, precision)