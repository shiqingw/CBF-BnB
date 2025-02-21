import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def generate_sh_script_dreal_inclusion_solve(filename, exp_nums, dreal_precision):
    with open(filename, "w") as file:
        for exp_num in exp_nums:
            command1 = f"mkdir eg3_results/{exp_num:03d}\n"
            command2 = f"python -u eg3_unicycle/dreal_inclusion_solve.py --exp_num {exp_num} " + \
                f"--dreal_precision {dreal_precision:.1E} " + \
                f"> eg3_results/{exp_num:03d}/output_dreal_inclusion_solve_{dreal_precision:.1E}.out\n"
            file.write(command1)
            file.write(command2)

def generate_sh_script_dreal_feasibility_solve(filename, exp_nums, dreal_precision):
    with open(filename, "w") as file:
        for exp_num in exp_nums:
            command1 = f"mkdir eg3_results/{exp_num:03d}\n"
            command2 = f"python -u eg3_unicycle/dreal_feasibility_solve.py --exp_num {exp_num} " + \
                f"--dreal_precision {dreal_precision:.1E} " + \
                f"> eg3_results/{exp_num:03d}/output_dreal_feasibility_solve_{dreal_precision:.1E}.out\n"
            file.write(command1)
            file.write(command2)

exp_nums = list(range(1, 13))

dreal_precision = 1e-3

file = os.path.join(str(Path(__file__).parent.parent), f"run_dreal_inclusion_solve.sh")
generate_sh_script_dreal_inclusion_solve(file, exp_nums, dreal_precision)

file = os.path.join(str(Path(__file__).parent.parent), f"run_dreal_feasibility_solve.sh")
generate_sh_script_dreal_feasibility_solve(file, exp_nums, dreal_precision)