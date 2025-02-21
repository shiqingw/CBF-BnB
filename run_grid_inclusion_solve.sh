mkdir eg4_results/001
python -u eg4_quadrotor_2d/grid_inclusion_solve_second_order.py --exp_num 1 --device cuda --precision 1.0E-03 > eg4_results/001/output_grid_inclusion_solve_cuda_1.0E-03.out
mkdir eg4_results/002
python -u eg4_quadrotor_2d/grid_inclusion_solve_second_order.py --exp_num 2 --device cuda --precision 1.0E-03 > eg4_results/002/output_grid_inclusion_solve_cuda_1.0E-03.out
mkdir eg4_results/003
python -u eg4_quadrotor_2d/grid_inclusion_solve_second_order.py --exp_num 3 --device cuda --precision 1.0E-03 > eg4_results/003/output_grid_inclusion_solve_cuda_1.0E-03.out
mkdir eg4_results/004
python -u eg4_quadrotor_2d/grid_inclusion_solve_second_order.py --exp_num 4 --device cuda --precision 1.0E-03 > eg4_results/004/output_grid_inclusion_solve_cuda_1.0E-03.out
