from src.solver.branch_and_price import BranchAndPrice
from src.problem.mock_problem import mock_problem as problem
from src.instance_parser.instance_parser import solomon_parser

# 读取算例
problem = solomon_parser(25, 'r102')

# homberger算例读取示例
# from src.instance_parser.instance_parser import homberger_parser
# problem = homberger_parser(200, 'C1_2_1', 2)


# 创建Branch-and-Price求解器
bp = BranchAndPrice(problem, time_limit=3600)

# 求解问题
solution = bp.solve()

# 打印结果和统计
bp.print_stats()

if solution:
    print("\nBest solution routes:")
    for veh_id, route in solution.routes.items():
        print(f"Vehicle {veh_id}: {route}")