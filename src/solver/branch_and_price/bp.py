from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import math
import time
import copy

from src.solver.branch_and_price.rmp import PriceProblem
from src.problem.solution import Solution

class NodeSelection(Enum):
    DEPTH_FIRST = 1
    BEST_BOUND = 2
    BEST_ESTIMATE = 3

class BranchingDecision:
    def __init__(self, arc: Optional[Tuple[int, int]] = None, 
                 vehicle: Optional[int] = None,
                 value: Optional[float] = None):
        self.arc = arc  # (i, j)弧分支
        self.vehicle = vehicle    # 车辆分支
        self.value = value        # 分支值
        
    def __str__(self):
        if self.arc:
            return f"Arc({self.arc[0]}->{self.arc[1]})"
        elif self.customer is not None:
            return f"Customer({self.customer})"
        elif self.vehicle is not None:
            return f"Vehicle({self.vehicle})"
        return "Unknown Branch"

class SearchNode:
    def __init__(self, parent=None, branching_decision: BranchingDecision = None, estimated_lower_bound=-float('inf')):
        self.parent = parent
        self.branching_decision = branching_decision
        self.estimated_lower_bound = -float('inf')
        self.lower_bound = -float('inf')  # 下界（最小化问题）
        self.price_problem = None  # RMP模型
        self.solution = None  # 当前解
        self.children = []  # 子节点
        self.is_integer = False

    def __lt__(self, other):
        # 用于节点选择策略
        return self.lower_bound < other.lower_bound

class BranchAndPrice:
    def __init__(self, problem, node_selection=NodeSelection.BEST_BOUND, time_limit=60):
        self.problem = problem
        self.node_selection = node_selection
        self.best_solution = None
        self.upper_bound = float('inf')
        self.search_tree = []
        self.stats = {
            'nodes_explored': 0,
            'total_time': 0,
            'lp_solves': 0,
            'branching_decisions': 0
        }
        self.max_lp_iterations = 1000
        self.time_limit = time_limit  # 1小时
        self.search_record = []
        self.best_solution = Solution()
        
    def solve(self):
        start_time = time.time()
        
        # 生成初始解
        root = SearchNode()
        root.price_problem = PriceProblem(self.problem, initial_route=[])
        self.search_tree.append(root)
        
        # 主循环
        while self.search_tree and time.time() - start_time < self.time_limit:
            # 更新统计
            self.stats['nodes_explored'] += 1
            # 选择节点
            node = self.select_node()
            self.search_record.append(node)
            # 列生成循环
            self.column_generation(node)
            
            # 剪枝：如果下界超过上界，则剪枝
            if node.lower_bound >= self.upper_bound:
                self.prune_node(node)
                continue
                
            # 检查整数解
            if node.is_integer:
                # 更新最好解
                if node.lower_bound < self.upper_bound:
                    self.upper_bound = node.lower_bound 
                    self.best_solution = node.solution
                self.prune_node(node)
                continue
                
            # 分支
            self.branch(node)
            
            
        
        # 记录总时间
        self.stats['total_time'] = time.time() - start_time
        self.best_solution.solve_time = self.stats['total_time']
        return self.best_solution
        
    def select_node(self) -> SearchNode:
        """根据节点选择策略选择下一个节点"""
        if self.node_selection == NodeSelection.DEPTH_FIRST:
            # 深度优先：选择最后一个节点
            return self.search_tree.pop()
        elif self.node_selection == NodeSelection.BEST_BOUND:
            # 最佳下界：找到下界最小的节点
            best_node = min(self.search_tree, key=lambda n: n.lower_bound)
            # 从搜索树中移除该节点
            self.search_tree.remove(best_node)
            return best_node
            
        elif self.node_selection == NodeSelection.BEST_ESTIMATE:
            # 最佳估计：使用估计值选择
            best_node = min(self.search_tree, key=lambda n: n.estimate)
            # 从搜索树中移除该节点
            self.search_tree.remove(best_node)
            return best_node
        else:
            return self.search_tree.pop(0)
            
    def column_generation(self, node: SearchNode):
        """在给定节点上执行列生成"""
        
        node.is_integer, node.solution = node.price_problem.solve(
            max_iter = self.max_lp_iterations, 
            max_time_limit = self.time_limit
        )
        node.optimal = node.price_problem.optimal
        if node.optimal:
            node.lower_bound = node.price_problem.objective
        

    
    def branch(self, node: SearchNode):
        """在给定节点上执行分支"""
        # 选择分支变量
        branching_decision = self.select_branching_variable(node)
        
        if not branching_decision:
            return
            
        self.stats['branching_decisions'] += 1
        
        # 创建两个子节点
        child1 = SearchNode(parent=node, branching_decision=branching_decision)
        child2 = SearchNode(parent=node, branching_decision=branching_decision)
        
        # 应用分支约束
        self.apply_branching_constraint(child1, branching_decision, direction=1)
        self.apply_branching_constraint(child2, branching_decision, direction=0)
        
        # 添加到搜索树
        node.children = [child1, child2]
        self.search_tree.extend([child1, child2])
    
    def select_branching_variable(self, node: SearchNode) -> BranchingDecision:
        """选择分支变量"""
        # 首先尝试分数弧分支
        fractional_arc = self.find_fractional_arc(node)
        if fractional_arc:
            return BranchingDecision(arc=fractional_arc)

        # 最后尝试车辆分支
        fractional_vehicle = self.find_fractional_vehicle(node)
        if fractional_vehicle:
            return BranchingDecision(vehicle=fractional_vehicle)
            
        return None
    
    def find_fractional_arc(self, node: SearchNode) -> Optional[Tuple[int, int]]:
        """找到分数弧用于分支"""
        # 计算每条弧的使用频率
        arc_usage = {}
        for idx, route in node.solution.routes.items():
            for i in range(len(route)-1):
                arc = (route[i], route[i+1])
                arc_usage[arc] = arc_usage.get(arc, 0) + node.solution.route_used[idx]
        
        # 找到最接近0.5的弧
        min_diff = float('inf')
        best_arc = None
        for arc, usage in arc_usage.items():
            diff = abs(usage - 0.5)
            if diff < min_diff:
                min_diff = diff
                best_arc = arc
                
        return best_arc

    def find_fractional_vehicle(self, node: SearchNode) -> Optional[int]:
        """找到分数车辆用于分支"""
        # 在VRP中，通常所有车辆相同，但如果有异构车队，可以按类型分支
        # 这里简化处理：分支在路径数量上
        total_vehicles = sum(var.x for var in node.rmp.theta)
        if abs(total_vehicles - round(total_vehicles)) > 1e-3:
            return total_vehicles  # 分支在整数部分
        return None
    
    def apply_branching_constraint(self, node: SearchNode, 
                                decision: BranchingDecision, direction: int):
        """应用分支约束到子节点RMP"""
        M = 1e6
        # 复制父节点的RMP
        problem = copy.deepcopy(node.parent.price_problem.problem)
        initial_route = copy.deepcopy(node.parent.price_problem.solution.rmp_used_route)
        if decision.arc:
            # 弧分支
            i, j = decision.arc
            if direction == 1:  # 强制使用弧(i,j)
                # 添加约束：所有包含弧(i,j)的路径的θ之和 >= 1
                for (k,m), cost in problem.travel_cost.items():
                    if k == i and m != j:
                        problem.travel_cost[k,m] = M
                for route in initial_route:
                    cus_ids = [problem.depot_id]+ route.cus_ids + [problem.depot_id]
                    for idx in range(len(cus_ids)-1):
                        if cus_ids[idx] == i and cus_ids[idx+1] != j:
                            route.cost = M
            else:  # 禁止使用弧(i,j)
                # 添加约束：所有包含弧(i,j)的路径的θ之和 = 0
                for (k,m), cost in problem.travel_cost.items():
                    if k == i and m == j:
                        problem.travel_cost[k,m] = M
                for route in initial_route:
                    cus_ids = [problem.depot_id]+ route.cus_ids + [problem.depot_id]
                    for idx in range(len(cus_ids)-1):
                        if cus_ids[idx] == i and cus_ids[idx+1] == j:
                            route.cost = M

        elif decision.vehicle is not None:
            # 车辆分支
            k = decision.vehicle
            if direction == 1:  # 至少使用k辆车
                problem._min_vehicle = math.ceil(k)
            else:  # 最多使用k-1辆车
                problem._max_vehicle = math.floor(k)
        
        
        child_pp = PriceProblem(
            problem, initial_route=initial_route
        )
        
        node.price_problem = child_pp
    
    def prune_node(self, node: SearchNode):
        """剪枝节点"""
        if node in self.search_tree:
            self.search_tree.remove(node)
            
    def print_stats(self):
        """打印求解统计信息"""
        print("\n=== Branch-and-Price Statistics ===")
        print(f"Total time: {self.stats['total_time']:.2f} seconds")
        print(f"Nodes explored: {self.stats['nodes_explored']}")
        print(f"LP solves: {self.stats['lp_solves']}")
        print(f"Branching decisions: {self.stats['branching_decisions']}")
        if self.best_solution:
            print(f"Best solution cost: {self.best_solution.travel_cost}")
            print(f"Vehicles used: {self.best_solution.num_vehicles}")
        else:
            print("No feasible solution found")

# 示例使用
if __name__ == "__main__":
    from src.problem.mock_problem import mock_problem as problem
    from src.instance_parser.instance_parser import solomon_parser
    problem = solomon_parser(25, 'r102')
    
    # 创建Branch-and-Price求解器
    bp = BranchAndPrice(problem, 
            time_limit=3600)
    
    from src.solver.branch_and_price.plot_iter import plot_iter_records
    # 求解问题
    solution = bp.solve()
    
    # 打印结果和统计
    bp.print_stats()
    
    if solution:
        print("\nBest solution routes:")
        for veh_id, route in solution.routes.items():
            print(f"Vehicle {veh_id}: {route}")