from mip import Model, BINARY, xsum, minimize, Column, OptimizationStatus
from typing import List, Dict, Tuple, Optional
from collections import deque
import heapq
import random
import time

from src.solver.branch_and_price.plot_iter import plot_iter_records
from src.problem.solution import Solution

class Route:
    def __init__(
            self,
            cost,
            cus_ids,
        ):
        self.cost = cost
        self.cus_ids = cus_ids

class RMP:
    def __init__(
            self,
            problem,
            initial_route: List[Route]
        ):
        self.integer = False
        self.problem = problem
        self.M = 1e6
        self.routes = [
            Route(self.M, self.problem.customer_ids)
        ]
        self.model = Model(solver_name='CBC', name='RMP')
        self.theta = []
        # 全部客户点都访问的路径
        slack_route = self.model.add_var(
            obj = self.M, name='slack_route'
        )
        self.theta.append(slack_route)
        # 创建约束
        self.cus_constraints = []
        for cus_id in problem.customer_ids:
            self.cus_constraints.append(
                self.model.add_constr(
                    slack_route >= 1, name='cus_%d'%cus_id
                )
            ) #todo =1求的解更好 
        self.fleet_size_ub = [self.model.add_constr(
            slack_route <= problem.max_vehicle, name='fleet_size_ub'
        )]

        self.fleet_size_lb = [self.model.add_constr(
            slack_route >= problem.min_vehicle, name='fleet_size_lb'
        )]

        # 添加路径
        for route in initial_route:
            self.add_column(route)
    
    def add_column(self, route):
        self.routes.append(route)
        cus_coeff = [0]*self.problem.customer_num
        for cus_id in route.cus_ids:
            cus_coeff[self.problem.get_customer_index(cus_id)] = 1
        column = Column(
            self.cus_constraints+self.fleet_size_ub,
            cus_coeff + [1,1]
        )
        self.theta.append(
            self.model.add_var(
                obj=route.cost, column=column,
                name='theta_%d'%(len(self.theta)+1)
            )
        )
        # print(f'new_column={route.cus_ids}')

    def solve(self, max_seconds=60):
        self.model.verbose=2
        self.model.max_gap = 0
        self.status = self.model.optimize(max_seconds=max_seconds)
        if self.status != OptimizationStatus.OPTIMAL:
            print(f"Solving failed. Status: {self.status}")
            raise Exception

    def get_dual(self) -> Dict[int, float]:
        """获取每个覆盖约束的对偶价格，用于子问题求解。"""
        return {
            'customer_dual': {
                cust_id: self.cus_constraints[idx].pi for idx, cust_id in enumerate(self.problem.customer_ids)
            },
            'fleet_size_ub_dual': self.fleet_size_ub[0].pi,
            'fleet_size_lb_dual': self.fleet_size_lb[0].pi
        }
    
    def is_integer_solution(self, tol=1e-5) -> bool:
        """检查当前解是否为整数解"""
        for var in self.theta:
            # 忽略接近0的变量（数值误差）
            if var.x > tol:  
                # 检查是否接近整数
                if abs(var.x - round(var.x)) > tol:
                    return False
        return True

    def get_solution(self):
        solution = Solution()
        if self.model.objective_value == self.M:
            solution.solved = False
        elif self.status == OptimizationStatus.OPTIMAL:
            solution.solved = True
            solution.optimal = True
        elif self.status == OptimizationStatus.FEASIBLE:
            solution.solved = True
        else:
            return 
        vehicle_count = 0
        travel_cost = 0
        route_used = {}
        rmp_used_route = []
        for idx, v in enumerate(self.theta):
            if v.x > 1e-9:
                vehicle_count += 1
                route_used[vehicle_count] = round(v.x, 9)
                rmp_used_route.append(self.routes[idx])
                solution.routes[vehicle_count]=[self.problem.depot_id]+ self.routes[idx].cus_ids + [self.problem.depot_id]
                travel_cost += v.x * self.routes[idx].cost
        solution.travel_cost = travel_cost
        solution.num_vehicles = vehicle_count
        solution.route_used = route_used
        solution.rmp_used_route = rmp_used_route
        return solution
    
    def get_obj_value(self):
        return self.model.objective_value
class Label:
    """表示路径上的一个状态（标签）"""
    __slots__ = ('rmp_dual', 'current_node', 'cost', 'dual_sum', 'time', 'load', 'visited', 'prev_label', 'arc_used')
    
    def __init__(self, rmp_dual: dict, current_node: int, cost: float, dual_sum: float, time: float, 
                 load: float, visited: int, prev_label: Optional['Label'] = None, 
                 arc_used: Optional[Tuple[int, int]] = None):
        self.rmp_dual = rmp_dual
        self.current_node = current_node    # 当前节点索引
        self.cost = cost                    # 累计实际成本
        self.dual_sum = dual_sum            # 累计对偶值之和
        self.time = time                    # 当前时间（离开前节点的时间）
        self.load = load                    # 当前载重
        self.visited = visited              # 已访问节点位掩码
        self.prev_label = prev_label        # 前驱标签（用于回溯路径）
        self.arc_used = arc_used            # 到达当前节点的弧\

    @property
    def reduced_cost(self) -> float:
        return self.cost - self.dual_sum - self.rmp_dual['fleet_size_ub_dual'] - self.rmp_dual['fleet_size_lb_dual']

class SP:
    def __init__(self, problem, dual: Dict, config: Dict = None):
        self.problem = problem
        self.dual = dual
        print(self.dual)
        self.config = config or {}

        self.find_first_negative = self.config.get('find_first_negative', False)
        self.enable_strong_dominance = self.config.get('enable_strong_dominance', False)

        self.depot_start = 0
        self.depot_end = self.problem.node_num              # 👈 虚拟终点编号
        self.total_node_num = self.problem.node_num + 1     # 包含终点的节点总数

        # 节点编号 -> 客户 ID（起点/终点都映射为 0）
        self.index_to_node = {self.depot_start: 0, self.depot_end: 0}
        for idx, cid in enumerate(self.problem.customer_ids, 1):
            self.index_to_node[idx] = cid

    def solve(self) -> Tuple[Optional['Route'], float]:
        n = self.problem.customer_num
        labels = {i: [] for i in range(self.total_node_num)}
        queue = deque()

        best_label = None
        best_reduced_cost = 0

        # 使用堆来存储前N个最优解 (最小堆存储负约化成本，相当于最大堆存储约化成本)
        top_routes = []  # 存储元组: (负约化成本, 标签)
        topN = 3
        update_reduce_cost_count = 0
        # best_solution_strategy = 'topN' # best
        best_solution_strategy = 'best' # best

        # 初始化起点标签
        root = Label(
            rmp_dual=self.dual,
            current_node=self.depot_start,
            cost=0.0,
            dual_sum=0.0,
            time=0.0,
            load=0.0,
            visited=0
        )
        labels[self.depot_start].append(root)
        queue.append(root)

        while queue:
            label = queue.popleft()
            i = label.current_node
            for j in range(1, self.total_node_num):  # 不能再扩展到起点
                if j == self.depot_start:
                    continue
                
                is_customer = j < self.problem.node_num
                is_end = j == self.depot_end

                # 客户点：检查已访问
                if is_customer:
                    pos = j - 1
                    if label.visited & (1 << pos):
                        continue

                # 载重约束
                demand = 0 if is_end else self.problem.demand[self.index_to_node[j]]
                new_load = label.load + demand
                if new_load > self.problem.vehicle_globel_max_cap:
                    continue

                # 时间窗 & 服务时间
                travel_time = self.problem.travel_time[
                    self.index_to_node[i], 
                    self.index_to_node[j]
                    ] if self.index_to_node[i] != self.index_to_node[j] else 0
                arrival = label.time + travel_time

                latest = self.problem.node_serve_window[self.index_to_node[j]][1]
                if arrival > latest:
                    continue

                earliest = self.problem.node_serve_window[self.index_to_node[j]][0]
                new_time = max(arrival, earliest)
                # if not is_end:
                new_time += self.problem.node_serve_time[self.index_to_node[j]]

                # 成本
                travel_cost = self.problem.travel_cost[
                    self.index_to_node[i], 
                    self.index_to_node[j]
                ] if self.index_to_node[i] != self.index_to_node[j] else 0
                new_cost = label.cost + travel_cost

                # Dual 值更新（仅客户点有）
                new_dual_sum = label.dual_sum
                if is_customer:
                    cust_id = self.index_to_node[j]
                    new_dual_sum += self.dual['customer_dual'].get(cust_id, 0.0)

                # 访问集合更新（仅客户点）
                new_visited = label.visited
                if is_customer:
                    new_visited |= (1 << (j - 1))

                # 构建新标签
                new_label = Label(
                    rmp_dual=self.dual,
                    current_node=j,
                    cost=new_cost,
                    dual_sum=new_dual_sum,
                    time=new_time,
                    load=new_load,
                    visited=new_visited,
                    prev_label=label,
                    arc_used=(i, j)
                )

                if is_end:
                    if label.visited == 0:
                        continue  # 防止空路径直接连终点

                    reduced_cost = new_label.reduced_cost
                    if reduced_cost < best_reduced_cost:
                        best_reduced_cost = reduced_cost
                        best_label = new_label
                        if reduced_cost < 0 and self.find_first_negative:
                            return self.create_route(new_label), reduced_cost
                    # 如果堆未满，直接添加
                    if reduced_cost < 0:
                        update_reduce_cost_count += 1
                        if len(top_routes) < topN:
                            heapq.heappush(top_routes, (-reduced_cost, -update_reduce_cost_count, new_label))
                        else:
                            # 如果比堆中最小的负约化成本大（即实际约化成本更小）
                            if reduced_cost < top_routes[0][0]:
                                heapq.heapreplace(top_routes, (-reduced_cost, -update_reduce_cost_count, new_label))
                    continue

                if not self.is_dominated(new_label, labels[j]):
                    labels[j] = [l for l in labels[j] if not self.dominates(new_label, l)] #! 重复计算可优化
                    labels[j].append(new_label)
                    queue.append(new_label)

        if best_label is None or best_reduced_cost >= -1e-9:
            return None, best_reduced_cost
        
        if best_solution_strategy == 'topN':
            # 随机选择一个最优解
            best_reduced_cost, _, best_label = random.choice(top_routes)
            best_reduced_cost = -best_reduced_cost
            return self.create_route(best_label), best_reduced_cost
        else:
            return self.create_route(best_label), best_reduced_cost

    
    def create_route(self, label: Label) -> 'Route':
        path = []
        cur = label
        while cur.prev_label is not None:
            path.append(cur.current_node)
            cur = cur.prev_label
        path.reverse()
        customer_ids = [self.index_to_node[i] for i in path if 1 <= i < self.problem.node_num]
        return Route(cost=label.cost, cus_ids=customer_ids)

    def dominates(self, l1: Label, l2: Label) -> bool:
        return (
            l1.reduced_cost < l2.reduced_cost
            # l1.cost <= l2.cost and
            # l1.time <= l2.time and
            # l1.load <= l2.load and
            # and (~l1.visited & l2.visited) == 0 
            # (l1.cost < l2.cost or l1.time < l2.time or l1.load < l2.load)
        )

    def is_dominated(self, new_label: Label, label_list: List[Label]) -> bool:
        return any(self.dominates(old, new_label) for old in label_list)

class IterRecord:
    def __init__(self, iter_idx, best_obj):
        self.iter_idx = iter_idx
        self.best_obj = best_obj
    
    def __repr__(self):
        return f"IterRecord(iter_idx={self.iter_idx}, best_obj={self.best_obj})"

class PriceProblem:
    def __init__(self, problem, initial_route):
        self.problem = problem
        self.initial_route = initial_route
        self.solution = None
        self.objective = None
        self.is_integer = False
        self.optimal = False
        self.rmp = RMP(self.problem, self.initial_route)
        self.stats = {
            'column_generated':0,
            'total_time': 0,
            'iter_record': [],
        }

    def solve(self, max_iter=1000,  max_time_limit=300):
        start = time.time()
        rmp = self.rmp
        rmp.solve()
        print(rmp.get_solution())
        dual = rmp.get_dual()
        count_iter = 0
        self.stats['iter_record'].append(
            IterRecord(count_iter, rmp.get_obj_value())
        )
        while True:
            count_iter += 1
            sp_solver = SP(self.problem, dual)
            route, reduced_cost = sp_solver.solve()
            if route:
                print(f"Found improving route: Cost={route.cost}, Path={route.cus_ids}")
                print(f"Reduced cost: {reduced_cost:.9f}")
            else:
                print("No improving route found")
                self.optimal = True
                break
            if count_iter > max_iter:
                print('exceed max iter')
                break
            if time.time() - start > max_time_limit:
                print('exceed max time')
                break
            rmp.add_column(route)
            rmp.solve()
            dual = rmp.get_dual()
            self.stats['iter_record'].append(
                IterRecord(count_iter, rmp.get_obj_value())
            )
            print(f'round {count_iter}')
            if count_iter%100 == 0:
                print(dual)
                print(rmp.get_solution())
        
        # 获得解
        self.is_integer = rmp.is_integer_solution()
        self.solution = rmp.get_solution()
        print(self.solution)
        print(self.solution.route_used)
        self.objective = self.solution.travel_cost
        self.stats['total_time'] = round(time.time()-start, 2)
        self.stats['column_generated'] = count_iter

        return self.is_integer, self.solution
    
    def get_stats(self):
        return self.stats

    def plot(self,plot_id='default'):
        plot_iter_records(self.stats['iter_record'], plot_id=plot_id)


if __name__ == '__main__':
    from src.instance_parser.instance_parser import solomon_parser
    problem = solomon_parser(25, 'c104', 4)

    # from src.instance_parser.instance_parser import homberger_parser
    # problem = homberger_parser(200, 'C1_2_1', 2)
    route_1 = Route(
        2, [1]
    )
    route_2 = Route(
        3, [2]
    )
    route_3 = Route(
        10, [1,2]
    )
    routes = [
        # route_1, route_2, route_3
    ]

    pp = PriceProblem(problem, routes)
    solution = pp.solve()
