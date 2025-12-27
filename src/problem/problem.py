import math
from typing import List, Dict, Tuple, Union
from src.problem.solution import Solution

def calculate_euclidean_distance(coord1: Tuple[float, float], coord2: Tuple[float, float], round_bit:int=1) -> float:
    """计算两点之间的欧几里得距离。"""
    return math.floor(math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)*pow(10,round_bit))/pow(10,round_bit)

# --- Vehicle Class Definition ---
class Vehicle:
    """Represents a vehicle in the VRPTW problem."""
    def __init__(self, id: int, capacity: float):
        if not isinstance(id, int) or id < 0:
            raise ValueError("Vehicle ID must be a non-negative integer.")
        if not isinstance(capacity, (int, float)) or capacity <= 0:
            raise ValueError("Vehicle capacity must be a positive number.")
        self.id: int = id
        self.capacity: float = capacity

    def __repr__(self):
        return f"Vehicle(id={self.id}, capacity={self.capacity})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vehicle):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

class Node:
    """
    表示VRP问题中的一个节点，可以是仓库（Depot）或客户点。
    """
    def __init__(self,
                id: int,
                x: float,
                y: float,
                demand: float = 0.0,
                service_time: float = 0.0,
                ready_time: float = 0.0, # 最早服务开始时间
                due_time: float = float('inf'), # 最晚服务开始时间
                is_depot: bool=False
                ):
        """
        初始化一个节点实例。

        :param id: 节点的唯一标识符。仓库通常为0。
        :param x: 节点的X坐标。
        :param y: 节点的Y坐标。
        :param demand: 节点的需求量。对于仓库，通常为0。
        :param service_time: 在该节点提供服务所需的时间。对于仓库，通常为0。
        :param ready_time: 车辆可以开始在该节点服务的 Earliest Time (时间窗开始)。
        :param due_time: 车辆必须在该节点开始服务的 Latest Time (时间窗结束)。
                         如果无限制，可以设置为 `float('inf')`。
        """
        if not isinstance(id, int) or id < 0:
            raise ValueError("Node ID 必须是非负整数。")
        if not all(isinstance(coord, (int, float)) for coord in [x, y]):
            raise ValueError("坐标 (x, y) 必须是数值。")
        if not all(isinstance(val, (int, float)) and val >= 0 for val in [demand, service_time, ready_time]):
            raise ValueError("需求、服务时间、最早服务时间必须是非负数值。")
        if not isinstance(due_time, (int, float)) or due_time < ready_time:
            raise ValueError("最晚服务时间必须是数值且不小于最早服务时间。")

        self.id: int = id
        self.x: float = x
        self.y: float = y
        self.demand: float = demand
        self.service_time: float = service_time
        self.ready_time: float = ready_time
        self.due_time: float = due_time
        self.is_depot: bool = is_depot

    @property
    def coordinates(self) -> Tuple[float, float]:
        """返回节点的 (x, y) 坐标元组。"""
        return (self.x, self.y)

    @property
    def time_window(self) -> Tuple[float, float]:
        """返回节点的时间窗 (ready_time, due_time)。"""
        return (self.ready_time, self.due_time)

    def __str__(self) -> str:
        """返回节点的字符串表示，便于打印。"""
        node_type = "仓库" if self.is_depot else "客户"
        return (
            f"{node_type}节点 ID: {self.id}\n"
            f"  坐标: ({self.x:.2f}, {self.y:.2f})\n"
            f"  需求: {self.demand:.2f}\n"
            f"  服务时间: {self.service_time:.2f}\n"
            f"  时间窗: [{self.ready_time:.2f}, {self.due_time:.2f}]"
        )

    def __repr__(self) -> str:
        """返回节点的官方字符串表示，便于调试和重建对象。"""
        return (
            f"Node(id={self.id}, x={self.x}, y={self.y}, demand={self.demand}, "
            f"service_time={self.service_time}, ready_time={self.ready_time}, "
            f"due_time={self.due_time})"
        )

    def __eq__(self, other) -> bool:
        """定义节点相等性，基于ID。"""
        if not isinstance(other, Node):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """允许将Node对象放入集合或作为字典键。"""
        return hash(self.id)

class Arc:
    """
    表示VRP问题中连接两个节点的弧（或边）。
    弧包含起点节点、终点节点、旅行成本和旅行时间。
    """
    def __init__(self,
                from_node_id: int,
                to_node_id: int,
                travel_cost: Union[int, float],
                travel_time: Union[int, float]):
        """
        初始化一个弧实例。

        :param from_node_id: 弧的起始节点的唯一标识符ID。
        :param to_node_id: 弧的结束节点的唯一标识符ID。
        :param travel_cost: 沿着这条弧旅行的成本（例如，距离、燃料成本）。
        :param travel_time: 沿着这条弧旅行所需的时间。
        """
        # 参数校验
        if not isinstance(from_node_id, int) or from_node_id < 0:
            raise ValueError("起始节点ID必须是非负整数。")
        if not isinstance(to_node_id, int) or to_node_id < 0:
            raise ValueError("结束节点ID必须是非负整数。")
        if from_node_id == to_node_id:
            raise ValueError("弧的起始节点和结束节点不能相同。")
        if not isinstance(travel_cost, (int, float)) or travel_cost < 0:
            raise ValueError("旅行成本必须是非负数值。")
        if not isinstance(travel_time, (int, float)) or travel_time < 0:
            raise ValueError("旅行时间必须是非负数值。")

        self.from_node_id: int = from_node_id
        self.to_node_id: int = to_node_id
        self.travel_cost: Union[int, float] = travel_cost
        self.travel_time: Union[int, float] = travel_time

    @property
    def key(self) -> Tuple[int, int]:
        """
        返回弧的唯一键，通常是 (from_node_id, to_node_id) 元组。
        这有助于在字典中作为键使用或进行快速查找。
        """
        return (self.from_node_id, self.to_node_id)

    def __str__(self) -> str:
        """返回弧的字符串表示，便于打印。"""
        return (
            f"弧: {self.from_node_id} -> {self.to_node_id}\n"
            f"  旅行成本: {self.travel_cost:.2f}\n"
            f"  旅行时间: {self.travel_time:.2f}"
        )

    def __repr__(self) -> str:
        """返回弧的官方字符串表示，便于调试和重建对象。"""
        return (
            f"Arc(from_node_id={self.from_node_id}, to_node_id={self.to_node_id}, "
            f"travel_cost={self.travel_cost}, travel_time={self.travel_time})"
        )

    def __eq__(self, other) -> bool:
        """定义弧相等性，基于起点ID和终点ID。"""
        if not isinstance(other, Arc):
            return NotImplemented
        return self.from_node_id == other.from_node_id and \
               self.to_node_id == other.to_node_id

    def __hash__(self) -> int:
        """允许将Arc对象放入集合或作为字典键，基于其唯一键。"""
        return hash(self.key)


class VRPTWProblem:
    """
    VRPTW (Vehicle Routing Problem with Time Windows) data container class.
    Its sole responsibility is to store all relevant problem instance data
    and provide it in formats convenient for solvers.
    """
    def __init__(self,
                problem_name: str,
                vehicles: List[Vehicle],
                nodes: List[Node], # List of all nodes (depot + customers), depot should be nodes[0]
                arcs: List[Arc]
                ):
        """
        Initializes the VRPTW problem data instance.

        :param problem_name: The name of the problem.
        :param vehicles: A list of Vehicle objects.
        :param nodes: A list of Node objects representing all locations. The depot
                      Node must be the first element (index 0) in this list.
        :param arcs: A list of Arc objects representing all possible travel segments.
        """
        # --- Input Validation ---
        if not problem_name:
            raise ValueError("Problem name cannot be empty.")
        if not vehicles:
            raise ValueError("Vehicle list cannot be empty.")
        if not nodes:
            raise ValueError("Nodes list cannot be empty.")
        if not arcs:
            raise ValueError("Arcs list cannot be empty.")
        if not isinstance(nodes[0], Node) or nodes[0].id != 0 or not nodes[0].is_depot:
             raise ValueError("The first node in the 'nodes' list must be the depot (ID 0).")

        # --- Internal Storage ---
        self.problem_name: str = problem_name
        self._vehicles: List[Vehicle] = vehicles
        self._max_vehicle = len(self._vehicles)
        self._min_vehicle = 0
        self._nodes: List[Node] = nodes
        self._node_num = len(self._nodes)
        self._arcs: List[Arc] = arcs

        # --- Internal Lookup Dictionaries for Efficient Access ---
        self._depot: Node = self._nodes[0] # Depot is always the first node
        self._node_by_id: Dict[int, Node] = {node.id: node for node in self._nodes}
        self._vehicle_by_id: Dict[int, Vehicle] = {vehicle.id: vehicle for vehicle in self._vehicles}
        self._arc_by_key: Dict[Tuple[int, int], Arc] = {arc.key: arc for arc in self._arcs}

        # --- Pre-compute Solver-Friendly Data Structures ---
        # These properties will return the data in the exact format the solver expects.
        self._precompute_solver_data()

        self.best_known_solution = Solution()

    def _precompute_solver_data(self):
        """
        Precomputes and caches common problem data in formats optimized for solver access.
        This runs once during initialization.
        """
        self._node_ids: List[int] = [node.id for node in self._nodes]
        self._customer_ids: List[int] = [node.id for node in self._nodes[1:]]
        self._customer_id_map_index = {
            cus_id: cus_idx for cus_idx, cus_id in enumerate(self._customer_ids)
        }
        self._customer_num = len(self._customer_ids)
        self._vehicle_ids: List[int] = [vehicle.id for vehicle in self._vehicles]
        self._depot_id: int = self._depot.id
        self._arc_keys: List[Tuple[int, int]] = [arc.key for arc in self._arcs] # (i,j) tuples

        # Dictionaries mapping node ID to its properties
        self._node_serve_window: Dict[int, Tuple[float, float]] = {node.id: node.time_window for node in self._nodes}
        self._demand: Dict[int, float] = {node.id: node.demand for node in self._nodes}
        self._node_serve_time: Dict[int, float] = {node.id: node.service_time for node in self._nodes}

        # Dictionary mapping vehicle ID to its capacity
        self._vehicle_max_cap: Dict[int, float] = {vehicle.id: vehicle.capacity for vehicle in self._vehicles}
        self._vehicle_globel_max_cap = max(self._vehicle_max_cap.values())
        # Dictionaries mapping (from_node_id, to_node_id) tuple to arc properties
        self._travel_time: Dict[Tuple[int, int], float] = {arc.key: arc.travel_time for arc in self._arcs}
        self._travel_cost: Dict[Tuple[int, int], float] = {arc.key: arc.travel_cost for arc in self._arcs}

        # Adjacency lists for incoming and outgoing nodes
        self._out_node_ids: Dict[int, List[int]] = {node.id: [] for node in self._nodes}
        self._in_node_ids: Dict[int, List[int]] = {node.id: [] for node in self._nodes}
        for arc in self._arcs:
            self._out_node_ids[arc.from_node_id].append(arc.to_node_id)
            self._in_node_ids[arc.to_node_id].append(arc.from_node_id)
        
        self._min_ready_time = min(node.ready_time for node in self._nodes)
        self._max_due_time = max(node.due_time for node in self._nodes)

    # --- Properties for Solver Access ---
    # These mimic the exact attribute names expected by your VRPArcflowSolver
    @property
    def node_ids(self) -> List[int]:
        """Returns a list of all node IDs (depot + customers)."""
        return self._node_ids

    @property
    def node_num(self) -> List[int]:
        """Returns number of node IDs (depot + customers)."""
        return self._node_num

    @property
    def customer_ids(self) -> List[int]:
        """Returns a list of all customer IDs (excluding depot)."""
        return self._customer_ids
    
    @property
    def customer_num(self):
        return self._customer_num

    @property
    def vehicle_ids(self) -> List[int]:
        """Returns a list of all vehicle IDs."""
        return self._vehicle_ids

    @property
    def max_vehicle(self) -> int:
        """Returns ."""
        return self._max_vehicle

    @property
    def min_vehicle(self) -> int:
        """Returns ."""
        return self._min_vehicle

    @property
    def depot_id(self) -> int:
        """Returns the ID of the depot node."""
        return self._depot_id

    @property
    def arcs(self) -> List[Tuple[int, int]]:
        """Returns a list of (from_node_id, to_node_id) tuples for all arcs."""
        return self._arc_keys

    @property
    def node_serve_window(self) -> Dict[int, Tuple[float, float]]:
        """Returns a dictionary mapping node ID to its (ready_time, due_time) tuple."""
        return self._node_serve_window

    @property
    def demand(self) -> Dict[int, float]:
        """Returns a dictionary mapping node ID to its demand."""
        return self._demand

    @property
    def vehicle_max_cap(self) -> Dict[int, float]:
        """Returns a dictionary mapping vehicle ID to its maximum capacity."""
        return self._vehicle_max_cap

    @property
    def vehicle_globel_max_cap(self) -> Dict[int, float]:
        """Returns a dictionary mapping vehicle ID to its maximum capacity."""
        return self._vehicle_globel_max_cap

    @property
    def node_serve_time(self) -> Dict[int, float]:
        """Returns a dictionary mapping node ID to its service time."""
        return self._node_serve_time

    @property
    def travel_time(self) -> Dict[Tuple[int, int], float]:
        """Returns a dictionary mapping (from_id, to_id) tuple to travel time."""
        return self._travel_time

    @property
    def travel_cost(self) -> Dict[Tuple[int, int], float]:
        """Returns a dictionary mapping (from_id, to_id) tuple to travel cost."""
        return self._travel_cost

    @property
    def out_node_ids(self) -> Dict[int, List[int]]:
        """Returns a dictionary mapping node ID to a list of IDs of nodes it can directly reach."""
        return self._out_node_ids

    @property
    def in_node_ids(self) -> Dict[int, List[int]]:
        """Returns a dictionary mapping node ID to a list of IDs of nodes that can directly reach it."""
        return self._in_node_ids

    @property
    def max_due_time(self) -> float:
        return self._max_due_time
    
    @property
    def min_ready_time(self) -> float:
        return self._min_ready_time

    # --- Other Utility Methods (for general problem data access) ---
    def get_node_by_id(self, node_id: int) -> Node:
        """Retrieves a Node object by its ID."""
        node = self._node_by_id.get(node_id)
        if node is None:
            raise ValueError(f"Node ID {node_id} does not exist.")
        return node

    def get_vehicle_by_id(self, vehicle_id: int) -> Vehicle:
        """Retrieves a Vehicle object by its ID."""
        vehicle = self._vehicle_by_id.get(vehicle_id)
        if vehicle is None:
            raise ValueError(f"Vehicle ID {vehicle_id} does not exist.")
        return vehicle

    def get_arc_object(self, from_id: int, to_id: int) -> Arc:
        """Retrieves an Arc object by its from/to node IDs."""
        arc = self._arc_by_key.get((from_id, to_id))
        if arc is None:
            raise ValueError(f"Arc from {from_id} to {to_id} does not exist.")
        return arc
    
    def get_customer_index(self, customer_id):
        return self._customer_id_map_index[customer_id]

    def total_demand(self) -> float:
        """Calculates and returns the total demand of all customers."""
        return sum(node.demand for node in self._nodes if not node.is_depot)

    def __str__(self) -> str:
        """Returns a summary string of the problem instance."""
        depot_node = self.get_node_by_id(self.depot_id)
        return (
            f"--- VRPTW Problem Summary ---\n"
            f"Problem Name: {self.problem_name}\n"
            f"Number of Vehicles: {len(self._vehicles)}, Avg. Capacity: {self.vehicle_max_cap.get(self.vehicle_ids[0], 0.0):.2f}\n"
            f"Total Nodes: {len(self._nodes)} (including depot)\n"
            f"Total Customer Demand: {self.total_demand():.2f}\n"
            f"Depot (ID: {depot_node.id}) Location: ({depot_node.x:.2f}, {depot_node.y:.2f})\n"
            f"-----------------------------"
        )
    def validate_solution_time_windows(self, solution: Solution) -> Tuple[bool, str]:
        """
        验证解决方案的时间窗约束是否满足。
        
        对于每个节点的服务开始时间，检查：
        1. 是否在时间窗内 [ready_time, due_time]
        2. 是否满足连续性约束：当前节点的服务开始时间 >= max(
            上一个节点的服务开始时间 + 服务时间 + 旅行时间,
            当前节点的ready_time
        )
        
        返回: (验证结果, 错误信息)
        """
        if not solution.solved:
            return False, "Solution is not solved"
        
        # 遍历每辆车的路径
        for vehicle_id, route in solution.routes.items():
            arrivals = solution.arrival_times[vehicle_id]
            
            # 遍历路径中的每个节点（从第一个节点开始）
            for i in range(len(route)):
                node_id = route[i]
                arrival_time = arrivals[i]
                
                # 获取节点的时间窗和服务时间
                ready_time, due_time = self.node_serve_window[node_id]
                service_time = self.node_serve_time[node_id]
                
                # 1. 检查时间窗约束
                if arrival_time < ready_time:
                    return False, (f"Vehicle {vehicle_id} at node {node_id}: "
                                  f"Arrival time {arrival_time} < ready time {ready_time}")
                
                if arrival_time > due_time:
                    return False, (f"Vehicle {vehicle_id} at node {node_id}: "
                                  f"Arrival time {arrival_time} > due time {due_time}")
                
                # 2. 检查连续性约束（从第二个节点开始）
                if i > 0:
                    prev_node_id = route[i-1]
                    prev_arrival = arrivals[i-1]
                    prev_service_time = self.node_serve_time[prev_node_id]
                    
                    # 获取从前一个节点到当前节点的旅行时间
                    try:
                        travel_time = self.travel_time[(prev_node_id, node_id)]
                    except KeyError:
                        return False, (f"Missing arc from {prev_node_id} to {node_id} "
                                      f"for vehicle {vehicle_id}")
                    
                    # 计算最小允许到达时间
                    min_arrival = max(
                        prev_arrival + prev_service_time + travel_time,
                        ready_time
                    )
                    
                    # 检查实际到达时间是否满足约束
                    if arrival_time < min_arrival:
                        print(
                            f"Vehicle {vehicle_id} at node {node_id}: "
                            f"Arrival time {arrival_time} < required min arrival time {min_arrival}\n"
                            f"  Breakdown: prev_arrival={prev_arrival}, "
                            f"prev_service={prev_service_time}, "
                            f"travel_time={travel_time}, "
                            f"ready_time={ready_time}"
                        )
                        return False, (
                            f"Vehicle {vehicle_id} at node {node_id}: "
                            f"Arrival time {arrival_time} < required min arrival time {min_arrival}\n"
                            f"  Breakdown: prev_arrival={prev_arrival}, "
                            f"prev_service={prev_service_time}, "
                            f"travel_time={travel_time}, "
                            f"ready_time={ready_time}"
                        )

        return True, "All time windows and continuity constraints are satisfied"
    
    def validate_and_print_solution_details(self, solution: Solution) -> Tuple[bool, str]:
        """
        验证解决方案并打印详细路径信息，包括：
        - 每个节点的时间窗
        - 服务时间
        - 与上个点的距离
        - 实际服务时间
        - 时间窗约束检查
        
        返回: (验证结果, 错误信息)
        """
        if not solution.solved:
            print("解决方案未求解")
            return False, "Solution is not solved"
        
        all_valid = True
        error_messages = []
        
        print(f"\n{'='*50}")
        print(f"解决方案详细分析: {self.problem_name}")
        print(f"使用车辆数: {solution.num_vehicles}")
        print(f"总行驶成本: {solution.travel_cost:.2f}")
        print(f"{'='*50}\n")
        
        # 遍历每辆车的路径
        for vehicle_id, route in solution.routes.items():
            arrivals = solution.arrival_times[vehicle_id]
            valid_route = True
            
            print(f"\n{'='*30}")
            print(f"车辆 {vehicle_id} 路径分析:")
            print(f"路径: {' -> '.join(map(str, route))}")
            print(f"到达时间: {arrivals}")
            print(f"{'-'*30}")
            
            # 打印表头
            print(f"{'节点':<5} | {'时间窗':<15} | {'服务时间':<8} | {'距上点距离':<10} | {'实际到达':<8} | {'状态'}")
            print(f"{'-'*5}|{'-'*15}|{'-'*8}|{'-'*10}|{'-'*8}|{'-'*15}")
            
            # 遍历路径中的每个节点
            for i in range(len(route)):
                node_id = route[i]
                arrival_time = arrivals[i]
                
                # 获取节点属性
                ready_time, due_time = self.node_serve_window[node_id]
                service_time = self.node_serve_time[node_id]
                
                # 计算与上个点的距离（如果是第一个节点则为0）
                prev_distance = 0.0
                if i > 0:
                    prev_node_id = route[i-1]
                    try:
                        prev_distance = self.travel_cost.get((prev_node_id, node_id), 0.0)
                    except KeyError:
                        prev_distance = -1.0  # 表示缺失数据
                
                # 检查时间窗约束
                time_window_valid = True
                status = "OK"
                
                if arrival_time < ready_time:
                    status = f"EARLY ({arrival_time} < {ready_time})"
                    time_window_valid = False
                elif arrival_time > due_time:
                    status = f"LATE ({arrival_time} > {due_time})"
                    time_window_valid = False
                
                # 检查连续性约束（从第二个节点开始）
                continuity_valid = True
                if i > 0:
                    prev_node_id = route[i-1]
                    prev_arrival = arrivals[i-1]
                    prev_service_time = self.node_serve_time[prev_node_id]
                    
                    # 获取从前一个节点到当前节点的旅行时间
                    try:
                        travel_time = self.travel_time.get((prev_node_id, node_id), 0.0)
                    except KeyError:
                        travel_time = 0.0
                    
                    # 计算最小允许到达时间
                    min_arrival = round(
                        max(
                        prev_arrival + prev_service_time + travel_time,
                        ready_time
                    ),5)
                    
                    if arrival_time < min_arrival:
                        continuity_valid = False
                        status = f"CONTINUITY ERROR (min: {min_arrival:.1f})"
                
                # 更新路径有效性
                if not (time_window_valid and continuity_valid):
                    valid_route = False
                    all_valid = False
                
                # 打印节点详情
                node_type = "Depot" if node_id == self.depot_id else f"C{node_id}"
                time_window_str = f"[{ready_time:.1f}, {due_time:.1f}]"
                
                print(f"{node_type:<5} | {time_window_str:<15} | {service_time:<8.1f} | "
                      f"{prev_distance:<10.1f} | {arrival_time:<8.1f} | {status}")
            
            # 打印路径总结
            if valid_route:
                print(f"{'-'*30}\n路径 {vehicle_id} 有效 ✓")
            else:
                print(f"{'-'*30}\n路径 {vehicle_id} 存在时间窗问题 ✗")
                error_messages.append(f"车辆 {vehicle_id} 存在时间窗问题")
        
        # 整体解决方案验证
        if all_valid:
            print(f"\n{'='*50}")
            print("所有路径时间窗约束均满足 ✓")
            print(f"{'='*50}")
            return True, "所有时间窗约束均满足"
        else:
            print(f"\n{'='*50}")
            print("解决方案存在时间窗问题 ✗")
            print(f"问题: {', '.join(error_messages)}")
            print(f"{'='*50}")
            return False, "; ".join(error_messages)

# 使用示例
if __name__ == "__main__":
    # 创建问题实例
    problem = VRPTWProblem("C101_first_25")
    
    # 从文件加载数据
    problem.load_from_file("C101_first_25.txt")
    
    # 打印问题摘要
    print(problem)
    
    # 获取仓库信息
    depot = problem.get_depot()
    print(f"\n仓库信息: {depot}")
    
    # 获取客户信息示例
    customer_5 = problem.get_customer(5)
    print(f"\n客户5信息: {customer_5}")
    
    # 获取距离示例
    distance = problem.get_distance(1, 2)
    print(f"\n客户1到客户2的距离: {distance:.2f}")
    
    # 验证解决方案示例
    sample_solution = [
        [0, 1, 2, 3, 4, 0],  # 路径1
        [0, 5, 6, 7, 8, 0],  # 路径2
        [0, 9, 10, 11, 12, 0]  # 路径3
    ]
    is_valid, message = problem.validate_solution(sample_solution)
    print(f"\n解决方案验证: {is_valid}, {message}")