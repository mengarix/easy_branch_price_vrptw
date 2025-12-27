

class Solution:
    def __init__(self):
        self.routes = {}          # 车辆路径：{vehicle_id: [node_id1, node_id2, ...]}
        self.arrival_times = {}    # 到达时间：{vehicle_id: [arrival_time_at_node1, ...]}
        self.total_cost = -1  # 总成本（车辆使用成本 + 行驶成本）
        self.travel_cost = -1   # 总行驶成本
        self.num_vehicles = -1      # 使用的车辆数
        self.solved = False        # 是否成功求解
        self.optimal = False
        self.solve_time = -1

    def __str__(self):
        if not self.solved:
            return "No solution found"
        
        output = [f"(Vehicles: {self.num_vehicles}, Travel: {self.travel_cost:.2f})"]
        for vid, route in self.routes.items():
            # route_str = " -> ".join(f"{node}({times[i]:.1f})" for i, node in enumerate(route))
            route_str = " -> ".join(f"{node}" for i, node in enumerate(route))
            output.append(f"Vehicle {vid}: {route_str}")
        return "\n".join(output)
    
    def get_summary(self):
        return {
            'solved': self.solved,
            'optimal': self.optimal,
            'num_vehicles': self.num_vehicles,
            'travel_cost': self.travel_cost, 
            'routes': self.routes,
            'solve_time': self.solve_time
        }

