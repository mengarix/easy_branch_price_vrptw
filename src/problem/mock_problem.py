from src.problem.problem import Node, Vehicle, Arc, VRPTWProblem, calculate_euclidean_distance

print("--- Starting VRPTW Arcflow` Model Test Case ---")

# 1. Define a simple problem instance
# Depot at (0,0), 2 customers, 2 vehicles
nodes = [
    Node(id=0, x=0, y=4, demand=0, ready_time=0, due_time=40, service_time=0, is_depot=True), # Depot
    Node(id=1, x=1, y=1, demand=1, ready_time=0, due_time=40, service_time=0), # Customer 1
    Node(id=2, x=3, y=3, demand=1, ready_time=0, due_time=40, service_time=0),  # Customer 2
    Node(id=3, x=2, y=2, demand=1, ready_time=0, due_time=40, service_time=0)  # Customer 2
]

vehicles = [
    Vehicle(id=0, capacity=500), # Vehicle 0 can handle both customers
    Vehicle(id=1, capacity=500)  # Vehicle 1 can also handle both customers
]

# Generate all possible arcs and calculate travel time/cost (Euclidean distance)
arcs = []
arcs = [
    Arc(0, 1, travel_cost=1, travel_time=1),
    Arc(0, 2, travel_cost=1.4, travel_time=1),
    Arc(0, 3, travel_cost=1, travel_time=1),
    Arc(1, 2, travel_cost=1, travel_time=1),
    Arc(1, 3, travel_cost=1.4, travel_time=1.4),
    Arc(2, 3, travel_cost=1, travel_time=1),
    Arc(1, 0, travel_cost=1, travel_time=1),
    Arc(2, 0, travel_cost=1.4, travel_time=1),
    Arc(3, 0, travel_cost=1, travel_time=1),
    Arc(2, 1, travel_cost=1, travel_time=1),
    Arc(3, 1, travel_cost=1.4, travel_time=1.4),
    Arc(3, 2, travel_cost=1, travel_time=1),
]
# for i in nodes:
#     for j in nodes:
#         if i.id == j.id: continue
#         dist = calculate_euclidean_distance(i.coordinates, j.coordinates)
#         arcs.append(Arc(i.id, j.id, travel_cost=dist, travel_time=dist)) # Assuming cost = time

mock_problem = VRPTWProblem(
    problem_name="Simple_Test_Case",
    vehicles=vehicles,
    nodes=nodes,
    arcs=arcs
)

print(f"\nProblem Definition:\n{mock_problem}")
print(f"Nodes: {mock_problem.node_ids}")
print(f"Customers: {mock_problem.customer_ids}")
print(f"Vehicles: {mock_problem.vehicle_ids}")
print(f"Depot: {mock_problem.depot_id}")
print(f"Node Serve Times: {mock_problem.node_serve_time}")
print(f"Demands: {mock_problem.demand}")
print(f"Vehicle Capacities: {mock_problem.vehicle_max_cap}")
print(f"Travel Times (selected): {(0,1)}:{mock_problem.travel_time.get((0,1)):.2f}, {(1,0)}:{mock_problem.travel_time.get((1,0)):.2f}")
print(f"Max Due Time: {mock_problem.max_due_time}")


nodes = [
    Node(id=0, x=0, y=0, demand=0, ready_time=0, due_time=100, service_time=0, is_depot=True), # Depot
    Node(id=1, x=10, y=0, demand=20, ready_time=10, due_time=40, service_time=5), # Customer 1
    Node(id=2, x=0, y=10, demand=30, ready_time=20, due_time=50, service_time=5)  # Customer 2
]

vehicles = [
    Vehicle(id=0, capacity=50), # Vehicle 0 can handle both customers
    Vehicle(id=1, capacity=50)  # Vehicle 1 can also handle both customers
]

# Generate all possible arcs and calculate travel time/cost (Euclidean distance)
arcs = []
for i in nodes:
    for j in nodes:
        if i.id == j.id: continue
        dist = calculate_euclidean_distance(i.coordinates, j.coordinates)
        arcs.append(Arc(i.id, j.id, travel_cost=dist, travel_time=dist)) # Assuming cost = time

mock_problem2 = VRPTWProblem(
    problem_name="Simple_Test_Case",
    vehicles=vehicles,
    nodes=nodes,
    arcs=arcs
)