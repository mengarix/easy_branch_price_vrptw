import os
from src.instance_parser.standard_parser import StandardTxtParser
from src.problem.problem import VRPTWProblem, Node, Arc, Vehicle, calculate_euclidean_distance

def txt2problem(file_path, problem_id, round_bit=1):
    max_vehicle, vehicle_capacity, depot, customers = StandardTxtParser.load_from_file(file_path)
    vehicles = [
        Vehicle(id=i, capacity=vehicle_capacity)
        for i in range(1, max_vehicle+1)
    ]

    nodes = [
        Node(id=depot['id'], x=depot['x'], y=depot['y'], demand=depot['demand'], ready_time=depot['ready_time'], due_time=depot['due_date'],service_time=depot['service_time'], is_depot=True)
    ] + [
        Node(id=c['id'], x=c['x'], y=c['y'], demand=c['demand'], ready_time=c['ready_time'], due_time=c['due_date'], service_time=c['service_time'],is_depot=False)
        for c in customers
    ]

    arcs = []
    for i in nodes:
        for j in nodes:
            if i.id == j.id: continue
            dist = calculate_euclidean_distance(i.coordinates, j.coordinates, round_bit)
            arcs.append(Arc(i.id, j.id, travel_cost=dist, travel_time=dist)) 

    problem = VRPTWProblem(
        problem_name=problem_id,
        vehicles=vehicles,
        nodes=nodes,
        arcs=arcs
    )
    return problem


def solomon_parser(customer_num, instance_id, round_bit=1):
    instance_folder_path = './instance/solomon'
    file_path = os.path.join(instance_folder_path, str(customer_num), f'{instance_id}.txt')
    problem_id = f'{instance_id}.{customer_num}'
    problem = txt2problem(file_path, problem_id, round_bit)
    return problem


def homberger_parser(customer_num, instance_id, round_bit=1):
    instance_folder_path = './instance/homberger'
    file_path = os.path.join(instance_folder_path, str(customer_num), f'{instance_id}.TXT')
    problem_id = f'{instance_id}'
    problem = txt2problem(file_path, problem_id, round_bit)
    return problem