# cython:language_level=3
# -*- encoding: utf-8 -*-

# import package
import math
import os
from typing import List, Dict, Tuple


class StandardTxtParser:
    def __init__(self):
        pass
    
    @classmethod
    def load_from_file(cls, file_path: str):
        """
        从文件加载问题数据
        
        :param file_path: 文件路径
        :param num_customers: 要加载的客户数量（包括仓库），None表示加载所有
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # 解析车辆信息
        vehicle_number, vehicle_capacity = cls._parse_vehicle_info(lines)
        
        # 解析客户信息
        depot, customers = cls._parse_customer_info(lines)

        return vehicle_number, vehicle_capacity, depot, customers
        
    
    @classmethod
    def _parse_vehicle_info(cls, lines: List[str]):
        """解析车辆信息部分"""
        vehicle_number = None
        vehicle_capacity = None
        for i, line in enumerate(lines):
            if "VEHICLE" in line:
                # 找到车辆信息表头
                header_index = i
                # 跳过表头行
                data_line = lines[header_index + 2].split()
                
                if len(data_line) >= 2:
                    vehicle_number = int(data_line[0])
                    vehicle_capacity = int(data_line[1])
                break
        if vehicle_number is None:
            raise Exception('车辆信息读取失败')
        return vehicle_number, vehicle_capacity
    
    @classmethod
    def _parse_customer_info(cls, lines: List[str]):
        """解析客户信息部分"""
        depot = None
        customers = []
        coordinates = []
        # 查找客户信息起始位置
        start_index = -1
        for i, line in enumerate(lines):
            if "CUST NO." in line:
                start_index = i
                break
        
        if start_index == -1:
            raise ValueError("未找到客户信息标题行")
        
        # 处理客户数据
        customer_count = 0
        for line in lines[start_index + 1:]:
                
            data = line.split()
            if len(data) < 7:  # 确保有足够的数据列
                continue
            if int(data[0]) == 0:
                depot = {
                    'id': int(data[0]),
                    'x': int(data[1]),
                    'y': int(data[2]),
                    'demand': int(data[3]),
                    'ready_time': int(data[4]),
                    'due_date': int(data[5]),
                    'service_time': int(data[6])
                }
            else:
                customer = {
                    'id': int(data[0]),
                    'x': int(data[1]),
                    'y': int(data[2]),
                    'demand': int(data[3]),
                    'ready_time': int(data[4]),
                    'due_date': int(data[5]),
                    'service_time': int(data[6])
                }
                customers.append(customer)
                customer_count += 1
            
        return depot, customers
