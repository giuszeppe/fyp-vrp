
from calendar import c
from turtle import distance

from sympy import O

from dvrptw_bench.common.typing import Node, VRPTWInstance
from dvrptw_bench.data.normalization import distance_matrix as compute_distance_matrix
from dvrptw_bench.dynamic.snapshot import SnapshotState, VehicleState


class DynamicInstance:
    depot: Node
    starts: list[Node]
    ends: list[Node]
    distance_matrix: list[list[float]]
    complete_set_of_nodes: list[Node]
    customers: list[Node]
    vehicles :list[VehicleState]
    instance_id: str
    customers: list[Node]
    vehicle_count: int
    vehicle_capacity: float
    distance_matrix: list[list[float]]
    vehicles: list[VehicleState]
    all_nodes: list[Node]
    n_customers: int



    def __init__(self, snapshot : SnapshotState, base_instance : VRPTWInstance):
        self.instance_id=base_instance.instance_id
        self.customers=[customer if customer.id in [id for id in snapshot.remaining_customers] else Node(id=customer.id, x=customer.x, y=customer.y, demand=0, ready_time=customer.ready_time, due_time=customer.due_time, service_time=customer.service_time) for customer in base_instance.customers]
        # print("Customers in snapshot:", [c.id for c in snapshot.remaining_customers])
        self.ignored_customers = [customer.id for customer in base_instance.customers if customer.id not in [id.id for id in snapshot.remaining_customers]]
        # print("Ignored customers:", [id for id in self.ignored_customers])
        self.depot=base_instance.depot
        self.vehicle_count=base_instance.vehicle_count
        self.vehicle_capacity=base_instance.vehicle_capacity
        self.vehicles = snapshot.vehicles
        self.all_nodes = [self.depot] + self.customers
        self.n_customers = len(self.customers)
        # first add vehicles that are in transit as "fake" depot, with demand 0 and time windows corresponding to their current position and elapsed time
        self.starts = []
        self.ends = [self.depot] * self.vehicle_count  # for simplicity we can just assume all vehicles have to return to the depot, but this can be easily modified to allow for different end nodes if needed
        for vehicle in snapshot.vehicles:
            isAtDepot = vehicle.x == self.depot.x and vehicle.y == self.depot.y
            customerBeingServed = list(filter(lambda c: c.x == vehicle.x and c.y == vehicle.y, snapshot.remaining_customers))
            customerBeingServed = customerBeingServed[0] if customerBeingServed else None

            ready_time = vehicle.elapsed_time + vehicle.remaining_service_time if vehicle.remaining_service_time > 0 else vehicle.elapsed_time
            newNodeId = max([n.id for n in self.starts] + [n.id for n in base_instance.customers]) + 1
            nodeId = newNodeId
            if isAtDepot:
                nodeId = 0  # if the vehicle is still at the depot, we can just use the depot as its start node
                ready_time = base_instance.depot.ready_time
            elif customerBeingServed is not None:
                nodeId = customerBeingServed.id  # if the vehicle is currently serving a customer, we can use that customer as its start node, with an updated ready time
            else:
                self.all_nodes.append(Node(
                    id=nodeId,
                    x=vehicle.x,
                    y=vehicle.y,
                    demand=0.0,
                    ready_time=ready_time,
                    due_time=base_instance.depot.due_time,  # we can just use the depot's due time as the due time for the start node, since the vehicle has to return to the depot by then
                    service_time=0.0,  # we can assume that the vehicle doesn't need any service time at its current location, since it's already there
                ))
            
            self.starts.append(Node(
                id=nodeId,
                x=vehicle.x,
                y=vehicle.y,
                demand=0.0,
                ready_time=ready_time,
                due_time=base_instance.depot.due_time,  
                service_time=0.0,   
            ))
        

        self.distance_matrix=compute_distance_matrix(self.all_nodes)
        pass
