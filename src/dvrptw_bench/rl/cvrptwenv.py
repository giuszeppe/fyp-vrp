from typing import Optional

from rl4co.envs.routing.cvrptw.env import CVRPEnv, CVRPTWEnv, gather_by_index, get_distance
from rl4co.envs.routing import MTVRPGenerator
from tensordict import TensorDict
import torch

from dvrptw_bench.rl.cvrptw_generator import CVRPTWGeneratorFixed
class CVRPTWFixed(CVRPTWEnv):
    def __init__(self, generator = None, generator_params = None, **kwargs):
        if generator is None:
            generator=MTVRPGenerator(**(generator_params or {}))
            # generator = CVRPTWGeneratorFixed(**(generator_params or {}))
        super().__init__(generator, generator_params, **kwargs)

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        if td is None:
            return super()._reset(td, batch_size)
        if batch_size is None:
            batch_size = [1]
        device = td.device
        td_reset = TensorDict(
            {
                "locs": torch.cat((td["depot"][..., None, :], td["locs"]), -2),
                "demand": td["demand"],
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "current_time": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=device
                ),
                "used_capacity": torch.zeros((*batch_size, 1), device=device),
                "vehicle_capacity": torch.full(
                    (*batch_size, 1), self.generator.vehicle_capacity, device=device
                ),
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2] + 1),
                    dtype=torch.uint8,
                    device=device,
                ),
                "durations": td["durations"],
                "time_windows": td["time_windows"],
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset
    # @staticmethod
    # def get_action_mask(td: TensorDict) -> torch.Tensor:
        """In addition to the constraints considered in the CVRPEnv, the time windows are considered.
        The vehicle can only visit a location if it can reach it in time, i.e. before its time window ends.
        """
        # not_masked = CVRPTWFixed.get_action_mask_cvrp(td)
        # current_loc = gather_by_index(td["locs"], td["current_node"])
        # dist = get_distance(current_loc[..., None, :], td["locs"])
        # td.update({"current_loc": current_loc, "distances": dist})
        # can_reach_in_time = (
        #     td["current_time"] + dist <= td["time_windows"][..., 1]
        # )  # I only need to start the service before the time window ends, not finish it.
        # return not_masked & can_reach_in_time
    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """
        Returns a mask of feasible actions for CVRPTW.
        1 = feasible, 0 = infeasible

        Rules:
            - Already visited customers cannot be selected
            - Vehicle cannot exceed capacity
            - Vehicle must start service before customer's time window closes
            - Depot is always allowed
        """
        batch_size, num_nodes, _ = td["locs"].shape

        # Start from CVRP mask (visited + capacity)
        mask = CVRPEnv.get_action_mask(td)  # shape: [batch_size, num_nodes]

        # Current vehicle location
        current_loc = gather_by_index(td["locs"], td["current_node"])  # [batch_size, 2]
        current_time = td["current_time"]  # [batch_size, 1]

        # Distances to all nodes
        distances = get_distance(current_loc[..., None, :], td["locs"])  # [batch_size, num_nodes]

        # Service durations
        durations = td["durations"].reshape(batch_size, num_nodes, 1)  # [batch_size, num_nodes, 1]

        # Time windows
        tw_start = td["time_windows"][..., 0]  # [batch_size, num_nodes]
        tw_end = td["time_windows"][..., 1]    # [batch_size, num_nodes]

        # Compute earliest start if we go to each customer now
        earliest_start = torch.max(current_time + distances, tw_start)  # [batch_size, num_nodes]

        # Cannot start service after time window ends
        feasible_time = earliest_start <= tw_end


        # Combine with existing mask
        mask = mask & feasible_time

        # Save extra info for efficiency
        td.update({
            "current_loc": current_loc,
            "distances": distances,
        })

        return mask
    @staticmethod
    def get_action_mask_cvrp(td: TensorDict) -> torch.Tensor:
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = td["demand"] + td["used_capacity"] > td["vehicle_capacity"] + 1e-5

        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = td["visited"][..., 1:].to(exceeds_cap.dtype) | exceeds_cap

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (td["current_node"] == 0) & ((mask_loc == 0).int().sum(-1) > 0)[
            :, None
        ]
        return ~torch.cat((mask_depot, mask_loc), -1)
    
    def extract_from_solomon(self, instance: dict, batch_size: int = 1):
        # extract parameters for the environment from the Solomon instance
        self.min_demand = instance["demand"][1:].min()
        self.max_demand = instance["demand"][1:].max()
        self.vehicle_capacity = instance["capacity"]
        self.min_loc = instance["node_coord"][1:].min()
        self.max_loc = instance["node_coord"][1:].max()
        self.min_time = instance["time_window"][:, 0].min()
        self.max_time = instance["time_window"][:, 1].max()
        # assert the time window of the depot starts at 0 and ends at max_time
        assert self.min_time == 0, "Time window of depot must start at 0."
        assert (
            self.max_time == instance["time_window"][0, 1]
        ), "Depot must have latest end time."
        # convert to format used in CVRPTWEnv
        td = TensorDict(
            {
                "depot": torch.tensor(
                    instance["node_coord"][0],
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(batch_size, 1),
                "locs": torch.tensor(
                    instance["node_coord"][1:],
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(batch_size, 1, 1),
                "demand": torch.tensor(
                    instance["demand"][1:] / self.vehicle_capacity,
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(batch_size, 1),
                "durations": torch.tensor(
                    instance["service_time"],
                    dtype=torch.int64,
                    device=self.device,
                ).repeat(batch_size, 1),
                "time_windows": torch.tensor(
                    instance["time_window"],
                    dtype=torch.int64,
                    device=self.device,
                ).repeat(batch_size, 1, 1),
            },
            batch_size=1,  # we assume batch_size will always be 1 for loaded instances
        )
        return self.reset(td, batch_size=batch_size)
