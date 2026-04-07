from tensordict import TensorDict
from routefinder.envs.mtvrp import MTVRPGenerator
import torch


class SolomonMTVRPGenerator(MTVRPGenerator):
    def __init__(self, solomon_instances, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(solomon_instances) > 0, "No Solomon instances provided"
        self.solomon_instances = solomon_instances

    def _get_batch_size_int(self, batch_size):
        if isinstance(batch_size, (tuple, list, torch.Size)):
            assert len(batch_size) == 1, f"Expected 1D batch size, got {batch_size}"
            return batch_size[0]
        return batch_size

    def _generate(self, batch_size) -> TensorDict:
        B = self._get_batch_size_int(batch_size)

        idx = torch.randint(0, len(self.solomon_instances), (B,))
        batch = [self.solomon_instances[i] for i in idx.tolist()]

        locs = torch.stack(
            [item["locs"][: self.num_loc + 1] for item in batch], dim=0
        )  # [B, N+1, 2]

        time_windows = torch.stack(
            [item["time_windows"][: self.num_loc + 1] for item in batch], dim=0
        )  # [B, N+1, 2]

        service_time = torch.stack(
            [item["service_time"][: self.num_loc + 1] for item in batch], dim=0
        )  # [B, N+1]

        if all("demand_linehaul" in item for item in batch):
            demand_linehaul = torch.stack(
                [item["demand_linehaul"][: self.num_loc].float() for item in batch], dim=0
            )
        else:
            demand_linehaul, _ = super().generate_demands((B), self.num_loc)

        demand_backhaul = torch.zeros_like(demand_linehaul)

        if all("vehicle_capacity" in item for item in batch):
            vehicle_capacity = torch.tensor(
                [[float(item["vehicle_capacity"])] for item in batch],
                dtype=torch.float32,
            )
        else:
            vehicle_capacity = torch.full((B, 1), self.capacity, dtype=torch.float32)

        capacity_original = vehicle_capacity.clone()

        backhaul_class = self.generate_backhaul_class(
            shape=(B, 1), sample=self.sample_backhaul_class
        )
        open_route = self.generate_open_route(shape=(B, 1))
        speed = self.generate_speed(shape=(B, 1))
        distance_limit = self.generate_distance_limit(shape=(B, 1), locs=locs)

        if self.scale_demand:
            demand_linehaul = demand_linehaul / vehicle_capacity
            demand_backhaul = demand_backhaul / vehicle_capacity
            vehicle_capacity = vehicle_capacity / vehicle_capacity

        td = TensorDict(
            {
                "locs": locs,
                "demand_backhaul": demand_backhaul,
                "demand_linehaul": demand_linehaul,
                "backhaul_class": backhaul_class,
                "distance_limit": distance_limit,
                "time_windows": time_windows,
                "service_time": service_time,
                "vehicle_capacity": vehicle_capacity,
                "capacity_original": capacity_original,
                "open_route": open_route,
                "speed": speed,
            },
            batch_size=[B],
        )

        if self.subsample:
            td = self.subsample_problems(td)

        return td