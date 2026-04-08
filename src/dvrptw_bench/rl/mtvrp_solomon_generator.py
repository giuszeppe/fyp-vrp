from tensordict import TensorDict
from routefinder.envs.mtvrp import MTVRPGenerator
import torch


# class SolomonMTVRPGenerator(MTVRPGenerator):
#     def __init__(self, solomon_instances, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         assert len(solomon_instances) > 0, "No Solomon instances provided"
#         self.solomon_instances = solomon_instances

#     def _get_batch_size_int(self, batch_size):
#         if isinstance(batch_size, (tuple, list, torch.Size)):
#             assert len(batch_size) == 1, f"Expected 1D batch size, got {batch_size}"
#             return batch_size[0]
#         return batch_size
    
# class SolomonMTVRPGenerator(MTVRPGenerator):
#     def __init__(
#         self,
#         solomon_instances,
#         tw_start_noise_std=0.05,
#         tw_width_noise_std=0.08,
#         tw_center_noise_std=0.05,
#         service_noise_std=0.05,
#         min_width=0.05,
#         preserve_depot=True,
#         *args,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         assert len(solomon_instances) > 0, "No Solomon instances provided"

#         self.solomon_instances = solomon_instances
#         self.tw_start_noise_std = tw_start_noise_std
#         self.tw_width_noise_std = tw_width_noise_std
#         self.tw_center_noise_std = tw_center_noise_std
#         self.service_noise_std = service_noise_std
#         self.min_width = min_width
#         self.preserve_depot = preserve_depot

#         self._current_batch_indices = None

#     def _get_batch_size_int(self, batch_size):
#         if isinstance(batch_size, (tuple, list, torch.Size)):
#             assert len(batch_size) == 1, f"Expected 1D batch size, got {batch_size}"
#             return batch_size[0]
#         return batch_size

#     def _sample_batch_indices(self, batch_size, device=None):
#         B = self._get_batch_size_int(batch_size)
#         return torch.randint(0, len(self.solomon_instances), (B,), device=device)

#     def _get_batch_items(self):
#         assert self._current_batch_indices is not None, "Batch indices not initialized"
#         return [self.solomon_instances[i] for i in self._current_batch_indices.tolist()]

#     def generate_locations(self, batch_size, num_loc):
#         batch = self._get_batch_items()

#         locs = torch.stack(
#             [item["locs"][: num_loc + 1] for item in batch],
#             dim=0,
#         )
#         return locs


#     def generate_time_windows(self, locs, speed):
#         batch = self._get_batch_items()

#         time_windows = torch.stack(
#             [item["time_windows"][: self.num_loc + 1] for item in batch],
#             dim=0,
#         ).clone()

#         service_time = torch.stack(
#             [item["service_time"][: self.num_loc + 1] for item in batch],
#             dim=0,
#         ).clone()

#         tw = time_windows[:, 1:, :]          # customers only
#         st = service_time[:, 1:]             # customers only

#         start = tw[:, :, 0]
#         end = tw[:, :, 1]
#         width = end - start
#         center = 0.5 * (start + end)

#         # multiplicative perturbation
#         center = center * (1.0 + self.tw_center_noise_std * torch.randn_like(center))
#         width = width * (1.0 + self.tw_width_noise_std * torch.randn_like(width))
#         st = st * (1.0 + self.service_noise_std * torch.randn_like(st))

#         width = width.clamp(min=self.min_width)
#         st = st.clamp(min=0.0)

#         # travel time from depot
#         d0 = torch.norm(locs[:, 1:] - locs[:, 0:1], dim=-1) / speed  # [B, N]
#         max_time = time_windows[:, 0, 1].unsqueeze(-1)               # [B, 1]

#         # latest feasible START time for a direct depot->i->depot trip
#         latest_start = max_time - st - d0

#         # rebuild from perturbed center/width
#         new_start = center - 0.5 * width
#         new_end = center + 0.5 * width

#         # enforce customer-level feasibility
#         # must be reachable before due time
#         new_end = torch.maximum(new_end, d0 + self.min_width)

#         # start cannot be later than latest feasible start
#         new_start = torch.minimum(new_start, latest_start)

#         # nonnegative start
#         new_start = torch.clamp(new_start, min=0.0)

#         # end must be at least start + min_width
#         new_end = torch.maximum(new_end, new_start + self.min_width)

#         # end cannot exceed depot horizon
#         new_end = torch.minimum(new_end, max_time)

#         # if start got pushed too far right, repair again
#         new_start = torch.minimum(new_start, new_end - self.min_width)
#         new_start = torch.clamp(new_start, min=0.0)

#         time_windows[:, 1:, 0] = new_start
#         time_windows[:, 1:, 1] = new_end
#         service_time[:, 1:] = st

#         # preserve depot
#         time_windows[:, 0, 0] = 0.0
#         time_windows[:, 0, 1] = self.max_time
#         service_time[:, 0] = 0.0

#         return time_windows, service_time

#     def _generate(self, batch_size) -> TensorDict:
#         self._current_batch_indices = self._sample_batch_indices(batch_size)

#         td = super()._generate(batch_size)

#         self._current_batch_indices = None
#         return td

class SolomonMTVRPGenerator(MTVRPGenerator):
    def __init__(self, solomon_instances, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subsample = False

        self.all_locs = torch.stack([x["locs"][: self.num_loc + 1] for x in solomon_instances])
        self.all_time_windows = torch.stack([x["time_windows"][: self.num_loc + 1] for x in solomon_instances])
        self.all_service_time = torch.stack([x["service_time"][: self.num_loc + 1] for x in solomon_instances])
        self.all_demand_linehaul = torch.stack([x["demand_linehaul"][: self.num_loc].float() for x in solomon_instances])
        self.all_vehicle_capacity = torch.tensor(
            [[float(x["vehicle_capacity"])] for x in solomon_instances],
            dtype=torch.float32,
        )

        self.num_instances = self.all_locs.shape[0]

    def perturb_time_windows(self, locs, time_windows, service_time):
        tw = time_windows.clone()
        st = service_time.clone()

        start = tw[:, 1:, 0]
        end = tw[:, 1:, 1]
        width = end - start

        width_factor = (1.0 + 0.03 * torch.randn_like(width)).clamp(0.8, 1.2)
        service_factor = (1.0 + 0.03 * torch.randn_like(st[:, 1:])).clamp(0.8, 1.2)

        new_width = (width * width_factor).clamp_min(0.05)
        new_service = (st[:, 1:] * service_factor).clamp_min(0.0)

        center = 0.5 * (start + end)
        new_start = center - 0.5 * new_width
        new_end = center + 0.5 * new_width

        depot_due = tw[:, 0, 1].unsqueeze(-1)
        new_start = new_start.clamp_min(0.0)
        new_end = torch.minimum(new_end, depot_due)
        new_end = torch.maximum(new_end, new_start + 0.05)

        tw[:, 1:, 0] = new_start
        tw[:, 1:, 1] = new_end
        st[:, 1:] = new_service

        tw[:, 0, 0] = 0.0
        tw[:, 0, 1] = self.max_time
        st[:, 0] = 0.0

        return tw, st
    
    def _generate(self, batch_size):
        if isinstance(batch_size, (tuple, list, torch.Size)):
            B = batch_size[0]
        else:
            B = batch_size

        idx = torch.randint(0, self.num_instances, (B,))

        locs = self.all_locs[idx].clone()
        time_windows = self.all_time_windows[idx].clone()
        service_time = self.all_service_time[idx].clone()
        demand_linehaul = self.all_demand_linehaul[idx].clone()
        vehicle_capacity = self.all_vehicle_capacity[idx].clone()
        capacity_original = vehicle_capacity.clone()

        # optional: small perturbation here, fully vectorized
        # time_windows, service_time = self.perturb_time_windows(locs, time_windows, service_time)

        demand_backhaul = torch.zeros_like(demand_linehaul)
        backhaul_class = self.generate_backhaul_class(shape=(B, 1), sample=self.sample_backhaul_class)
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