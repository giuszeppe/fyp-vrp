from typing import Callable

import torch
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import get_sampler
from rl4co.envs.routing.cvrptw.env import CVRPTWGenerator
from rl4co.utils.ops import get_distance


class CVRPTWGeneratorFixed(CVRPTWGenerator):
    def __init__(
        self,
        *args,
        min_service_time: float = 10.0,
        max_service_time: float = 10.0,
        service_time_distribution: int | float | str | type | Callable = Uniform,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_service_time = float(min_service_time)
        self.max_service_time = float(max_service_time)
        self.service_time_distribution = service_time_distribution
        self.service_time_sampler = (
            None
            if self.min_service_time == self.max_service_time
            else get_sampler(
                "service_time",
                service_time_distribution,
                self.min_service_time,
                self.max_service_time,
                **kwargs,
            )
        )

    def _sample_durations(self, batch_size) -> torch.Tensor:
        shape = (*batch_size, self.num_loc + 1)
        durations = torch.zeros(shape, dtype=torch.float32)
        if self.service_time_sampler is None:
            durations[..., 1:] = self.min_service_time
        else:
            durations[..., 1:] = self.service_time_sampler.sample((*batch_size, self.num_loc))
        return durations

    def _generate(self, batch_size) -> TensorDict:
        td = super(CVRPTWGenerator, self)._generate(batch_size)
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        durations = self._sample_durations(batch_size)

        dist = get_distance(td["depot"], td["locs"].transpose(0, 1)).transpose(0, 1)
        dist = torch.cat((torch.zeros(*batch_size, 1), dist), dim=1)

        upper_bound = self.max_time - dist - durations

        ts_1 = torch.rand(*batch_size, self.num_loc + 1)
        ts_2 = torch.rand(*batch_size, self.num_loc + 1)

        min_ts = (dist + (upper_bound - dist) * ts_1).int()
        max_ts = (dist + (upper_bound - dist) * ts_2).int()

        min_times = torch.min(min_ts, max_ts)
        max_times = torch.max(min_ts, max_ts)

        min_times[..., :, 0] = 0.0
        max_times[..., :, 0] = self.max_time

        mask = min_times == max_times
        if torch.any(mask):
            min_tmp = min_times.clone()
            min_tmp[mask] = torch.max(dist[mask].int(), min_tmp[mask] - 1)
            min_times = min_tmp

            mask = min_times == max_times
            if torch.any(mask):
                max_tmp = max_times.clone()
                max_tmp[mask] = torch.min(
                    torch.floor(upper_bound[mask]).int(),
                    torch.max(
                        torch.ceil(min_tmp[mask] + durations[mask]).int(),
                        max_tmp[mask] + 1,
                    ),
                )
                max_times = max_tmp

        if self.scale:
            durations = durations / self.max_time
            min_times = min_times / self.max_time
            max_times = max_times / self.max_time
            td["depot"] = td["depot"] / self.max_time
            td["locs"] = td["locs"] / self.max_time

        time_windows = torch.stack((min_times, max_times), dim=-1)

        assert torch.all(
            min_times < max_times
        ), "Please make sure the relation between max_loc and max_time allows for feasible solutions."

        durations[:, 0] = 0.0
        td.update({"durations": durations, "time_windows": time_windows})
        return td
