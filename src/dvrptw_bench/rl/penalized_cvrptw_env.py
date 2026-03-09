import torch
from rl4co.envs import CVRPTWEnv


class PenalizedCVRPTWEnv(CVRPTWEnv):
    def __init__(self, vehicle_penalty: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.vehicle_penalty = float(vehicle_penalty)

    @staticmethod
    def count_routes_from_actions(actions: torch.Tensor) -> torch.Tensor:
        """
        actions: [B, T]
        Assumes depot token is 0 and counts non-empty routes.
        """
        batch_routes = []
        for seq in actions.tolist():
            in_route = False
            routes = 0
            for a in seq:
                if a == 0:
                    in_route = False
                else:
                    if not in_route:
                        routes += 1
                        in_route = True
            batch_routes.append(routes)
        return torch.tensor(batch_routes, device=actions.device, dtype=torch.float32)

    def _get_reward(self, td, actions):
        # Start from the parent env cost/reward logic
        base_reward = super()._get_reward(td, actions)

        # RL4CO routing rewards are typically negative cost, so add penalty as extra cost
        num_routes = self.count_routes_from_actions(actions)
        penalized_reward = base_reward + self.vehicle_penalty * num_routes

        return penalized_reward