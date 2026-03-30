from routefinder.envs.mtvrp import MTVRPGenerator
import torch


class SolomonMTVRPGenerator(MTVRPGenerator):
    def __init__(self, solomon_instances, *args, **kwargs):
        """
        solomon_instances: list of dicts or tensors
            Each item must contain:
                - 'locs': Tensor [N+1, 2] (including depot)
        """
        super().__init__(*args, **kwargs)

        assert len(solomon_instances) > 0, "No Solomon instances provided"

        self.solomon_instances = solomon_instances

    # def generate_locations(self, batch_size, num_loc) -> torch.Tensor:
    #     B = batch_size[0] if isinstance(batch_size, tuple) else batch_size

    #     locs_batch = []

    #     for i in range(len(B)):
    #         instance = self.solomon_instances[i % len(self.solomon_instances)]
    #         locs = instance  # expected shape [N+1, 2]

    #         # Critical check
    #         assert locs.shape[0] == num_loc + 1, (
    #             f"Mismatch: expected {num_loc+1}, got {locs.shape[0]}"
    #         )

    #         locs_batch.append(locs)

    #     locs_batch = torch.stack(locs_batch, dim=0)  # [B, N+1, 2]

    #     return locs_batch
    def generate_locations(self, batch_size, num_loc):
            B = batch_size[0] if isinstance(batch_size, tuple) or isinstance(batch_size, list) else batch_size

            locs_batch = []

            for i in range(B):
                locs = self.solomon_instances[i % len(self.solomon_instances)] # [N+1, 2]

                assert locs.ndim == 2, locs.shape
                assert locs.shape[1] == 2, locs.shape

                locs = locs[: num_loc + 1]

                locs_batch.append(locs)

            locs_batch = torch.stack(locs_batch, dim=0)  # [B, N+1, 2]

            return locs_batch