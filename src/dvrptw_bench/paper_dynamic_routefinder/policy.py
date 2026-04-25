"""RouteFinder policy with stepwise dynamic re-encoding."""

from __future__ import annotations

from typing import Any

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.common.constructive.base import get_decoding_strategy, get_log_likelihood
from rl4co.utils.ops import calculate_entropy
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict

from routefinder.models import RouteFinderPolicy

log = get_pylogger(__name__)


class DynamicAttentionRouteFinderPolicy(RouteFinderPolicy):
    """RouteFinder policy that re-encodes after each environment transition.

    This matches the paper excerpt more closely than the stock RouteFinder policy:
    at every decoding step, the transformed input and its mask are recomputed from the
    current disclosure state, then the encoder is run again before decoding the next action.
    """

    # def _get_encoder_mask(self, td: TensorDict):
    #     if "encoder_mask" in td.keys():
    #         return td["encoder_mask"]
    #     if "revealed" in td.keys():
    #         mask = td["revealed"].clone()
    #         mask[..., 0] = True
    #         return mask
    #     return None

    # def forward(
    #     self,
    #     td: TensorDict,
    #     env: str | RL4COEnvBase | None = None,
    #     phase: str = "train",
    #     calc_reward: bool = True,
    #     return_actions: bool = True,
    #     return_entropy: bool = False,
    #     return_hidden: bool = False,
    #     return_init_embeds: bool = False,
    #     return_sum_log_likelihood: bool = True,
    #     actions=None,
    #     max_steps=1_000_000,
    #     **decoding_kwargs,
    # ) -> dict[str, Any]:
    #     if isinstance(env, str) or env is None:
    #         raise ValueError(
    #             "DynamicAttentionRouteFinderPolicy requires an instantiated environment."
    #         )

    #     decode_type = decoding_kwargs.pop("decode_type", None)
    #     if actions is not None:
    #         decode_type = "evaluate"
    #     elif decode_type is None:
    #         decode_type = getattr(self, f"{phase}_decode_type")

    #     decode_strategy = get_decoding_strategy(
    #         decode_type,
    #         temperature=decoding_kwargs.pop("temperature", self.temperature),
    #         tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
    #         mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
    #         store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
    #         **decoding_kwargs,
    #     )

    #     td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

    #     step = 0
    #     last_hidden = None
    #     last_init_embeds = None
    #     while not td["done"].all():
    #         encoder_mask = self._get_encoder_mask(td)
    #         hidden, init_embeds = self.encoder(td, mask=encoder_mask)
    #         td_step, env, cache = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)
    #         logits, mask = self.decoder(td_step, cache, num_starts)
    #         td = decode_strategy.step(
    #             logits,
    #             mask,
    #             td_step,
    #             action=actions[..., step] if actions is not None else None,
    #         )
    #         td = env.step(td)["next"]
    #         last_hidden = hidden
    #         last_init_embeds = init_embeds
    #         step += 1
    #         if step > max_steps:
    #             log.error(f"Exceeded maximum number of steps ({max_steps}) duing decoding")
    #             break

    #     logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

    #     if calc_reward:
    #         td.set("reward", env.get_reward(td, actions))

    #     outdict = {
    #         "reward": td["reward"],
    #         "log_likelihood": get_log_likelihood(
    #             logprobs, actions, td.get("mask", None), return_sum_log_likelihood
    #         ),
    #     }
    #     if return_actions:
    #         outdict["actions"] = actions
    #     if return_entropy:
    #         outdict["entropy"] = calculate_entropy(logprobs)
    #     if return_hidden:
    #         outdict["hidden"] = last_hidden
    #     if return_init_embeds:
    #         outdict["init_embeds"] = last_init_embeds
    #     return outdict
