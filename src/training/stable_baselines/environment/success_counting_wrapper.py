import logging
import time
from typing import Tuple, Any, Dict

import gymnasium as gym

logger = logging.getLogger(__name__)


class SuccessCountingWrapper(gym.Wrapper):
    """
    Wrapper, der pro Episode Erfolge zählt (erwartet, dass inneres env in info['success'] ein bool liefert).
    Bei Ende der Episode werden einige Kern-Metriken ergänzt in info dict mit dem Key `stats_key`
    """
    def __init__(self, env: gym.Env, stats_key):
        super().__init__(env)
        self._episode_steps = 0
        self._stats_key = stats_key
        self._reset_episode_counters()
        self._start_time = None

    def _reset_episode_counters(self):
        self._success_count = 0
        self._cum_reward = 0.0
        self._length = 0
        self._start_time = None

    def reset(self, *, seed=None, options=None) -> Tuple[Any, Dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._reset_episode_counters()
        self._start_time = time.time()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._cum_reward += float(reward)
        self._length += 1

        # falls das wrapped env ein 'success' boolean liefert, zählen
        if isinstance(info, dict) and info.get("success") is True:
            self._success_count += 1

        done = terminated or truncated
        if done:
            score = float(info["score"]) if "score" in info else None
            initial_score = float(info["initial_score"]) if "initial_score" in info else None
            success = info.get("success", False)
            matches_expected = info.get("matches_expected", False)
            ep_info = {
                "r": float(self._cum_reward),
                "l": int(self._length),
                "t": float(time.time() - (self._start_time or time.time())),
                "success": success,
                "matches_expected": matches_expected,
                "score": score,
                "initial_score": initial_score,
                "score_change": float(score - initial_score) if score is not None and initial_score is not None else None
            }
            # Informationen bei abgeschlossener Episode ins info dict packen
            info[self._stats_key] = dict(ep_info)
            info["is_success"] = success
            logger.debug("Episode ended: info=%s" % str(ep_info))
        return obs, reward, terminated, truncated, info
