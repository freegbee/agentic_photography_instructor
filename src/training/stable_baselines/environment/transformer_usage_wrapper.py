import logging
from collections import Counter
from stable_baselines3.common.vec_env import VecEnvWrapper

from transformer.TransformerTypes import TransformerTypeEnum

logger = logging.getLogger(__name__)


class TransformerUsageVecEnvWrapper(VecEnvWrapper):
    """
    Ein Wrapper für VecEnvs, der die Nutzung von Transformern (basierend auf 'step_history' in info) mitzählt.
    Dient primär dazu, die Nutzung während der Evaluation zu erfassen.
    """
    def __init__(self, venv):
        super().__init__(venv)
        self.buffer_usage = Counter()
        self.buffer_score_deltas = Counter()
        self.crop_stats = {"attempts": 0, "changed": 0, "score_delta_sum": 0.0}
        self.mdp_stats = {"count": 0, "total": 0}
        self.has_data = False
        self._warned_missing_history = False

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        for i, info in enumerate(infos):
            # Wir prüfen, ob 'step_history' vorhanden ist (ImageTransformEnv)
            if 'step_history' in info:
                if info['step_history']:
                    # Der letzte Eintrag ist der aktuell angewendete Transformer
                    last_step = info['step_history'][-1]
                    label = last_step.get('label')
                    if label:
                        self.buffer_usage[label] += 1
                        self.buffer_score_deltas[label] += last_step.get('score_delta', 0.0)
                        self.has_data = True
                        
                        # Crop Statistics
                        if last_step.get('transformer_type') == TransformerTypeEnum.CROP:
                            self.crop_stats['attempts'] += 1
                            if last_step.get('dims_changed'):
                                self.crop_stats['changed'] += 1
                                self.crop_stats['score_delta_sum'] += last_step.get('score_delta', 0.0)
                    else:
                        # Debug: Label fehlt im step_history Eintrag
                        logger.debug("TransformerUsageVecEnvWrapper: Entry in step_history has no 'label': %s", last_step)
            # Fallback: Wenn step_history fehlt (z.B. durch Wrapper entfernt), prüfen wir auf 'transformer_label'
            elif 'transformer_label' in info:
                label = info['transformer_label']
                if label:
                    self.buffer_usage[label] += 1
                    self.has_data = True
            elif not self._warned_missing_history:
                logger.warning(f"TransformerUsageVecEnvWrapper: 'step_history' and 'transformer_label' keys missing in info dict. Keys found: {list(info.keys())}. Usage tracking will fail.")
                self._warned_missing_history = True
            
            # MDP Tracking at end of episode
            if dones[i]:
                self.mdp_stats["total"] += 1
                if info.get("mdp", False):
                    self.mdp_stats["count"] += 1
        return obs, rews, dones, infos

    def pop_usage(self):
        if not self.has_data:
            return None
        data = self.buffer_usage.copy()
        deltas = self.buffer_score_deltas.copy()
        self.buffer_usage.clear()
        self.buffer_score_deltas.clear()
        self.has_data = False
        return data, deltas

    def pop_crop_stats(self):
        # Return copy and reset
        stats = self.crop_stats.copy()
        self.crop_stats = {"attempts": 0, "changed": 0, "score_delta_sum": 0.0}
        return stats

    def pop_mdp_stats(self):
        # Return copy and reset
        stats = self.mdp_stats.copy()
        self.mdp_stats = {"count": 0, "total": 0}
        return stats