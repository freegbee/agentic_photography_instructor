from collections import Counter
from stable_baselines3.common.vec_env import VecEnvWrapper


class TransformerUsageVecEnvWrapper(VecEnvWrapper):
    """
    Ein Wrapper für VecEnvs, der die Nutzung von Transformern (basierend auf 'step_history' in info) mitzählt.
    Dient primär dazu, die Nutzung während der Evaluation zu erfassen.
    """
    def __init__(self, venv):
        super().__init__(venv)
        self.buffer_usage = Counter()
        self.has_data = False

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        for i, info in enumerate(infos):
            # Wir prüfen, ob 'step_history' vorhanden ist (ImageTransformEnv)
            if 'step_history' in info and info['step_history']:
                # Der letzte Eintrag ist der aktuell angewendete Transformer
                last_step = info['step_history'][-1]
                label = last_step.get('label')
                if label:
                    self.buffer_usage[label] += 1
                    self.has_data = True
        return obs, rews, dones, infos

    def pop_usage(self):
        if not self.has_data:
            return None
        data = self.buffer_usage.copy()
        self.buffer_usage.clear()
        self.has_data = False
        return data