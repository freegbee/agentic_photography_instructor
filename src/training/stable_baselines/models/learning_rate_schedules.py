def constant_schedule(value: float):
    """Konstante Lernrate."""
    def schedule(_progress_remaining: float) -> float:
        return value
    return schedule

def linear_schedule(initial_value: float):
    """Lineare Abnahme: lr = initial_value * progress_remaining."""
    def schedule(progress_remaining: float) -> float:
        return initial_value * progress_remaining
    return schedule

def exponential_decay_schedule(initial_value: float, decay_rate: float = 0.99):
    """Exponentialer Zerfall über den Fortschritt (decay_rate in (0,1])."""
    def schedule(progress_remaining: float) -> float:
        # progress_remaining: 1.0 -> initial_value, 0.0 -> initial_value * decay_rate**1
        return initial_value * (decay_rate ** (1.0 - progress_remaining))
    return schedule
