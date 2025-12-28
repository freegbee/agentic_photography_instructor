from typing import List, Tuple, Optional

from utils.Registries import TRANSFORMER_REGISTRY


def _normalize_reverse_labels(obj) -> List[str]:
    """Return a list of reverse labels from the transformer class/instance.

    Handles cases where the attribute is a list, tuple, string or missing.
    """
    rev = getattr(obj, "reverse_transformer_label", None)
    if rev is None:
        return []
    return [rev]


def generate_transformer_pairs(
    transformer_labels: Optional[List[str]] = None,
    exclude_identity: bool = True,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Generate transformer label pairs (cross-product) with filtering and optional deterministic sampling.

    Args:
        transformer_labels: Optional list of labels to use. If None, uses transformer.REVERSIBLE_TRANSFORMERS.
        exclude_identity: If True (default), exclude pairs where a == b.
        sample_size: Optional number of pairs to return (sampled without replacement). If None, return all pairs.
        seed: Optional integer seed for deterministic sampling. If provided, sampling is deterministic.

    Behavior:
        - Pairs (a, b) are excluded if 'b' appears in 'a.reverse_transformer_labels'.
        - If `exclude_identity` is True, pairs where a == b are excluded.
        - If `sample_size` is set, a deterministic selection is performed by shuffling the full list
          with a local Random(seed) and returning the first `sample_size` elements.

    Returns:
        A list of (label_a, label_b) tuples.
    """
    if transformer_labels is None:
        # lazy import to avoid circular imports with transformer.__init__
        import transformer as _transformer_pkg

        labels = list(_transformer_pkg.REVERSIBLE_TRANSFORMERS)
    else:
        labels = list(transformer_labels)

    pairs: List[Tuple[str, str]] = []

    for a in labels:
        try:
            a_obj = TRANSFORMER_REGISTRY.get(a)
        except Exception:
            a_obj = None

        a_rev = _normalize_reverse_labels(a_obj) if a_obj is not None else []

        for b in labels:
            # optionally exclude identity pairs
            if exclude_identity and a == b:
                continue
            # skip combinations where b is listed as a reverse transformer of a
            if b in a_rev:
                continue
            pairs.append((a, b))

    # If sampling requested, perform deterministic sampling using seed
    if sample_size is not None:
        if sample_size < 0:
            raise ValueError("sample_size must be non-negative")
        if sample_size > len(pairs):
            raise ValueError("sample_size cannot be larger than the number of available pairs")
        # deterministic shuffle then take first k
        import random

        rnd = random.Random(seed) if seed is not None else random.Random()
        shuffled = list(pairs)
        rnd.shuffle(shuffled)
        return shuffled[:sample_size]

    return pairs
