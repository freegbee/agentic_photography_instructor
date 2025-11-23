from transformer import generate_transformer_pairs, REVERSIBLE_TRANSFORMERS
from utils.Registries import init_registries


def test_generate_pairs_excludes_reverse_labels():
    # Ensure registries are initialized so TRANSFORMER_REGISTRY is populated
    init_registries()

    # default excludes identity
    pairs = generate_transformer_pairs()
    # Basic checks: expect len == N*N minus excluded combos
    labels = list(REVERSIBLE_TRANSFORMERS)
    # compute expected naive pairs excluding reverse relationships using registry
    from utils.Registries import TRANSFORMER_REGISTRY

    def get_rev(lbl):
        try:
            obj = TRANSFORMER_REGISTRY.get(lbl)
            rev = getattr(obj, 'reverse_transformer_labels', None)
            if rev is None:
                return []
            if isinstance(rev, str):
                return [rev]
            return list(rev)
        except Exception:
            return []

    expected = []
    for a in labels:
        a_rev = get_rev(a)
        for b in labels:
            if b in a_rev:
                continue
            expected.append((a, b))

    assert set(pairs) == set(expected)
    # Ensure that no pair has second element that is in first.reverse_transformer_labels
    for a, b in pairs:
        revs = get_rev(a)
        assert b not in revs


def test_sampling_and_identity_and_seed_behavior():
    init_registries()
    labels = list(REVERSIBLE_TRANSFORMERS)
    all_pairs = generate_transformer_pairs(transformer_labels=labels, exclude_identity=False)
    # request deterministic sample of size 5
    s1 = generate_transformer_pairs(transformer_labels=labels, exclude_identity=False, sample_size=5, seed=12345)
    s2 = generate_transformer_pairs(transformer_labels=labels, exclude_identity=False, sample_size=5, seed=12345)
    assert s1 == s2
    assert len(s1) == 5

    # requesting too-large sample should raise
    try:
        generate_transformer_pairs(transformer_labels=labels, exclude_identity=False, sample_size=len(all_pairs) + 1)
        raised = False
    except ValueError:
        raised = True
    assert raised
