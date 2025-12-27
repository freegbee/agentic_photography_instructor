import os
from pathlib import Path

import numpy as np

from utils.Registries import init_registries, TRANSFORMER_REGISTRY
from utils.TestingUtils import TestingUtils


def _normalize_to_list(rev):
    if rev is None:
        return []
    if isinstance(rev, str):
        return [rev]
    return list(rev)


def test_transform_roundtrip():
    """Für alle Transformer, die ein reverse_transformer_label definieren,
    prüfe, dass die Anwendung der Transformation und der Rücktransformation
    das Bild (nahezu) zum Original zurückführt.

    Dieser Test sammelt alle fehlschlagenden Paare und gibt am Ende eine
    zusammenfassende Fehlermeldung mit max-Differenzen aus, anstatt beim
    ersten Fehler abzubrechen.
    """
    init_registries()

    image_path = Path(__file__).parent / "resources" / "testimage.png"
    assert image_path.exists(), f"Testimage nicht gefunden: {image_path}"

    orig = TestingUtils.load_image_from_path(image_path)
    assert orig is not None, "Laden des Testbilds fehlgeschlagen"

    orig_arr = np.asarray(orig)

    failures = []
    # tolerance: maximale absolute Differenz pro Farbkanal
    MAX_ALLOWED_DIFF = int(os.environ.get("TRANSFORMER_ROUNDTRIP_MAX_DIFF", "1"))

    for label in list(TRANSFORMER_REGISTRY.keys()):
        transformer = TRANSFORMER_REGISTRY.get(label)
        t = transformer

        # prefer method when available
        rev_attr = None
        try:
            rev_attr = t.get_reverse_transformer_label()
        except Exception:
            # fallback to attribute lookup
            rev_attr = getattr(t, 'reverse_transformer_label', None) if hasattr(t, 'reverse_transformer_label') else None
            if rev_attr is None:
                rev_attr = getattr(t, 'reverse_transformer_labels', None) if hasattr(t, 'reverse_transformer_labels') else None

        rev_labels = _normalize_to_list(rev_attr)
        if not rev_labels:
            continue

        for rev_label in rev_labels:
            if rev_label not in TRANSFORMER_REGISTRY.keys():
                failures.append((label, rev_label, 'missing_reverse_in_registry'))
                continue

            rev_t = TRANSFORMER_REGISTRY.get(rev_label)

            try:
                transformed = t.transform(orig_arr.copy())
            except Exception as e:
                failures.append((label, rev_label, f'transform_error: {e}'))
                continue
            try:
                reverted = rev_t.transform(transformed)
            except Exception as e:
                failures.append((label, rev_label, f'reverse_transform_error: {e}'))
                continue

            if getattr(reverted, 'shape', None) != getattr(orig_arr, 'shape', None):
                failures.append((label, rev_label, f'shape_mismatch: {getattr(reverted, "shape", None)} != {getattr(orig_arr, "shape", None)}'))
                continue

            diff = np.abs(reverted.astype(np.int32) - orig_arr.astype(np.int32))
            max_diff = int(diff.max()) if diff.size else 0
            if max_diff > MAX_ALLOWED_DIFF:
                failures.append((label, rev_label, f'max_diff={max_diff}'))

    if failures:
        # Baue eine lesbare Fehlermeldung
        msgs = [f"{a} -> {b}: {c}" for a, b, c in failures]
        joined = "\n".join(msgs)
        raise AssertionError(f"Roundtrip-Validierung für Transformer fehlgeschlagen (toleranz={MAX_ALLOWED_DIFF}):\n" + joined)
