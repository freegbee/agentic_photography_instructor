"""
Transformation composition detection and prevention.

This module provides functionality to detect when a sequence of transformations
can be replaced with a shorter equivalent sequence.
"""
from typing import List, Optional, Set, Dict, Tuple


class TransformationCompositionTable:
    """
    Defines composition rules for transformations.

    When transformer A followed by transformer B equals transformer C (or identity),
    we should prevent such sequences to ensure we're not creating redundant transformations.
    """

    def __init__(self, block_independent_inversions: bool = False):
        """
        Initialize the composition table.

        Args:
            block_independent_inversions: If True, block inversions on different channels
                that compose to multi-channel inversions (e.g., INV_B + INV_G = INV_BG).
                If False, only block inversions with channel overlap.
                Default: False (allow independent inversions)
        """
        # Maps (transformer_a, transformer_b) -> equivalent_single_transformer
        # If the result is None, it means the composition results in identity (no change)
        self._composition_rules: Dict[Tuple[str, str], Optional[str]] = {}

        # Maps (transformer_a, transformer_b) -> True if this pair should be blocked
        self._blocked_pairs: Set[Tuple[str, str]] = set()

        self._block_independent_inversions = block_independent_inversions

        self._initialize_swap_compositions()
        self._initialize_inversion_compositions()

    def _initialize_swap_compositions(self):
        """
        Channel swaps form a permutation group (S3).

        Composition rules:
        - Swap(X,Y) ∘ Swap(X,Y) = Identity (self-inverse)
        - Swap(R,B) ∘ Swap(R,G) = Swap(B,G)
        - Swap(R,G) ∘ Swap(R,B) = Swap(B,G)
        - Swap(R,B) ∘ Swap(B,G) = Swap(R,G)
        - Swap(B,G) ∘ Swap(R,B) = Swap(R,G)
        - Swap(R,G) ∘ Swap(B,G) = Swap(R,B)
        - Swap(B,G) ∘ Swap(R,G) = Swap(R,B)
        """
        # Self-inverse (applying same swap twice = identity)
        self._composition_rules[("CA_SWAP_RB", "CA_SWAP_RB")] = None  # Identity
        self._composition_rules[("CA_SWAP_RG", "CA_SWAP_RG")] = None  # Identity
        self._composition_rules[("CA_SWAP_GB", "CA_SWAP_GB")] = None  # Identity

        # Two different swaps compose to a third swap
        # Swap(R,B) then Swap(R,G) = Swap(B,G)
        self._composition_rules[("CA_SWAP_RB", "CA_SWAP_RG")] = "CA_SWAP_GB"
        self._composition_rules[("CA_SWAP_RG", "CA_SWAP_RB")] = "CA_SWAP_GB"

        # Swap(R,B) then Swap(B,G) = Swap(R,G)
        self._composition_rules[("CA_SWAP_RB", "CA_SWAP_GB")] = "CA_SWAP_RG"
        self._composition_rules[("CA_SWAP_GB", "CA_SWAP_RB")] = "CA_SWAP_RG"

        # Swap(R,G) then Swap(B,G) = Swap(R,B)
        self._composition_rules[("CA_SWAP_RG", "CA_SWAP_GB")] = "CA_SWAP_RB"
        self._composition_rules[("CA_SWAP_GB", "CA_SWAP_RG")] = "CA_SWAP_RB"

        # Block all swap compositions since they can be reduced
        swap_labels = ["CA_SWAP_RB", "CA_SWAP_RG", "CA_SWAP_GB"]
        for a in swap_labels:
            for b in swap_labels:
                self._blocked_pairs.add((a, b))

    def _initialize_inversion_compositions(self):
        """
        Channel inversions form a Boolean algebra (XOR-like behavior).

        Composition rules:
        - Invert(X) ∘ Invert(X) = Identity (self-inverse)
        - Invert(A,B) ∘ Invert(A) = Invert(B) (channel cancellation)
        - Inversions compose by XOR of their channel sets

        Examples:
        - CA_INV_BR ∘ CA_INV_B = CA_INV_R  (invert B&R, then invert B again → only R inverted)
        - CA_INV_BR ∘ CA_INV_R = CA_INV_B  (invert B&R, then invert R again → only B inverted)
        - CA_INV_BR ∘ CA_INV_G = CA_INV_ALL (invert B&R, then invert G → all three inverted)
        """
        # Map each inversion to its channel set
        inversion_channels = {
            "CA_INV_B": {"B"},
            "CA_INV_G": {"G"},
            "CA_INV_R": {"R"},
            "CA_INV_BG": {"B", "G"},
            "CA_INV_BR": {"B", "R"},
            "CA_INV_GR": {"G", "R"},
            "CA_INV_ALL": {"B", "G", "R"},
        }

        # Map channel sets back to labels
        channels_to_label = {
            frozenset({"B"}): "CA_INV_B",
            frozenset({"G"}): "CA_INV_G",
            frozenset({"R"}): "CA_INV_R",
            frozenset({"B", "G"}): "CA_INV_BG",
            frozenset({"B", "R"}): "CA_INV_BR",
            frozenset({"G", "R"}): "CA_INV_GR",
            frozenset({"B", "G", "R"}): "CA_INV_ALL",
        }

        inversion_labels = list(inversion_channels.keys())

        # Compute all pairwise compositions
        for label1 in inversion_labels:
            for label2 in inversion_labels:
                channels1 = inversion_channels[label1]
                channels2 = inversion_channels[label2]

                # XOR operation: symmetric difference
                result_channels = channels1.symmetric_difference(channels2)

                # Check if result is a valid single-transformer operation
                result_label = channels_to_label.get(frozenset(result_channels))

                if len(result_channels) == 0:
                    # Inverting the same channels twice = identity
                    self._composition_rules[(label1, label2)] = None
                    self._blocked_pairs.add((label1, label2))
                elif result_label is not None:
                    # Composition results in a known single transformer
                    self._composition_rules[(label1, label2)] = result_label

                    # Determine if we should block this pair
                    has_overlap = len(channels1 & channels2) > 0

                    if has_overlap or self._block_independent_inversions:
                        # Block if:
                        # 1. Channels overlap (cancellation), OR
                        # 2. Policy says to block independent inversions too
                        self._blocked_pairs.add((label1, label2))

    def is_pair_blocked(self, first_label: str, second_label: str) -> bool:
        """
        Check if a pair of transformations should be blocked because they compose
        to a simpler transformation or to identity.

        Args:
            first_label: The label of the first transformation
            second_label: The label of the second transformation to apply after the first

        Returns:
            True if this pair should be prevented
        """
        return (first_label, second_label) in self._blocked_pairs

    def get_composition_result(self, first_label: str, second_label: str) -> Optional[str]:
        """
        Get the result of composing two transformations.

        Args:
            first_label: The label of the first transformation
            second_label: The label of the second transformation

        Returns:
            - None if the composition results in identity (no change)
            - The label of the equivalent single transformer
            - The second_label if no composition rule exists (no simplification possible)
        """
        return self._composition_rules.get((first_label, second_label), second_label)

    def is_sequence_reducible(self, labels: List[str]) -> bool:
        """
        Check if a sequence of transformations can be reduced to a shorter sequence.

        Args:
            labels: List of transformation labels in order of application

        Returns:
            True if the sequence contains composable pairs that could be simplified
        """
        for i in range(len(labels) - 1):
            if self.is_pair_blocked(labels[i], labels[i + 1]):
                return True
        return False

    def get_conflicting_transformers(self, selected_label: str) -> Set[str]:
        """
        Get all transformer labels that would create a reducible composition
        if applied after the selected transformer.

        Args:
            selected_label: The transformer that was already selected

        Returns:
            Set of transformer labels that should be excluded from further selection
        """
        conflicting = set()
        for (first, second) in self._blocked_pairs:
            if first == selected_label:
                conflicting.add(second)
        return conflicting


# Global singleton instance
# Default: Only block inversions with channel overlap, not independent inversions
# This allows combining single-channel inversions to create multi-channel effects
_composition_table = TransformationCompositionTable(block_independent_inversions=False)


def get_composition_table() -> TransformationCompositionTable:
    """
    Get the global transformation composition table.

    By default, this blocks:
    - All swap combinations (they form a group)
    - Inversions with channel overlap (e.g., INV_BR + INV_B = INV_R)
    - Self-inverse operations (e.g., INV_B + INV_B = identity)

    But allows:
    - Independent inversions (e.g., INV_B + INV_G, which creates a multi-channel effect)
    """
    return _composition_table
