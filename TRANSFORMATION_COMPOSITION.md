# Transformation Composition Detection

## Problem

When applying multiple transformations to images, certain combinations can be reduced to simpler transformations, creating redundant training data. For example:

**Channel Swaps:**
- Swap(R,B) followed by Swap(R,G) = Swap(B,G)
- Swap(R,B) followed by Swap(R,B) = Identity (no change)

**Channel Inversions:**
- Invert(R) followed by Invert(R) = Identity (no change)

These redundant sequences should be prevented to ensure each training sample represents a genuinely distinct transformation.

## Solution

### Architecture

The solution consists of three components:

1. **TransformationCompositionTable** (`src/transformer/TransformationComposition.py`)
   - Defines composition rules for transformations
   - Maintains a lookup table of blocked pairs
   - Provides methods to check if pairs create reducible compositions

2. **MultiRandomTransformationDegradingFunction** (`src/training/degrading/degrading_functions.py`)
   - Selects N random transformations per image
   - Uses the composition table to exclude problematic pairs
   - Ensures no duplicates, reversing pairs, or composable sequences

3. **Configuration** (`src/training/rl_training/training_params.py`)
   - Added `num_transformations` parameter
   - Controls how many transformations to apply per image

### Composition Rules Implemented

#### Channel Swap Group (S3 Permutation Group)

All swap combinations are blocked because they compose:

```
CA_SWAP_RB ∘ CA_SWAP_RG = CA_SWAP_GB
CA_SWAP_RG ∘ CA_SWAP_RB = CA_SWAP_GB
CA_SWAP_RB ∘ CA_SWAP_GB = CA_SWAP_RG
CA_SWAP_GB ∘ CA_SWAP_RB = CA_SWAP_RG
CA_SWAP_RG ∘ CA_SWAP_GB = CA_SWAP_RB
CA_SWAP_GB ∘ CA_SWAP_RG = CA_SWAP_RB
```

**Result:** Only ONE swap can be selected per image.

#### Channel Inversions (Boolean Algebra / XOR Logic)

Channel inversions behave like XOR operations on channel sets.

**1. Self-Inverse Operations (Blocked):**
```
CA_INV_B ∘ CA_INV_B = Identity
CA_INV_G ∘ CA_INV_G = Identity
CA_INV_R ∘ CA_INV_R = Identity
CA_INV_BR ∘ CA_INV_BR = Identity
```

**Result:** Same inversion cannot be applied twice.

**2. Channel Overlap / Cancellation (Blocked):**

When two inversions share channels, those channels cancel out:

```
CA_INV_BR ∘ CA_INV_B = CA_INV_R  (B cancels, only R remains)
CA_INV_BR ∘ CA_INV_R = CA_INV_B  (R cancels, only B remains)
CA_INV_BG ∘ CA_INV_B = CA_INV_G  (B cancels, only G remains)
CA_INV_ALL ∘ CA_INV_B = CA_INV_GR (B cancels, G&R remain)
```

**Result:** Inversions with overlapping channels are blocked.

**3. Independent Inversions (Allowed):**

Inversions on completely different channels are allowed:
```
CA_INV_B ∘ CA_INV_G = Valid (creates multi-channel effect)
CA_INV_B ∘ CA_INV_R = Valid (creates multi-channel effect)
CA_INV_G ∘ CA_INV_R = Valid (creates multi-channel effect)
```

**Result:** Allows combining single-channel inversions.

**Note:** There's a configurable option (`block_independent_inversions`) to also block independent inversions if your transformer pool includes multi-channel inversions like `CA_INV_BG`.

#### Mixed Transformations

Swaps and inversions on different channels are independent:
```
CA_SWAP_RB ∘ CA_INV_G = Valid (independent operations)
```

## Selection Algorithm

The `MultiRandomTransformationDegradingFunction._get_transformers()` method:

1. **Shuffle** available transformers randomly
2. **For each candidate transformer:**
   - Check if already selected or excluded → skip
   - Check if it composes with any previously selected transformer → skip
   - Add to selection and exclude:
     - The transformer itself
     - Its reverse transformer(s)
     - All transformers that would compose with it
3. **Continue** until N transformations selected

## Examples

### Valid 2-Transformation Sequences

✓ `[CA_SWAP_RB, CA_INV_G]` - Swap and inversion on different channels
✓ `[CA_INV_B, CA_INV_G]` - Inversions on different channels
✓ `[CA_INV_R, CA_SWAP_GB]` - Inversion and swap on different channels

### Blocked 2-Transformation Sequences

✗ `[CA_SWAP_RB, CA_SWAP_RG]` - Composes to `CA_SWAP_GB`
✗ `[CA_SWAP_RB, CA_SWAP_RB]` - Composes to identity
✗ `[CA_INV_B, CA_INV_B]` - Composes to identity

## Usage

### Configuration

```python
transform_preprocessing_params.set({
    "batch_size": 64,
    "transformer_names": POC_MULTI_ONE_STEP_TRANSFORMERS,
    "use_random_transformer": True,
    "num_transformations": 2,  # Apply 2 transformations per image
    "split": SplitRatios.create_default_split_ratios()
})
```

### Adding New Transformers

To add composition rules for new transformers:

1. Edit `src/transformer/TransformationComposition.py`
2. Add rules to `_initialize_*` methods
3. Document the mathematical properties

Example:
```python
def _initialize_my_new_transformer_group(self):
    # Define composition rules
    self._composition_rules[("T1", "T2")] = "T3"  # T1 ∘ T2 = T3
    self._composition_rules[("T1", "T1")] = None  # T1 ∘ T1 = Identity

    # Block the pairs
    self._blocked_pairs.add(("T1", "T2"))
    self._blocked_pairs.add(("T1", "T1"))
```

## Mathematical Background

### Symmetric Group S3

Channel swaps form the symmetric group S3 (permutations of 3 elements). This group has:
- 6 elements: identity + 3 transpositions + 2 3-cycles
- The 3 swap operations are the transpositions
- Composing two transpositions gives the third (if they share an element) or a 3-cycle

### Self-Inverse Operations

Both swaps and single-channel inversions are self-inverse (involutions):
- `f ∘ f = identity`
- Order of operation: 2

### Independent Operations

Operations that affect disjoint sets of pixels/channels commute and don't compose:
- Inverting different channels
- Swapping channels and inverting non-swapped channels

## Benefits

1. **No redundant training data** - Each sample is a unique transformation
2. **Efficient training** - No wasted computation on equivalent transformations
3. **Clearer learning signal** - Agent learns distinct transformation effects
4. **Extensible** - Easy to add new composition rules

## Testing

Run tests to verify composition detection:
```bash
python -m pytest test-suite/TestTransformationComposition.py
```

## Future Extensions

### Support for 3+ Transformations

The current implementation supports any value of `num_transformations`. For N > 2:
- The algorithm checks all pairwise compositions
- More restrictions apply as N increases
- May need to increase the transformer pool size

### Dynamic Composition Detection

For transformers without predefined rules:
- Could test transformations on sample images
- Compare results to detect equivalent sequences
- Automatically build composition table

### Commutative Reordering

Some transformations commute (e.g., independent channel inversions):
- Could normalize sequences to canonical order
- Detect `[A, B]` and `[B, A]` as equivalent
- Further reduce redundancy
