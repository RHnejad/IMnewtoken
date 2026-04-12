# Codebook Size Configuration in VQ-VAE

## Question: Do we set the number of codes in the codebook from the beginning?

**Short Answer:** **Yes!** The codebook size is fixed from the beginning of training and never changes.

---

## How Codebook Size is Set

### 1. Configuration Parameter

From `options/vq_option.py:35`:

```python
parser.add_argument("--nb_code", type=int, default=1024,
                    help="nb of embedding")
```

**Your current configuration:**
- `nb_code = 1024` (number of codebook entries)
- `code_dim = 512` (dimension of each code vector)

This means your codebook is a **fixed matrix of shape [1024, 512]** throughout training.

---

### 2. Initialization Process

The codebook goes through a **two-stage initialization**:

#### Stage 1: Model Creation (before training starts)
From `models/vq/quantizer.py:43-47`:

```python
def reset_codebook(self):
    self.init = False
    self.code_sum = None
    self.code_count = None
    # Create empty codebook with zeros
    self.register_buffer('codebook',
                        torch.zeros(self.nb_code, self.code_dim,
                                   requires_grad=False).cuda())
```

**Result:** Codebook initialized as **[1024, 512] of zeros**

#### Stage 2: First Training Batch
From `models/vq/quantizer.py:60-65`:

```python
def init_codebook(self, x):
    # x contains first batch of encoder outputs
    out = self._tile(x)  # Repeat/tile encoder outputs to fill 1024 slots
    self.codebook = out[:self.nb_code]  # Take first 1024
    self.code_sum = self.codebook.clone()
    self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
    self.init = True
```

**The `_tile` function** (lines 49-58):
```python
def _tile(self, x):
    nb_code_x, code_dim = x.shape
    # If we have fewer encoder outputs than codebook entries
    if nb_code_x < self.nb_code:
        n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
        std = 0.01 / np.sqrt(code_dim)
        # Repeat the encoder outputs and add small noise
        out = x.repeat(n_repeats, 1)
        out = out + torch.randn_like(out) * std
    else:
        out = x
    return out
```

**Example:**
- First batch has 256 samples × T timesteps = ~5000 encoder outputs
- Need 1024 codebook entries
- Solution: Take first 1024 encoder outputs as initial codebook values

---

## Why Fixed Size?

### 1. **Architecture Constraint**

The quantization layer maps continuous vectors to discrete indices:

```python
# Encoder output: [Batch, 512, Time]
# Quantization: find nearest of 1024 possible codes
# Output indices: [Batch, Time] with values in range [0, 1023]
```

The transformer/decoder expects these indices to be in range `[0, nb_code-1]`.

### 2. **Codebook as a Lookup Table**

Think of the codebook as a **fixed dictionary**:

```
Code Index | 512-dim Vector
-----------|----------------
    0      | [-9.07, -3.60, 2.35, ...]
    1      | [1.23, -0.45, 5.67, ...]
    2      | [...]
   ...     | ...
   1023    | [2.52, -8.19, -13.78, ...]
```

- **Number of entries (1024):** Fixed
- **Content of each entry:** Updated via EMA during training

### 3. **Trade-offs**

| nb_code | Pros | Cons |
|---------|------|------|
| **Small (256)** | • Less memory<br>• Faster lookup<br>• Forces compression | • Less expressive<br>• Coarser reconstruction |
| **Medium (1024)** | • Good balance<br>• Standard choice | • Balanced trade-off |
| **Large (8192)** | • Very expressive<br>• Fine details | • More memory<br>• May underutilize codes<br>• Slower |

**Your choice (1024):** Standard in literature, good balance

---

## Can Codebook Size Change?

### During Training: **NO**
- Size is fixed throughout entire training
- Only the **values** of the 1024 entries are updated via EMA
- Cannot add or remove codes

### When Fine-tuning: **NO**
- Must use the same `nb_code` as the pretrained model
- Codebook shape must match: `[1024, 512]`
- Changing size would break checkpoint loading

### For New Experiments: **YES**
- Train from scratch with different `--nb_code` value
- Common choices: 256, 512, 1024, 2048, 4096

---

## Codebook Utilization

Not all 1024 codes may be used! The model might prefer certain codes.

### Dead Codes Problem

Some codes might never be selected:

```python
# In quantization: find nearest code
distance = ||encoder_output - codebook[i]||²
code_idx = argmin(distance)
```

If `codebook[157]` is never the nearest, it becomes a "dead code."

### Dead Code Reset Mechanism

From `models/vq/quantizer.py:115-117`:

```python
# After EMA update
usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
code_update = self.code_sum / self.code_count

# If code hasn't been used (count < 1.0), reset to random encoder output
self.codebook = usage * code_update + (1-usage) * code_rand
```

**This keeps all 1024 codes "alive" by occasionally resetting unused ones.**

---

## Checking Codebook Utilization

You can check how many codes are actually being used:

```python
import torch

# Load checkpoint
ckpt = torch.load('checkpoints/interhuman/vq_default_finetune/model/latest.tar')
codebook = ckpt['vq_model']['quantizer.layers.0.codebook']

# During evaluation, count unique code indices
# This would be done in your evaluation loop:
# code_indices = model.encode(data)  # shape: [B, T]
# unique_codes = torch.unique(code_indices)
# utilization = len(unique_codes) / 1024 * 100
# print(f"Codebook utilization: {utilization:.1f}%")
```

**Good utilization:** 70-95% of codes used
**Poor utilization:** <50% (might indicate codebook is too large)

---

## Configuration in Your Setup

From your error output:
```
nb_code: 1024
code_dim: 512
```

And from `checkpoints/interhuman/vq_default_finetune/opt.txt`:
```
nb_code: 1024
code_dim: 512
mu: 0.99
```

**Your codebook:**
- **Fixed size:** 1024 entries
- **Each entry:** 512-dimensional vector
- **Total parameters:** 1024 × 512 = 524,288 values
- **Memory:** ~2 MB (float32)

---

## Summary

| Question | Answer |
|----------|--------|
| **Is codebook size set from the beginning?** | ✅ Yes, via `--nb_code` parameter |
| **Does it change during training?** | ❌ No, only values update via EMA |
| **Can I change it for fine-tuning?** | ❌ No, must match checkpoint |
| **Can I train with different size?** | ✅ Yes, but must train from scratch |
| **How are initial values set?** | From first batch of encoder outputs |
| **What if some codes are unused?** | Dead code reset mechanism reactivates them |

---

## Practical Guidelines

### Choosing `nb_code` for New Training

1. **For simple datasets:** 256-512 codes
2. **For complex motion (your case):** 1024 codes ✓
3. **For very high-fidelity:** 2048-4096 codes
4. **General rule:** `nb_code ≈ sqrt(dataset_size)`

### Your Current Setup (1024)

✅ Good choice for InterHuman dataset
✅ Standard in motion generation literature
✅ Provides good reconstruction quality
✅ Not too large to cause optimization issues

**Bottom line:** The 1024 codebook entries are **fixed "slots"** that are filled during the first training batch and then **continuously refined** via EMA throughout training. Think of it as a fixed-size dictionary where the definitions (code vectors) change but the number of words (1024) stays constant.
