# VQ-VAE Codebook Fine-tuning Explained

## The Complete Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    VQ-VAE TRAINING LOOP                      │
└─────────────────────────────────────────────────────────────┘

Input Motion
     ↓
┌────────────────┐
│    ENCODER     │ ← Learnable via gradients (AdamW)
│  (66 params)   │   ✓ In optimizer state
└────────────────┘
     ↓
  z (latent)
     ↓
┌────────────────────────────────────────────────────────────┐
│              VECTOR QUANTIZATION                           │
│                                                            │
│  1. Find nearest code:   idx = argmin ||z - codebook[i]|| │
│  2. Quantize:            z_q = codebook[idx]              │
│  3. Update codebook via EMA (NOT gradients!):             │
│                                                            │
│     codebook[idx] = 0.99 * codebook[idx] +                │
│                     0.01 * mean(z_assigned_to_idx)        │
│                                                            │
│  Codebook (1 param): [1024, 512]                          │
│  ✗ NOT in optimizer state                                 │
│  ✓ Updated via direct EMA assignment                      │
└────────────────────────────────────────────────────────────┘
     ↓
  z_q (quantized)
     ↓
┌────────────────┐
│    DECODER     │ ← Learnable via gradients (AdamW)
│  (35 params)   │   ✓ In optimizer state
└────────────────┘
     ↓
Reconstructed Motion
     ↓
┌────────────────────────────────────────────────────────────┐
│                   LOSS & BACKPROP                          │
│                                                            │
│  loss = reconstruction + commitment                        │
│  loss.backward()  ← Gradients for encoder/decoder ONLY    │
│  optimizer.step() ← Updates 66 params, skips codebook     │
└────────────────────────────────────────────────────────────┘
```

---

## Two Update Mechanisms

### A. Gradient-Based Learning (via AdamW optimizer)
- **What:** Encoder, Decoder, and other network parameters
- **How:** Standard backpropagation with Adam optimization
- **Parameters tracked:** 66 out of 67 total parameters

### B. Exponential Moving Average (EMA) Learning
- **What:** The codebook vectors
- **How:** Direct statistical updates without gradients
- **Parameters tracked:** 1 parameter (`quantizer.layers.0.codebook`) - **NOT in optimizer**

---

## How EMA Codebook Update Works

From `models/vq/quantizer.py:100-123`:

```python
@torch.no_grad()  # <-- NO GRADIENTS! This is why it's not in optimizer
def update_codebook(self, x, code_idx):
    # Step 1: Count which codes are being used
    code_onehot = torch.zeros(self.nb_code, x.shape[0])
    code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

    # Step 2: Sum up all encoder outputs assigned to each code
    code_sum = torch.matmul(code_onehot, x)  # nb_code x code_dim
    code_count = code_onehot.sum(dim=-1)     # nb_code

    # Step 3: EMA Update (mu=0.99 from your config)
    self.code_sum = 0.99 * self.code_sum + 0.01 * code_sum
    self.code_count = 0.99 * self.code_count + 0.01 * code_count

    # Step 4: Update codebook = running average / running count
    code_update = self.code_sum / self.code_count

    # Step 5: Replace active codes, reset dead ones
    usage = (self.code_count >= 1.0).float()
    self.codebook = usage * code_update + (1-usage) * code_rand
```

---

## Detailed Step-by-Step During Fine-tuning

**Training Step N:**

1. **Encoder forward:**
   ```
   motion → encoder → z (shape: [B, 512, T])
   ```

2. **Quantization:**
   ```python
   # Find nearest codebook vector for each z[i]
   distance = ||z - codebook[j]||²
   code_idx = argmin(distance)  # Which codebook entry is closest?

   # Get the quantized vector
   z_quantized = codebook[code_idx]
   ```

3. **Codebook Update (EMA):**
   ```python
   # For each code that was used:
   # Example: code 157 was used for encoder outputs [z1, z5, z9]

   current_mean = (z1 + z5 + z9) / 3

   # Update with momentum (mu=0.99)
   codebook[157] = 0.99 * codebook[157] + 0.01 * current_mean
   ```

   **This is a DIRECT assignment - no gradients needed!**

4. **Decoder forward:**
   ```
   z_quantized → decoder → reconstructed_motion
   ```

5. **Gradient-based updates:**
   ```python
   loss = reconstruction_loss + commitment_loss
   loss.backward()  # Gradients for encoder/decoder only!
   optimizer.step()  # Updates encoder/decoder, NOT codebook
   ```

---

## Why This Two-Tier Approach?

The original VQ-VAE paper found this works better because:

**Problem with gradient-based codebook updates:**
- Only codes that are currently used get gradient updates
- Unused codes never improve → "dead codes"
- Training instability

**Solution with EMA:**
- ✅ More stable training
- ✅ Codebook continuously adapts to encoder output distribution
- ✅ Dead code reset mechanism (line 117 in quantizer.py)
- ✅ Smoother updates via momentum

---

## What Happens During Fine-tuning

When you fine-tune from a checkpoint:

**Encoder/Decoder (gradient-based):**
```
- Start from checkpoint weights ✓
- Start from checkpoint optimizer state (Adam momentum buffers) ✓
- Continue with same learning rate schedule ✓
```

**Codebook (EMA-based):**
```
- Start from checkpoint codebook vectors ✓
- NO optimizer state (because there isn't any!) ✓
- EMA statistics (code_sum, code_count) loaded from checkpoint ✓
- Continues updating via EMA as before ✓
```

---

## Key Comparison

| Aspect | Encoder/Decoder | Codebook |
|--------|----------------|----------|
| **Learning method** | Gradient descent (AdamW) | Exponential Moving Average |
| **Update trigger** | `optimizer.step()` | Every forward pass (line 143 in quantizer.py) |
| **In optimizer?** | ✅ Yes (66 params) | ❌ No (1 param) |
| **`requires_grad`** | `True` | `False` (line 47 in quantizer.py) |
| **Momentum** | Adam momentum (`exp_avg`) | EMA momentum (`mu=0.99`) |
| **When fine-tuning** | Load weights + optimizer state | Load weights only (no optimizer state exists) |

---

## Why Your Error Occurred

1. **Checkpoint saved:** 67 model params, 66 optimizer states (codebook excluded)
2. **Loading attempt:** PyTorch tries to match optimizer states to parameters by **index**
3. **Index mismatch:** Parameter #12 optimizer state tried to load into wrong parameter
4. **Result:** `RuntimeError: size mismatch (3) vs (512) at dimension 3`

## Why the Fix Works

The try-except block in `vq_trainer.py:78-94` catches the mismatch and skips loading optimizer state. This is **safe** because:
- ✅ Model weights (all 67 params including codebook) load correctly
- ✅ Codebook continues updating via EMA as designed
- ✅ Encoder/Decoder start with fresh Adam momentum (minor setback, not critical)
- ✅ Training continues normally from epoch 46

---

## Practical Verification

Run this after a few training steps to verify codebook is updating:

```bash
# Note current codebook values
python3 verify_codebook_update.py

# Train for 100 iterations
# (your training script)

# Check codebook again - values should have changed!
python3 verify_codebook_update.py
```

The codebook values will be different, proving EMA updates are working even without being in the optimizer!

---

## Bottom Line

**Your codebook IS being fine-tuned**, just through a different mechanism (EMA) that doesn't require gradient-based optimization. This is the standard VQ-VAE design and is working as intended.

The codebook learns by:
1. Observing which encoder outputs get assigned to it
2. Moving towards the mean of those outputs via exponential moving average
3. NOT through backpropagation or gradient descent

This is why it's not in the optimizer state, and this is completely normal and expected behavior.
