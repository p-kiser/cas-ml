# Autoregressive Models (PixelCNN)

Autoregressive models generate data one element at a time, conditioning each step on all previous outputs.

## Discriminative vs Generative

| | Discriminative | Generative |
|---|---|---|
| Goal | label from sample | sample from label (or distribution) |
| Output | one label | one of many possible outputs |
| Example | image classifier | image generator |

## PixelCNN

Generates images one pixel at a time, left-to-right, top-to-bottom. Each pixel is predicted from all previously generated pixels — this is the **autoregressive property**.

**Training:** treat each pixel as a Bernoulli variable (0 or 1). Minimize Binary Cross-Entropy:
```
loss = BCEWithLogitsLoss(predicted_logits, true_pixel_values)
```

**Sampling:** start with an all-zero canvas, fill pixels sequentially:
```python
for row in range(28):
    for col in range(28):
        logits = model(canvas)
        prob = torch.sigmoid(logits[:, :, row, col])
        canvas[:, :, row, col] = torch.bernoulli(prob)
```

Slow: generating a 28×28 image requires 784 sequential forward passes.

## Masked Convolutions

Standard convolutions see all neighboring pixels — this would leak future pixels into past predictions.

**Masked convolutions** zero out weights for future pixels:

- **Type A (first layer):** blocks current pixel + all future pixels — strictly causal
- **Type B (later layers):** allows current pixel (already partially computed via previous layers) + blocks future pixels

```python
class MaskedConv2d(nn.Conv2d):
    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, ...)
```

The mask is built once at init: zeros in the top-left causal region (type B includes the center, type A does not).

## Architecture

```
MaskedConv2d (type A)
→ [MaskedConv2d (type B) → ReLU] × N residual blocks
→ MaskedConv2d 1×1 (type B)
→ sigmoid → pixel probabilities
```

Wider/deeper networks generate higher-quality samples but train slower.
