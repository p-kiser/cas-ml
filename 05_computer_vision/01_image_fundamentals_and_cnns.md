# Image Fundamentals and CNNs

## Image Representation

Images are tensors shaped `[C, H, W]` (channels, height, width):
- Grayscale: C=1
- Color (RGB): C=3
- Pixel values are floats in [0, 1] after normalization (`transforms.ToTensor()`)

Batches add a leading dimension: `[B, C, H, W]`

## Key Datasets

| Dataset | Size | Channels | Classes |
|---|---|---|---|
| MNIST | 28×28 | 1 | 10 digits |
| FashionMNIST | 28×28 | 1 | 10 clothing |
| CIFAR-10 | 32×32 | 3 | 10 objects |
| Imagenette | 160px | 3 | 10 (ImageNet subset) |
| COCO | varies | 3 | 80 (detection) |

## Data Pipeline

```
Dataset → Transform (normalize, augment) → DataLoader (batch + shuffle)
```

1. Dataset returns a single `[C, H, W]` tensor
2. Transforms normalize pixels to [0, 1], optionally augment
3. DataLoader stacks into batches `[B, C, H, W]` and labels into `[B]`

## CNN Architecture

Convolutional Neural Networks exploit spatial structure instead of treating each pixel independently.

**Building blocks:**

| Layer | What it does |
|---|---|
| Conv2d | Slides a learned kernel over the image, produces feature maps |
| ReLU | Non-linearity after each conv |
| MaxPool2d | Downsamples by taking max in each region |
| AdaptiveAvgPool | Forces a fixed output size |
| Dropout | Randomly zeroes neurons to prevent overfitting |
| Flatten | Collapses spatial dims for the classifier head |
| Linear | Fully connected classification layer |

**Typical structure:**
```
[Conv → ReLU → Pool] × N  →  Flatten  →  Linear  →  Softmax
      feature extractor              classifier
```

**Parameter count:**
- Conv layer: `(out_ch × in_ch × k_h × k_w) + out_ch`
- Linear layer: `(in_features × out_features) + out_features`

## Kernels

A kernel (filter) is a small weight matrix (e.g., 3×3) that slides over the input and computes a dot product at each position.

**Handcrafted examples:**
- Edge detection: `[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]`
- Sobel (horizontal edges): `[[-1,0,1],[-2,0,2],[-1,0,1]]`
- Blur: uniform kernel (all values 1/9)
- Sharpen: `[[0,-1,0],[-1,5,-1],[0,-1,0]]`

In a trained CNN, kernels are learned automatically. Early layers detect edges/textures, deeper layers detect complex patterns.

## Feature Maps

After a convolution, the output is a feature map — a spatial map of where a particular pattern was detected. Use forward hooks to visualize them:

```python
activations = []
handle = model.features[0].register_forward_hook(
    lambda m, i, o: activations.append(o.detach().cpu()[0])
)
model(x)
handle.remove()
```

## Training

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    for images, labels in train_loader:
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Monitor train vs. val loss to detect overfitting. Spot-check predictions by visualizing `pred | true` on a batch.
