# Neural Networks

## The DL Hierarchy

- **AI:** a program that can sense, reason, adapt
- **ML:** algorithms whose performance improves with more data
- **DL:** subset of ML using multilayer neural networks trained on large data

## Architecture

A neural network is a stack of layers. Each layer transforms its input via learned weights:

```
output = activation(W · input + b)
```

**Fully connected (Dense) layer:** every input neuron connects to every output neuron.

**Parameter count:** `(in_features × out_features) + out_features`

**Common activation functions:**

| Function | Output | Use case |
|---|---|---|
| ReLU | [0, ∞) | Hidden layers (default) |
| Sigmoid | (0, 1) | Binary output |
| Softmax | probabilities summing to 1 | Multi-class output |

## Training

1. **Forward pass:** compute predictions from input to output
2. **Compute loss:** measure how wrong the predictions are
3. **Backward pass (backprop):** compute gradients via chain rule using `GradientTape`
4. **Update weights:** optimizer steps in the direction that reduces loss

**Loss functions:**
- `sparse_categorical_crossentropy`: integer class labels, multi-class
- `categorical_crossentropy`: one-hot encoded labels, multi-class
- `binary_crossentropy`: binary classification
- `mse`: regression

**Optimizers:**
- `Adam`: adaptive learning rate, default choice, works well out of the box
- `SGD`: stochastic gradient descent, more manual tuning needed

## Regularization Techniques

**Dropout:** randomly zeroes a fraction of neurons during training → prevents co-adaptation, reduces overfitting. Disabled during inference.

```python
tf.keras.layers.Dropout(0.2)
```

**L1/L2 weight regularization:** adds penalty to loss directly on layer weights.

## Diagnosing Training

Plot train vs. val loss and accuracy over epochs:
- Both decreasing: learning correctly
- Val loss increasing while train loss decreases: overfitting → add dropout, reduce model size, get more data
- Both flat / high: underfitting → increase model capacity, train longer, adjust learning rate

## CNNs (Convolutional Neural Networks)

For image data. Replace dense layers with convolutions that exploit spatial structure.

```
[Conv2D → ReLU → MaxPool] × N  →  Flatten  →  Dense  →  Softmax
```

**Conv2D:** slides a small kernel over the image, each output pixel = weighted sum of a local patch. Learns to detect edges, textures, shapes.

**MaxPooling2D:** takes the maximum value in each region → downsamples, provides translation invariance.

**Parameter count per conv layer:** `(out_ch × in_ch × k_h × k_w) + out_ch`

Feature maps from early layers detect low-level features (edges). Deeper layers detect complex patterns (faces, objects).
