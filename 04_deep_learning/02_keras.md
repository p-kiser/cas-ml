# Keras

Keras is the high-level API for TensorFlow. Two main ways to build models.

## Sequential API

Simple stack of layers, one input → one output:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## Functional API

More flexible — supports multiple inputs/outputs, branching, shared layers:

```python
inputs = tf.keras.layers.Input(shape=(28, 28))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
```

## Compile and Train

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)
```

`history.history` contains `loss`, `val_loss`, `accuracy`, `val_accuracy` per epoch.

## Custom Training Loop

For full control over the training process (useful for research / custom losses):

```python
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for epoch in range(epochs):
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
```

`GradientTape` records operations so gradients can be computed via backpropagation.

## Inspecting Learned Weights

```python
weights, biases = layer.get_weights()
```

For a dense layer connecting pixels to hidden units, reshape `weights` to `(28, 28)` to visualize which pixels the network pays attention to.

Effective input-to-output weights: `W1 @ W2` — multiply first and second layer weights to see what each output class "looks for" in the input.

## Visualizing Feature Maps (CNNs)

Use a hook / intermediate model to extract conv layer activations:

```python
activation_model = tf.keras.Model(
    inputs=model.input,
    outputs=[layer.output for layer in model.layers if isinstance(layer, Conv2D)]
)
activations = activation_model.predict(sample_image)
```

Plot channels side by side — bright regions show where filters fired strongly.
