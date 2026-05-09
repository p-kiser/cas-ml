# GANs (Generative Adversarial Networks)

## Core Idea

Two networks trained in opposition:

- **Generator (G):** takes random noise z, produces fake images
- **Discriminator (D):** tries to distinguish real images from fakes

**Training objective:**
- D wants to output 1 for real, 0 for fake
- G wants D to output 1 for its generated images (fool D)

Over time, G improves until D can no longer reliably tell real from fake → generator produces realistic images.

## GAN Variants

### CycleGAN — Unpaired Image Translation

Translates images between two domains (e.g., summer ↔ winter, horse ↔ zebra) **without needing paired training data**.

Uses two generators and two discriminators:
- G_AB: domain A → domain B
- G_BA: domain B → domain A

**Cycle-consistency loss:** `G_BA(G_AB(x)) ≈ x` — translating back should recover the original. This forces the generators to preserve content while changing style.

```python
from cvcourse.models.gans import load_cyclegan_generator

generator = load_cyclegan_generator("summer2winter")
output = generator(input_tensor)  # input normalized to [-1, 1]
```

### ESRGAN — Super-Resolution

Upscales low-resolution images by 4× using a GAN. The generator hallucinates high-frequency detail (textures, edges) that was lost in downsampling.

**vs. bicubic interpolation:** bicubic is blurry. ESRGAN produces sharper, more realistic textures because the discriminator penalizes blurry outputs that don't look like real high-res images.

```python
from cvcourse.models.gans import get_esrgan_session, run_esrgan

session = get_esrgan_session()   # ONNX Runtime for CPU inference
high_res = run_esrgan(session, low_res_image)
```

## Training Challenges

GANs are notoriously tricky to train:
- **Mode collapse:** generator learns to produce only a few outputs that fool D
- **Training instability:** G and D need to stay in balance; if one gets too strong, the other stops learning
- Solutions: gradient penalty (WGAN-GP), spectral normalization, progressive growing

## GAN vs VAE vs PixelCNN

| | PixelCNN | VAE | GAN |
|---|---|---|---|
| Quality | Good | Blurry | Sharp |
| Diversity | Good | Good | Mode collapse risk |
| Latent space | None | Structured | Unstructured noise |
| Training | Stable | Stable | Unstable |
| Speed | Very slow sampling | Fast | Fast |
