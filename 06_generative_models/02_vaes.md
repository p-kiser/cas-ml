# Variational Autoencoders (VAEs)

## Concept

A VAE learns a compressed, continuous latent space that can be sampled to generate new data.

- **Encoder:** maps an image → latent distribution (mean μ and variance σ)
- **Decoder:** maps a latent vector z → reconstructed image
- **Sampling:** draw z ~ N(μ, σ²) during training (reparameterization trick allows backprop through sampling)

The latent space is regularized to be approximately Gaussian, so you can sample from N(0, I) at generation time.

## Why Latent Spaces are Useful

The latent space is **continuous and structured**:

- **Interpolation:** encode two images A and B, linearly blend their latents → decode each step → smooth visual morph
- **Latent directions:** moving along a direction in latent space changes a specific visual attribute (pose, lighting, expression)
- **Sampling:** draw random latent vectors and decode → novel images

```python
# Interpolation
alphas = torch.linspace(0, 1, steps=8)
for alpha in alphas:
    z = (1 - alpha) * z_a + alpha * z_b
    img = decoder(z)
```

## Stable Diffusion VAE

The course uses `stabilityai/sd-vae-ft-mse` — a VAE pretrained on faces as part of Stable Diffusion.

```python
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

# Encode image to latent
latent = vae.encode(image_normalized).latent_dist.sample() * SCALE_FACTOR

# Decode latent to image
image = vae.decode(latent / SCALE_FACTOR).sample
image = (image.clamp(-1, 1) + 1) / 2  # to [0, 1]
```

`SCALE_FACTOR = 0.18215` is specific to Stable Diffusion's latent space scaling.

**Important:** this VAE expects latents from Stable Diffusion's U-Net, not pure Gaussian noise. To generate faces, sample around encoded real face latents:

```python
base = vae.encode(real_face).latent_dist.sample()
noisy = base + torch.randn_like(base) * noise_scale
new_face = vae.decode(noisy / SCALE_FACTOR).sample
```

## Latent Grid Exploration

To understand what dimensions control: pick two directions in latent space (e.g., difference between two faces), sweep them as X/Y axes, decode every grid cell. Reveals which directions correspond to pose, expression, lighting, etc.
