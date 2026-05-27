# VAE Series — Social Posts

Covers the VAE math article and the two VAE projects (RGB colors with a 2D latent, and MNIST with a 32-D latent + conv encoder/decoder). Gif to be attached manually — latent-space walk for the colors one, or the random-samples grid for MNIST.

---

## X (Twitter)

**Title (first post):**
After diffusion I kept seeing VAEs everywhere in the literature, so I sat down and worked through the math.

**Thread:**

After diffusion I kept seeing VAEs everywhere in the literature, so I sat down and worked through the math. Wrote it up, then built two small projects to actually feel what the latent space does.

The core trick: don't encode to a point, encode to a *distribution*. Mean + log-variance, sample with the reparameterisation trick so gradients still flow. The KL term is what makes the latent space continuous. https://blog.decstar77.com/vae-math.html

Project 1 — a VAE that compresses RGB into a 2D latent plane. Click anywhere on the plane, the decoder turns the coordinate into a colour. Tiny model, but you can literally see the continuity the KL term buys you. https://blog.decstar77.com/project-vae-colors.html

Project 2 — same idea on MNIST. Conv encoder down to a 32-D latent, conv decoder back up. Sample z ~ N(0, I), decode, get a digit that never existed. [project url]

Biggest "oh" moment: log-variance instead of variance (numerical stability + free positivity), and β on the KL term so reconstruction doesn't get crushed early in training.

---

## Bluesky

**Post:**

After spending a while on diffusion models I kept running into VAEs in the literature, so I sat down and wrote up the math — ELBO derivation, the reparameterisation trick, why we predict log-variance instead of variance, the whole thing.

Then two small projects to actually see it work:
- An RGB-colour VAE with a 2D latent plane you can click around in
- An MNIST VAE with a 32-D latent and a conv encoder/decoder

The thing that finally made it click for me: the encoder outputs a *distribution*, not a point, and the KL term is what stitches the latent space together into something continuous you can sample from.

Links:
- Math write-up → [vae-math url]
- Colors VAE (live demo) → [vae-colors url]
- MNIST VAE → [vae-mnist url]

---

## Reddit — r/learnmachinelearning

**Title:**
I wrote up the math behind VAEs (ELBO, reparameterisation, the KL term) and built two small projects — a 2D-latent colour VAE and an MNIST VAE

**Body:**

Hey all,

Coming off a few weeks of diffusion-model math, I kept running into VAEs in the literature and decided to give them the same step-by-step treatment. Wrote up the math, then built two projects so I could feel what the latent space actually does.

**1. The Math Behind Variational Autoencoders**

The version I wish I'd had when I started. Covers:
- Why $p(x) = \int p(x \mid z) p(z) \, dz$ is the right starting point, and why it's intractable
- The encoder as an approximation to the true posterior $p(z \mid x)$
- Deriving the ELBO from $\log p(x)$ — multiply and divide by $q_\phi(z \mid x)$, apply Jensen's inequality
- The reparameterisation trick: why we sample $z = \mu + \sigma \odot \varepsilon$ instead of sampling $z$ directly, and what this does for gradients
- Why the encoder predicts *log-variance* instead of variance
- The KL term as the regulariser that makes the latent space continuous

**2. Project — A VAE on Colors**

The smallest VAE I could think of. Encoder takes an RGB triplet, decoder reconstructs it, latent space is 2D so you can literally plot it. There's a live demo where you click anywhere on the latent plane and the decoder turns that coordinate into a colour. Useful because you can *see* the KL term doing its job — nearby points decode to similar colours.

**3. Project — A VAE on MNIST**

Stepping up: conv encoder down to a 32-D latent, conv decoder back up via upsample + conv. β-weighted KL so the reconstruction term doesn't get crushed in the first few epochs. Sample $z \sim \mathcal{N}(0, I)$, decode, get a digit. They're a bit blurry — which is itself a useful lesson about why later work moved to discrete latents and diffusion-based decoders.

Links:
- Math write-up: [vae-math url]
- Colors VAE (live demo): [vae-colors url]
- MNIST VAE: [vae-mnist url]

Happy to take corrections — if I got something wrong, tell me and I'll fix it.
