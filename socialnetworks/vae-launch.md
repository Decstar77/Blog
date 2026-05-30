# VAE Color Project — Social Posts

Covers the RGB color VAE: a 3-dim → 2-dim VAE that learns to separate red, green, and blue in latent space. Latent space scatter plot image to be attached manually.

---

## X (Twitter)

**Title (first post):**
I built the smallest useful VAE I could think of a RGB (3D) color value through a 2D latent space

**Thread:**

I built the smallest useful VAE I could think of: 3 colour inputs, 2 latent dims, 3 colour outputs. Trained on blobs of near-pure red, green, and blue.

Without any class labels, the latent space separates into three clean clusters — one per colour. That's the model discovering structure on its own.

2/ Getting there wasn't clean. My first dataset was fully random RGB — uniform colours with no structure. The KL term overwhelmed reconstruction and the encoder collapsed. The model learned to ignore the input and just output grey (the mean of random colour).

This is called posterior collapse, and it's a well-known failure mode.

3/ Two fixes:
- Switch to a structured dataset: three tight blobs of red, green, blue with small noise
- Weight the KL term with β = 0.1 so reconstruction can dominate early

The β trick is from β-VAE. It's a lever between "tight latent space" and "good reconstructions."

4/ The reparameterization trick is what makes it trainable. Instead of sampling z ~ N(μ, σ²) directly (no gradient), you sample ε ~ N(0,I) and compute z = μ + σε. Gradients flow through μ and σ. The randomness is pushed outside the computation graph.

5/ Live demo — click anywhere in the latent space and see what colour the decoder produces:
• Color VAE → [project url]
• Blog post → [blog url]

---

## Bluesky

**Post:**

Built a VAE on synthetic RGB colours — 3 inputs, 2 latent dims, 3 outputs. Small enough to reason about completely.

When it works, the latent space organises red, green, and blue into three separate clusters with no class labels. The model finds the structure by itself.

Getting there required hitting posterior collapse first. My original dataset was random uniform colours — no structure to encode. The KL term dominated and the encoder gave up, outputting grey (the mean). Fixed it by using tight colour blobs and weighting the KL with β = 0.1.

There's a live demo where you can click anywhere in the 2D latent space and see the decoded colour. Worth a look if you're trying to build intuition for what a latent space actually represents.

• Project + demo → [project url]
• Blog post → [blog url]

---

## Reddit — r/learnmachinelearning

**Title:**
I built a tiny VAE on RGB colours (3 inputs → 2 latent dims) to build intuition — hit posterior collapse along the way

**Body:**

Hey all,

I wanted to understand what the VAE latent space is *actually* doing before touching image models. So I built the smallest version I could: a VAE trained on synthetic red, green, and blue colour blobs, compressing 3D RGB down to a 2D latent space.

**The model**

- Encoder: Linear(3→2) + ReLU → separate heads for μ and log σ²
- Latent: 2 dims (small enough to visualise entirely)
- Decoder: Linear(2→3) + Tanh
- Data: 2001 points in three tight clusters (near-pure red, green, blue) with small noise, mapped to [−1, 1]

When it works, the latent space separates the three colour families into three clusters — no class labels, just the VAE finding structure.

**Posterior collapse**

My first attempt used fully random RGB colours. The KL divergence term overwhelmed the reconstruction loss and the encoder collapsed — it learned to output something close to N(0,I) regardless of input. The decoded output converged to grey (the mean of random colour). Classic posterior collapse.

Two things fixed it:
- **Structured data**: three distinct colour blobs give the encoder something worth encoding
- **β weighting**: loss = MSE + β × KLD with β = 0.1, so reconstruction drives training early on

The β term comes from β-VAE. Setting it below 1 relaxes the constraint on the latent distribution, which helps when the KL term would otherwise dominate.

**The loss in full**

```
MSE = mean((pred - target)²)
KLD = −0.5 × mean(1 + log_var − mu² − exp(log_var))
loss = MSE + 0.1 × KLD
```

KLD measures how far the encoder's posterior q(z|x) is from the unit Gaussian prior. Minimising it keeps the latent space regularised — which is what makes sampling from N(0,I) at inference time produce something valid.

**Results**

The latent scatter plot shows three clean clusters, each coloured by its original RGB value. The decoded colour grid shows smooth interpolation across the 2D space — red, green, blue in their corners, transitions in between.

There's a live demo where you can click anywhere in the latent space and see the decoded colour.

Links:
- Color VAE (live demo): [project url]
- Blog post: [blog url]

Happy to take corrections — if I got the KL derivation or the β-VAE framing wrong, let me know.
