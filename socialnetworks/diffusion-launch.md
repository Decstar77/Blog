# Diffusion Series — Social Posts

Covers the two diffusion math articles (forward + backward pass) and the class-conditional DDPM on MNIST. Gif of the MNIST denoising rollout to be attached manually.

---

## X (Twitter)

**Title (first post):**
I finally sat down and worked through the math behind diffusion models — every step, no skips.

**Thread:**

I finally sat down and worked through the math behind diffusion models. Wrote it up as two posts (forward + backward pass) and a class-conditional DDPM on MNIST to prove it actually works.

The forward pass surprised me. It's not "just add noise." There's a closed-form solution jumping to any x_t directly from x_0 https://blog.decstar77.com/diffusion-forward-process.html

The backward pass is where it gets painful. Most explanations skip steps. I tried to fill in every gap 
https://blog.decstar77.com/diffusion-backward-pass.html

4/ Then I trained a tiny U-Net with FiLM-modulated ResNet blocks on MNIST. Sinusoidal time embedding + learned class embedding. Pick a digit, watch noise turn into a number.

5/ Links:
• Forward pass math → [forward url]
• Backward pass math → [backward url]
• Class-conditional DDPM on MNIST (live demo) → [project url]

---

## Bluesky

**Post:**

Spent the last few weeks unpacking the math behind diffusion models — wrote up the forward and backward Gaussian passes step by step, then trained a class-conditional DDPM on MNIST so you can pick a digit and watch noise turn into a number.

I tried to write the version I wish I'd had: no skipped algebra, the "why" behind the square roots, and the Bayes derivation laid out in full.

Articles + live demo:
• Forward pass → [forward url]
• Backward pass → [backward url]
• MNIST DDPM → [project url]

---

## Reddit — r/learnmachinelearning

**Title:**
I wrote up the full math behind diffusion models (forward + backward pass) with no skipped steps — plus a small DDPM on MNIST you can play with

**Body:**

Hey all,

I've been learning diffusion models and got frustrated with how many explanations either skip algebra or assume you already know the trick they're about to use. So I wrote up the version I wish I'd had when I started.

Two articles + one project:

**1. The Math of the Gaussian Forward Pass**
Builds up from the standard normal to the closed-form expression for x_t given x_0. Covers:
- Where the sampling form `x = μ + σε` actually comes from (change of variables, not magic)
- Why the noise is scaled by √β_t, not β_t
- Deriving the cumulative-product form so you can jump to any timestep in one operation
- Why both coefficients are square roots — the variance-preservation argument written out

**2. The Math of the Gaussian Backward Pass**
The painful one. I tried to fill in every step that other explanations leave as "exercise to the reader":
- Why we condition on x_0 even though we don't have it at inference time
- Bayes' theorem in three-variable form
- The Markov assumption and what it lets us drop
- How x_0 gets eliminated later via the noise-prediction substitution

**3. Class-Conditional DDPM on MNIST**
To make sure I actually understood it, I built one. Small U-Net with FiLM-modulated ResNet-style blocks. Sinusoidal time embedding + learned class-label embedding. There's a live demo where you pick a digit and watch the model denoise pure Gaussian noise into a handwritten sample.

Links:
- Forward pass: [forward url]
- Backward pass: [backward url]
- MNIST DDPM (live demo): [project url]

Happy to take corrections — if I got something wrong or skipped something I shouldn't have, let me know and I'll fix it.
