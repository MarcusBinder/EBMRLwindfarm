# Research Context: EBM + RL + Transformers for Wind Farm Control

## What are Energy-Based Models (EBMs)?

Energy-Based Models assign a scalar energy to each configuration of variables. Low energy = high compatibility/probability. Unlike explicit density models, EBMs don't need to normalize over the full space — they just need to assign *relative* energies. This makes them flexible for modeling complex, multimodal distributions over high-dimensional spaces.

Key properties:
- **Unnormalized**: E(x) is a scalar; no partition function needed for many uses
- **Multimodal**: Can represent multiple modes without mode collapse
- **Composable**: Energies add — E_total(x) = E_1(x) + E_2(x) enables compositional reasoning
- **Implicit**: Define "what's good" rather than "how to generate it"

## Why Combine EBMs with RL?

Standard RL learns a policy π(a|s) that maps states to action distributions. EBMs offer an alternative: learn an energy function E(s, a) over state-action pairs, then derive actions by minimizing energy.

Potential advantages for wind farm control:
1. **Multimodal actions** — Multiple good yaw configurations may exist for a given wind condition. EBMs naturally represent this without mode collapse (unlike Gaussian policies).
2. **Compositional objectives** — Combine separate energy terms (power, fatigue, wake, turbulence) by addition. Add/remove objectives without retraining from scratch.
3. **Implicit planning** — Energy minimization over action sequences enables implicit lookahead without explicit model-based planning.
4. **Uncertainty** — Energy landscape curvature provides natural uncertainty estimates.

## How the Existing Infrastructure Fits

The current codebase already solves several hard problems that transfer directly:

| Component | EBM Role |
|-----------|----------|
| Transformer encoder | Backbone for E(s,a) — processes variable-length turbine sequences |
| Positional encodings | Encode spatial structure of turbine layouts |
| Profile encodings | Encode wake interaction patterns |
| Wind-relative frame | Canonical coordinate system for invariant energy functions |
| Replay buffer | Off-policy data for contrastive energy training |
| Multi-layout env | Diverse training data for generalization |

The key change: instead of Actor → actions, train an energy network E_θ(s, a) and derive actions via gradient-based optimization (Langevin dynamics, MCMC, etc.).

## The SAC ↔ EBM Connection

A critical insight: **SAC is already an EBM-adjacent method.** The maximum entropy RL framework defines the optimal policy as:

```
π*(a|s) ∝ exp(Q*(s,a) / α)
```

This is a Boltzmann distribution — the canonical EBM policy. SAC's predecessor, **Soft Q-Learning** (Haarnoja et al. 2017), made this explicit: it trained an energy function E(s,a) = -Q(s,a) and sampled actions via amortized Stein Variational Gradient Descent (SVGD). SAC (Haarnoja et al. 2018) replaced SVGD with a reparameterized Gaussian actor for stability and speed.

What SAC gained: stable training, fast inference, practical scalability.
What SAC lost: multimodal action distributions, explicit energy landscape, compositionality.

**Our research isn't "applying EBMs to a new domain" — it's completing the circle.** We're making the implicit EBM structure in SAC explicit again, and leveraging EBM-specific properties (compositionality, energy landscapes, multimodality) that SAC's Gaussian approximation discards.

## Research Thesis

**Energy-based policies for wind farm control enable (1) post-hoc safety composition via external load surrogates and (2) better OOD layout generalization — without retraining.**

We train a diffusion-based policy (replacing SAC's Gaussian actor) on power maximization across diverse layouts. At deployment, operators can plug in any differentiable constraint — load surrogates, noise limits, turbine-specific maintenance constraints — as an additional energy term. The policy respects the new constraint immediately, without retraining.

### Why This Matters

Standard RL approaches bake objectives into the reward function at training time. If an operator wants to limit fatigue loads during a storm, prioritize a specific turbine near end-of-life, or satisfy a new noise regulation — they retrain. With energy-based policies, these constraints compose at deployment:

```
E_total(s,a) = E_policy(s,a) + λ · L(s,a)
                                ↑
              load surrogate, noise model, or any differentiable constraint
              added at deployment — never seen during training
```

This is possible because action generation in energy-based policies involves *sampling from an energy landscape* (via diffusion denoising or Langevin dynamics). Reshaping that landscape at test time reshapes the actions.

### Two Research Questions

**RQ1 — Post-hoc safety composition:** Can a diffusion policy trained only on power maximization respect load constraints at deployment by composing a load surrogate as classifier guidance during action generation?

- Evaluation: power-vs-load tradeoff curves at varying λ. Compare against constrained SAC (retrained per constraint) and post-hoc action clipping.
- The key result: comparable safety with zero retraining cost.

**RQ2 — OOD layout generalization:** Does the energy-based formulation improve zero-shot transfer to unseen farm layouts compared to standard SAC?

- Hypothesis: energy landscapes encode "what's good" (relative quality) rather than "what to do" (absolute actions), which may transfer better.
- Evaluation: performance degradation curves on held-out layouts at increasing distance from training distribution.
- Supporting result for RQ1 — if the policy also generalizes better, the safety composition works on unseen layouts too.

## Implementation Architecture

The change to the existing codebase is surgical — only the actor is replaced:

```
EXISTING (SAC):
  Transformer encoder → per-turbine embeddings → MLP actor head → Gaussian(μ, σ) → yaw actions

PROPOSED (Diffusion actor):
  Transformer encoder → per-turbine embeddings → Diffusion denoiser(noisy_a, t, embedding) → yaw actions
                                                  ↑
                                                  At deployment, add: - λ·∇_a L(s, a_t) to each denoising step
```

What stays the same:
- Transformer encoder backbone (turbines as tokens)
- All positional/profile encodings
- Critic network (Q-function guides diffusion via Diffusion-QL objective)
- Replay buffer, multi-layout training, wind-relative frame

What changes:
- Actor: Gaussian head → diffusion denoising network (small MLP per turbine, conditioned on transformer embeddings + diffusion timestep)
- Action generation: single forward pass → K denoising steps (mitigated later by consistency distillation)
- Safety composition: at deployment, classifier guidance from load surrogate modifies denoising

## Implementation Phases

### Phase 0: Baseline & Infrastructure
- Train existing SAC on 2-3 layouts, record power metrics
- Evaluate on held-out OOD layouts to establish generalization baseline
- Wrap a load surrogate as a differentiable PyTorch module
- Build a tiny synthetic environment (3 turbines, simple wake) for rapid iteration

### Phase 1: Diffusion Actor
- Replace Gaussian actor head with diffusion denoising network
- Integrate with SAC critic via Diffusion-QL training objective
- Train on same layouts as baseline
- Target: match or approach SAC power performance

### Phase 2: Safety Composition
- Add classifier guidance from load surrogate during inference
- Evaluate power-vs-load tradeoff at varying λ
- Compare: (a) no constraint, (b) composed constraint (ours), (c) retrained constrained SAC, (d) post-hoc action clipping
- This is the headline result

### Phase 3: OOD Generalization
- Evaluate diffusion policy on held-out layouts
- Compare generalization curves: SAC vs. diffusion policy
- Test safety composition on OOD layouts — does it still work?

### Phase 4: Extensions (if results are positive)
- Consistency distillation for single-step inference (Chen et al. 2024)
- Multiple composed constraints (load + noise + turbine-specific)
- Different load surrogate models (analytical, learned, hybrid)

## Future Direction: Sim-to-Real via Energy Composition

The classifier guidance mechanism used for safety composition is general — it works with any differentiable correction term. This opens a potential path for sim-to-real transfer: train a base policy in simulation, then compose a learned residual correction from real operational (SCADA) data at deployment.

```
E_deployed(s,a) = E_sim(s,a) + E_correction(s,a)
```

The correction network would be small (it only models the sim-to-real gap, not the full policy) and could be updated online as more data arrives. This could address multiple types of distribution shift — turbine type mismatch, flow model inaccuracies, terrain effects — through the same composition mechanism. Worth exploring once the core safety composition (Phase 2) is validated.

## Open Questions

- **Load surrogate**: What load surrogates are available? Do they have useful gradients? If not, can we fit a small neural net surrogate?
- **Diffusion steps**: How many denoising steps are needed? Fewer = faster but noisier. How does this interact with classifier guidance strength?
- **Guidance scale**: How sensitive is safety composition to λ? Is there a regime where power drops minimally but loads reduce significantly?
- **Evaluation**: How to define "OOD distance" between layouts? Turbine count difference? Spacing ratio? Topology change?
- **Inference speed**: Is K=10-20 denoising steps fast enough for the training loop, or do we need consistency distillation from the start?
