# TODO — EBM + RL + Transformers

## Literature (reference, not blocking)
- [ ] Read Soft Q-Learning paper (Haarnoja 2017) — the SAC→EBM connection
- [ ] Read Diffusion-QL paper (Wang et al. 2022) — diffusion actor in actor-critic framework
- [ ] Read Consistency Policy paper (Chen et al. 2024) — single-step diffusion inference
- [ ] Read Compositional EBM paper (Du et al. 2020) — energy compositionality theory

## Phase 0: Baseline & Infrastructure
- [ ] Train existing SAC baseline on 2-3 layouts, save checkpoints + power metrics
- [ ] Evaluate SAC baseline on held-out OOD layouts, record generalization curves
- [ ] Wrap load surrogate as differentiable PyTorch module (input: state+action, output: scalar load estimate)
- [ ] Build tiny synthetic env (3 turbines, simple wake model) for rapid diffusion actor iteration
- [ ] Update requirements.txt for diffusion dependencies (if needed)

## Phase 1: Diffusion Actor
- [ ] Implement diffusion denoising network (MLP conditioned on transformer embeddings + timestep)
- [ ] Implement forward diffusion process (noise schedule, DDPM/DDIM)
- [ ] Implement Diffusion-QL training objective (diffusion loss + Q-value guidance)
- [ ] Integrate with existing transformer encoder + critic (actor replacement only)
- [ ] Train on synthetic env first, then real layouts
- [ ] Compare power performance: diffusion actor vs. SAC Gaussian actor

## Phase 2: Safety Composition (headline result)
- [ ] Implement classifier guidance: add load surrogate gradient to denoising steps
- [ ] Evaluate power-vs-load tradeoff curves at varying λ (guidance strength)
- [ ] Baseline comparisons:
  - [ ] No constraint (diffusion actor, power only)
  - [ ] Composed constraint (ours — load surrogate as guidance, no retraining)
  - [ ] Retrained constrained SAC (Lagrangian, retrained per constraint level)
  - [ ] Post-hoc action clipping (naive baseline)
- [ ] Test with different load surrogates (if multiple available)
- [ ] Visualize: energy landscape with and without load guidance

## Phase 3: OOD Generalization
- [ ] Define OOD layout test suite (varying turbine count, spacing, topology)
- [ ] Evaluate diffusion policy vs. SAC on OOD layouts
- [ ] Test safety composition on OOD layouts — does guidance still work?
- [ ] Ablation: is improved OOD from diffusion actor, transformer, or both?

## Phase 4: Extensions
- [ ] Consistency distillation for single-step inference (if inference speed is a bottleneck)
- [ ] Multiple composed constraints (load + noise + per-turbine limits)
- [ ] Visualize energy landscapes for interpretability

## Setup (done)
- [x] Spring cleaning: remove old notebooks, archive, edge scripts
- [x] Update README for new direction
- [x] Create CLAUDE.md, CONTEXT.md, TODO.md
- [x] Update .gitignore
- [x] Create papers/PAPERS.md — curated literature review
- [x] Decide primary research direction
