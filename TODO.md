# TODO — EBM + RL + Transformers

## Literature (reference, not blocking)
- [ ] Read Soft Q-Learning paper (Haarnoja 2017) — the SAC→EBM connection
- [ ] Read Diffusion-QL paper (Wang et al. 2022) — diffusion actor in actor-critic framework
- [ ] Read Consistency Policy paper (Chen et al. 2024) — single-step diffusion inference
- [ ] Read Compositional EBM paper (Du et al. 2020) — energy compositionality theory

## Phase 0: Baseline & Infrastructure
- [x] Build tiny synthetic env (3 turbines, simple wake model) for rapid diffusion actor iteration
- [ ] Train existing SAC baseline on 2-3 layouts, save checkpoints + power metrics
- [ ] Evaluate SAC baseline on held-out OOD layouts, record generalization curves
- [ ] Wrap load surrogate as differentiable PyTorch module (input: state+action, output: scalar load estimate)

## Phase 1: Diffusion Actor
- [x] Implement diffusion denoising network (MLP conditioned on transformer embeddings + timestep)
- [x] Implement forward diffusion process (noise schedule, DDPM/DDIM)
- [x] Implement Diffusion-QL training objective (diffusion loss + Q-value guidance)
- [x] Integrate with existing transformer encoder + critic (actor replacement only)
- [x] Integrate with WindGym for real wind farm training
- [x] Wire cosine noise schedule (was implemented but never connected)
- [x] Add BC weight annealing, action regularization, LR warmup
- [x] Run training improvement sweep (cosine schedule, action reg, BC annealing)
- [ ] Train on real layouts and compare power performance vs SAC Gaussian actor

## Phase 1b: EBT Actor (alternative to diffusion)
- [x] Implement TransformerEBTActor with explicit energy head
- [x] Implement gradient-descent action generation with self-verification
- [x] Training script (ebt_sac_windfarm.py) with WindGym integration
- [ ] Tune hyperparameters (ebt_opt_lr most important per paper)
- [ ] Compare EBT vs diffusion on same layouts

## Phase 2: Safety Composition (headline result)
- [x] Implement classifier guidance: add load surrogate gradient to denoising steps
- [x] Implement per-turbine heterogeneous constraints (PerTurbineYawSurrogate)
- [x] Implement stateful yaw travel budget (YawTravelBudgetSurrogate)
- [x] Demo script for constraint scenarios (scripts/demo_per_turbine_constraints.py)
- [ ] Evaluate power-vs-load tradeoff curves at varying lambda
- [ ] Baseline comparisons:
  - [ ] No constraint (diffusion/EBT actor, power only)
  - [ ] Composed constraint (ours — load surrogate as guidance, no retraining)
  - [ ] Retrained constrained SAC (Lagrangian, retrained per constraint level)
  - [ ] Post-hoc action clipping (naive baseline)
- [ ] Test per-turbine constraint scenario: constrain T1 → show different optimum emerges
- [ ] Test travel budget: show agent converges then holds steady
- [ ] Visualize: energy landscape with and without load guidance

## Phase 3: OOD Generalization
- [ ] Define OOD layout test suite (varying turbine count, spacing, topology)
- [ ] Evaluate diffusion/EBT policy vs. SAC on OOD layouts
- [ ] Test safety composition on OOD layouts — does guidance still work?
- [ ] Ablation: is improved OOD from diffusion actor, transformer, or both?

## Phase 4: Extensions
- [ ] Consistency distillation for single-step inference (if inference speed is a bottleneck)
- [ ] Multiple composed constraints (load + noise + per-turbine limits)
- [ ] Visualize energy landscapes for interpretability

## Infrastructure
- [x] Spring cleaning: remove old notebooks, archive, edge scripts
- [x] Update README for new direction
- [x] Create CLAUDE.md, CONTEXT.md, TODO.md
- [x] Update .gitignore
- [x] Create papers/PAPERS.md — curated literature review
- [x] Decide primary research direction
- [x] Wandb scraping script (scripts/fetch_wandb_results.py)
- [x] Sweep runner script (scripts/run_sweep.py)
