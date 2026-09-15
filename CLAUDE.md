# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is at an early, pre-code stage: it currently contains only `.gitignore`, `LICENSE`, and `instructions_updated.md`. There is no source code, no README, no dependency manifest, and no build/lint/test tooling configured yet. The `.gitignore` is a Python template with Jupyter, Marimo, and Streamlit entries, suggesting the intended stack — but `pyproject.toml`/`requirements.txt`, a test runner (likely pytest), and a linter (likely ruff) still need to be set up as the project is bootstrapped. Don't assume any of these exist without checking.

`instructions_updated.md` at the repo root is the source of the guidance condensed below and is currently untracked in git — it should probably be `git add`ed and committed rather than left floating.

## Purpose

This repo implements an MSc thesis as a reproducible, composable scientific analysis system — a neuroscience analysis library over Allen Institute calcium imaging data (decoding, dimensionality reduction, stimulus controls, neural geometry). The goal is not a fully autonomous scientist but a trustworthy, inspectable library whose components an AI agent can compose and orchestrate.

The expected agent loop for any analysis work: inspect the available data and code → implement one analysis step at a time → make each step reproducible → validate it on synthetic ("vuffel") data → run it on real data → save outputs and provenance → use results to decide the next branch to pursue.

## Core philosophy

Prefer small, deterministic, mostly-pure functions with explicit inputs/outputs, typed scientific contracts, reproducible pipelines, visible assumptions, and filesystem-based artifacts.

Avoid hidden notebook state, monolithic scripts, implicit globals, silent data mutation, and analyses that can't be independently rerun. Same input + same parameters + same software version + same random seed should reproduce the same result.

"The intelligence may live partly in the agent. The reliability must live in the tools."

## Intended architecture

Two complementary organizing schemes are intended once code is added:

- A conventional layout: `src/{data,preprocessing,features,decoding,geometry,dimensionality,statistics,validation,plotting,pipelines}/`, `analysis/<name>/<run_id>/` holding `manifest.json`, `parameters.json`, `metrics.json`, `validation.json`, `provenance.json`, plus `figures/`, `arrays/`, `models/`, and `tests/{unit,scientific,regression}/`. Notebooks are for exploration only (`notebooks/exploratory_only/`); final analyses belong in reusable library code.

- **Functional graph architecture**: an `analysis_code/` directory where each file is one named scientific analysis thread branching from a shared upstream data node — e.g. `load_allen_calcium_data.py`, `animacy_decoding.py`, `pca_eigenspectrum.py`, `jpeg_residualization.py`, `regional_fisher_geometry.py`. Files = scientific branches, functions = nodes within a branch. When extending an existing branch with a downstream step, add the function to that branch's existing file; only create a new file when starting a genuinely different scientific thread. Filenames must describe the scientific content (not `analysis1.py`, `stuff.py`, `final2.py`).

  Every analysis-code file gets a mirrored `plots/<same_name>/` output folder (e.g. `analysis_code/pca_eigenspectrum.py` → `plots/pca_eigenspectrum/`) — this is the intended map from code to figures in place of a separate provenance system. Don't create a new folder per run; reproducibility should come from rerunning the thread, not from accumulating run-specific directories.

## Scientific rigor requirements

- Treat each analysis function as a documented scientific operation: what data it expects, the observational unit, array shapes, assumptions, train/test structure, random-seed behavior, parameters, outputs, known failure modes, and validation procedure.

- **Vuffel testing** — "vuffel" means synthetic/fake data built to test whether an analysis behaves scientifically correctly, kept as separate generator functions from the analysis itself (not mixed in invisibly). Every important analysis needs generators covering:
  - *Positive control* — contains a known planted structure the method should recover (known axis, subspace, correlation, decoding signal).
  - *Negative control* — contains no true target structure; performance should stay near chance and not invent an effect.
  - *Confounded/deceptive control* — an apparent effect driven by a nuisance variable rather than the real target (e.g. image complexity confounded with both neural response and label); the analysis or its control should not be fooled by it.
  - *Edge cases* — class imbalance, singular covariance, tiny sample size, duplicated samples, constant/zero-variance features, leakage, etc. Correct behavior may be to recover the effect, return null, warn, or fail explicitly — silent success is not automatically success.

  Real-data results should not be treated as trustworthy until the relevant vuffel tests pass.

- **Cross-validation and leakage**: for any predictive analysis, explicitly define the training unit, test unit, grouping variables, and whether preprocessing/feature-selection/residualization is fit only within each fold. If the independence structure is unclear, stop and ask rather than inventing a split.

- **Nulls and controls**: prefer analyses with meaningful controls (permutation, trial/stimulus shuffle, circular shift, bootstrap, alternative nuisance models). The null must preserve whatever structure should remain preserved — don't shuffle blindly.

- **Interpretation discipline**: keep prediction, decodability, association, representation, geometry, mechanism, and causality distinct. A successful decoder does not by itself establish mechanism or causal representation — match interpretive language to what the analysis actually supports.

- **Plots** are scientific artifacts, not decorative notebook output. Plotting functions should take explicit results (`plot_fisher_projection(result, labels)`) rather than being embedded in analysis logic, and each figure should make clear what data/split/metric/units are shown and whether the result is training, validation, or held-out.

## Agent workflow for analysis work

**Adding a new analysis:** inspect relevant data structures → inspect existing repo functions and check whether one can be reused → define the scientific question, observational unit, input/output contract, and assumptions/failure modes → implement the smallest reusable function → create vuffel generators and run scientific validation → add unit tests → run on real data → generate plots → save artifacts and provenance → summarize what passed, failed, and remains uncertain. Don't skip straight from prompt to a real-data conclusion.

**Modifying an existing analysis:** before changing anything, inspect its current contract, its tests, and prior artifacts, and determine what scientific behavior must remain unchanged. After modifying, rerun unit tests and vuffel validation, compare against previous outputs, and document intentional differences.

## When to ask instead of deciding alone

Stop and ask the user when: the scientific question is ambiguous, multiple valid estimands exist, the correct observational unit is unclear, a preprocessing choice could materially affect interpretation, a null model has multiple plausible definitions, a result invites causal language, a surprising result could be artifact or genuine discovery, or a new analysis branch would substantially change the thesis's story. Automation should reduce mechanical work, not silently make scientific commitments.
