https://chatgpt.com/g/g-p-68ed63f948108191bc3cff369cc0b2d5-agenticai/c/6aa96fd4-fd9c-83ed-9c7b-84ccbcaee383

claude --permission-mode auto

# Thesis Agent Instructions

## Purpose

This repository implements the MSc thesis as a reproducible, composable scientific analysis system.

The goal is **not** to build a fully autonomous scientist. The goal is to build a trustworthy analysis library whose components can be inspected, tested, composed, and orchestrated by an AI agent.

The agent should behave like a scientific workflow engineer:

1. inspect the available data and code,
2. propose or implement one analysis step at a time,
3. make each step reproducible,
4. test each scientific function on synthetic "vuffel" data,
5. run the validated function on real data,
6. save all outputs and provenance,
7. use the results to decide what branch of analysis to pursue next.

---

# Core philosophy

Use **simple, composable patterns** rather than complex agent frameworks.

Prefer:

- small deterministic functions,
- explicit inputs and outputs,
- pure or mostly pure functions,
- typed scientific contracts,
- reproducible pipelines,
- visible assumptions,
- explicit validation,
- filesystem-based artifacts,
- human-readable provenance.

Avoid:

- hidden notebook state,
- large monolithic scripts,
- implicit global variables,
- opaque multi-agent architectures,
- silent data mutation,
- analyses that cannot be independently rerun,
- scientific conclusions based only on plausible-looking output.

The intelligence may live partly in the agent.

The reliability must live in the tools.

---

# Repository design

The thesis should evolve from disconnected notebooks into a unified Python library.

A suggested structure:

```text
src/
  data/
  preprocessing/
  features/
  decoding/
  geometry/
  dimensionality/
  statistics/
  validation/
  plotting/
  pipelines/

analysis/
  <analysis_name>/
    <run_id>/
      manifest.json
      parameters.json
      metrics.json
      validation.json
      provenance.json
      figures/
      arrays/
      models/

tests/
  unit/
  scientific/
  regression/

notebooks/
  exploratory_only/
```

Notebooks may be used for exploration, but final analyses should live in reusable library code.

---


# Functional graph architecture

The thesis codebase should be organized as a **functional programming graph**.

The graph begins from one or more data-loading or canonical-data nodes. From there, scientific analyses branch into separate **analysis threads**.

Each analysis thread should live in one informatively named Python file.

Example:

```text
analysis_code/
  load_allen_calcium_data.py
  animacy_decoding.py
  pca_eigenspectrum.py
  jpeg_residualization.py
  regional_fisher_geometry.py
```

## One branch, one file

When two analyses represent different branches from the same upstream data node, they should live in separate files.

For example:

```text
canonical neural responses
    ├── animacy_decoding.py
    ├── pca_eigenspectrum.py
    ├── jpeg_residualization.py
    └── regional_fisher_geometry.py
```

Each file is therefore a **thread through the analysis graph**.

Its name should describe the scientific content clearly enough that the repository structure itself communicates what analyses exist.

Avoid vague filenames such as:

```text
analysis1.py
test_new.py
stuff.py
final2.py
```

Prefer names such as:

```text
animacy_decoding.py
vit_feature_control.py
pca_powerlaw_analysis.py
regional_fisher_axis.py
```

## Extensions stay in the same thread file

Any node or function inside an existing analysis thread may be extended with additional downstream functions.

If the new work is a continuation of the same scientific branch, keep it in the same file.

Example:

```text
animacy_decoding.py

load_response_matrix()
    ↓
fit_logistic_decoder()
    ↓
evaluate_cross_validation()
    ↓
compare_brain_regions()
    ↓
plot_region_accuracy()
```

If `compare_brain_regions()` is added later as a continuation of the animacy-decoding branch, it belongs in `animacy_decoding.py`.

Do not create a new file merely because a branch gains another function.

Create a new file when the analysis branches into a genuinely different scientific thread.

In this architecture:

```text
files = scientific branches / threads
functions = nodes within those branches
```

A function may consume outputs from earlier functions in the same thread.

A new thread may start from raw data, canonicalized data, or a stable output from an upstream data-preparation node.

The goal is to make the scientific dependency graph visible directly from the code organization.

---

# Functional style

Prefer functions with explicit input/output behavior.

Good:

```python
result = fisher_axis(X_train, y_train, regularization=1e-3)
```

Avoid functions that:

- load arbitrary files internally,
- depend on notebook variables,
- modify global state,
- save files as an undocumented side effect,
- depend on execution order outside the function.

Whenever practical:

```text
same input
+ same parameters
+ same software version
+ same random seed
= same result
```

---

# Scientific operation contract

Each analysis method should be treated as a **scientific operation**.

Every scientific operation should define:

```python
AnalysisSpec(
    name="...",
    inputs=[...],
    outputs=[...],
    assumptions=[...],
    validations=[...],
)
```

At minimum, document:

- what data the operation expects,
- the observational unit,
- array dimensions / shapes,
- assumptions,
- train/test structure if relevant,
- random seed behavior,
- parameters,
- outputs,
- known failure modes,
- validation procedures.

The agent should inspect this contract before running the operation.

---

# Vuffel testing

"Vuffel" means synthetic, fake, controlled data constructed to test whether an analysis behaves scientifically correctly.

Every important scientific operation should have associated vuffel generators.

Do **not** mix synthetic-data generation invisibly inside the real-data analysis function.

Instead separate:

```python
analysis_function(...)
make_positive_control(...)
make_negative_control(...)
make_confound_control(...)
make_edge_case(...)
validate_analysis(...)
```

## Required vuffel categories

### 1. Positive-control vuffel

Contains a known planted structure.

The method should recover it.

Examples:

- a known discriminative axis,
- a known low-rank subspace,
- a known correlation,
- a known decoding signal,
- known latent dimensionality.

Questions:

- Does the method recover the planted structure?
- How accurately?
- Under what signal-to-noise ratio?
- Under what sample size?

---

### 2. Negative-control vuffel

Contains no true target structure.

The method should not invent one.

Examples:

- labels independent of neural activity,
- random stimulus assignments,
- isotropic noise,
- unrelated feature matrices.

Questions:

- Is performance near chance?
- Are false positives controlled?
- Do p-values behave appropriately?
- Does the method remain conservative?

---

### 3. Confounded / deceptive vuffel

Contains an apparent effect caused by a nuisance variable rather than the scientific target.

Examples:

```text
image complexity -> neural response
image complexity -> animate/inanimate label
```

with no true animacy effect.

Questions:

- Does naive analysis produce an apparent effect?
- Does the intended control remove it?
- Does the pipeline identify the confound?
- Can the agent distinguish target signal from nuisance structure?

---

### 4. Edge-case vuffel

Tests numerical and scientific failure modes.

Examples:

- class imbalance,
- duplicated samples,
- singular covariance,
- very low sample size,
- extreme noise,
- highly correlated predictors,
- missing values,
- constant features,
- zero variance,
- leakage,
- train/test dependence.

The correct behavior may be:

- recover the effect,
- return null,
- warn,
- fail explicitly,
- refuse to run.

Silent success is not always success.

---

# Shadow validation

Whenever the agent implements or substantially modifies an analysis function, it should perform shadow work:

```text
real-data analysis
        │
        ├── positive vuffel
        ├── negative vuffel
        ├── confounded vuffel
        └── edge-case vuffel
```

The real-data result should not be treated as trustworthy until the relevant scientific tests pass.

The purpose is not merely software correctness.

The purpose is to test the **scientific meaning** of the method.

---

# Pipeline structure

The thesis should be represented as a branching analysis graph.

Example:

```text
raw Allen data
    ↓
inspection
    ↓
canonical dataset
    ↓
preprocessing
    ↓
response matrices
    ├── decoding
    │   ├── logistic
    │   ├── Fisher
    │   └── region-wise decoding
    │
    ├── dimensionality
    │   ├── PCA
    │   ├── eigenspectrum
    │   └── effective rank
    │
    ├── stimulus controls
    │   ├── pixel features
    │   ├── JPEG residualization
    │   └── ViT features
    │
    └── geometry
        ├── semantic axes
        ├── distances
        └── subspace analyses
```

Each branch should be reproducible independently.

The agent should extend the graph one validated node at a time.

---

# Data stages

Prefer explicit data stages:

```text
raw
→ inspected
→ canonical
→ transformed
→ model input
→ analysis result
→ validation
→ figure/table
```

Never overwrite raw data.

Derived datasets should retain links to their parent inputs and transformation parameters.

---

# Analysis outputs

Keep analysis outputs simple.

The persistent output should primarily be plots.

The plot directory should mirror the analysis-thread structure: **every analysis code file gets its own informatively named plot folder**.

Example:

```text
analysis_code/
  animacy_decoding.py
  pca_eigenspectrum.py
  jpeg_residualization.py
  regional_fisher_geometry.py

plots/
  animacy_decoding/
    cross_validated_accuracy.png
    confusion_matrix.png
    accuracy_by_region.png

  pca_eigenspectrum/
    eigenspectrum.png
    cumulative_variance.png
    projection_by_animacy.png

  jpeg_residualization/
    decoding_before_after_residualization.png

  regional_fisher_geometry/
    fisher_projection_by_region.png
```

The code-file name and plot-folder name should correspond.

For example:

```text
analysis_code/pca_eigenspectrum.py
        ↓
plots/pca_eigenspectrum/
```

This gives a simple index from scientific code to figures without maintaining a separate provenance system.

Plot filenames should also be informative. A researcher should be able to infer what a figure represents from its path.

Do not create a new folder for every execution or run.

Reproducibility should come from rerunning the corresponding analysis thread from code.

If an expensive intermediate array or fitted model genuinely needs to be cached, it may be saved deliberately, but this is optional and should not turn into a logging system.

The plot directory should remain a readable map of the scientific analysis graph.

---


# Plotting

Plots are scientific artifacts, not decorative notebook output.

Plotting functions should accept explicit analysis results.

Prefer:

```python
fig = plot_fisher_projection(result, labels)
```

over plot code embedded inside analysis logic.

Plots should be saved systematically and linked to the analysis run that produced them.

Each figure should make clear:

- what data are shown,
- what split or subset was used,
- what metric is shown,
- units,
- labels,
- important parameters,
- whether the result is training, validation, or held-out.

---

# Cross-validation and leakage

Cross-validation is part of the scientific method, not a final add-on.

For predictive analyses, explicitly define:

- training unit,
- test unit,
- grouping variables,
- whether stimuli repeat,
- whether preprocessing is fit only on training data,
- whether feature selection is performed inside the fold,
- whether residualization is fit inside the fold,
- whether temporal or session dependence requires grouped splitting.

The agent must actively look for leakage.

If the independence structure is unclear, stop and ask for clarification rather than inventing a split.

---

# Nulls and controls

The agent should prefer analyses that include meaningful controls.

Possible controls include:

- label permutation,
- trial shuffle,
- stimulus shuffle,
- circular temporal shift,
- random subspaces,
- random feature spaces,
- bootstrap intervals,
- alternative nuisance models,
- negative controls,
- synthetic falsification tests.

The null must preserve whatever structure should remain preserved.

Do not shuffle blindly.

---

# Scientific assumptions

Every analysis should expose its assumptions.

Examples:

- independence assumptions,
- repeated-measures structure,
- stationarity,
- linearity,
- covariance invertibility,
- sufficient sample size,
- correct alignment,
- absence of train/test leakage,
- temporal exchangeability,
- interpretation limits.

The agent should never hide assumptions behind implementation details.

---

# Interpretation discipline

Distinguish clearly between:

- prediction,
- decodability,
- association,
- representation,
- geometry,
- mechanism,
- causality.

A successful decoder does not by itself establish mechanism or causal representation.

Interpretation should be proportional to the evidence.

If an analysis only supports a predictive statement, use predictive language.

---

# Agent behavior when adding a new analysis

When asked to add an analysis, follow this sequence:

```text
1. Inspect relevant data structures.
2. Inspect existing repository functions.
3. Identify whether an existing operation can be reused.
4. Define the scientific question.
5. Define the observational unit.
6. Define input/output contract.
7. Define assumptions and failure modes.
8. Implement the smallest reusable function.
9. Create vuffel generators.
10. Run scientific validation.
11. Add unit tests where appropriate.
12. Run on real data.
13. Generate plots/tables.
14. Save artifacts and provenance.
15. Summarize what passed, failed, and remains uncertain.
```

Do not skip directly from prompt to real-data conclusion.

---

# Agent behavior when modifying an existing analysis

Before modifying a function:

1. inspect its current contract,
2. inspect its tests,
3. inspect prior artifacts if relevant,
4. determine what scientific behavior must remain unchanged.

After modification:

1. rerun unit tests,
2. rerun vuffel validation,
3. compare against previous outputs,
4. document intentional differences.

---

# Human checkpoints

The agent should request human judgment when:

- the scientific question is ambiguous,
- multiple valid estimands exist,
- the correct observational unit is unclear,
- a preprocessing choice could materially affect interpretation,
- a null model has multiple plausible definitions,
- a result invites causal language,
- a surprising result may be artifact or discovery,
- a new analysis branch substantially changes the thesis story.

Automation should reduce mechanical work, not silently make scientific commitments.

---

# Preferred development style

Prefer:

```text
small function
→ test
→ vuffel
→ real data
→ artifact
→ next function
```

over:

```text
large prompt
→ giant analysis script
→ many plots
→ retroactive debugging
```

The repository should become more reliable with every added function.

---

# Definition of done for an analysis node

An analysis node is complete when:

- [ ] reusable function exists,
- [ ] inputs and outputs are documented,
- [ ] assumptions are documented,
- [ ] random behavior is controlled,
- [ ] positive-control vuffel passes,
- [ ] negative-control vuffel passes,
- [ ] relevant confound control passes,
- [ ] edge cases are handled,
- [ ] real-data run succeeds,
- [ ] outputs are saved,
- [ ] plots are reproducible,
- [ ] interpretation is appropriately qualified.

---

# Overall research principle

Do not build an AI that merely analyzes the thesis.

Build a neuroscience analysis library whose functions are:

- composable,
- reproducible,
- inspectable,
- self-testing,
- scientifically falsifiable,
- easy for an agent to orchestrate.

The thesis is the first complete case study of this architecture.

The agent's role is:

```text
inspect
→ propose
→ implement
→ generate vuffel
→ validate
→ run on real data
→ produce artifacts
→ inspect results
→ choose the next branch
```

The long-term goal is a scientific workflow in which invalid or weak analyses are difficult to perform silently, and every figure and claim can be reproduced from the data, code, assumptions, controls, and computation.
