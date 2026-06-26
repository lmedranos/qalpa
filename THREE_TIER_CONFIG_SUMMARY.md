# Three-Tier Configuration Architecture Summary

## What Changed

✅ **Before**: Single monolithic `experiment_config.yaml`  
✅ **After**: Hierarchical three-tier configuration system

---

## Architecture

```
TIER 1: ROOT (/configs/)
├── system.yaml          (global system settings)
├── training.yaml        (default training parameters)
└── paths.yaml           (root directory paths)
        ↓
TIER 2: FRAMEWORK (EQMgen/configs/)
├── eqmgen_paths.yaml       (code directory paths)
├── eqmgen_pipeline.yaml    (pipeline structure)
├── eqmgen_model.yaml       (model configuration)
├── eqmgen_geometry.yaml    (geometry optimizer)
└── eqmgen_properties.yaml  (property calculator)
        ↓
TIER 3: EXPERIMENT (Experiments/{exp_name}/configs/)
├── exp_data.yaml        (dataset & properties)
├── exp_training.yaml    (training overrides)
├── exp_geometry.yaml    (geometry overrides)
└── exp_properties.yaml  (properties overrides)
```

---

## Directory Layout

```
activediffusion/
│
├── configs/                          ← TIER 1: GLOBAL
│   ├── system.yaml                  (seed, device)
│   ├── training.yaml                (batch size, iterations)
│   ├── paths.yaml                   (root paths)
│   └── config.yaml                  (root defaults)
│
├── EQMgen/
│   └── configs/                     ← TIER 2: FRAMEWORK
│       ├── eqmgen_paths.yaml
│       ├── eqmgen_pipeline.yaml
│       ├── eqmgen_model.yaml
│       ├── eqmgen_geometry.yaml
│       ├── eqmgen_properties.yaml
│       └── eqmgen_config.yaml
│
├── Experiments/
│   └── allostery/
│       └── configs/
│           └── experiment_config.yaml  ← TIER 3: EXPERIMENT (single file)
│
├── HIERARCHICAL_CONFIG_GUIDE.md     (detailed guide)
├── RUN_EXPERIMENT.sh                (usage examples)
└── REFACTORING_GUIDE.md             (package refactoring)
```

---

## Configuration Resolution

When you run an experiment, Hydra resolves configs in priority order:

```
experiment_config.yaml (Experiments/allostery/configs/)
    ↓ imports ↓
  /configs/system.yaml (root)
  /configs/training.yaml (root)
  /configs/paths.yaml (root)
  /EQMgen/configs/eqmgen_config.yaml (framework)
    ↓ imports ↓
    eqmgen_paths.yaml
    eqmgen_pipeline.yaml
    eqmgen_model.yaml
    eqmgen_geometry.yaml
    eqmgen_properties.yaml
  /configs/system.yaml, /configs/training.yaml, /configs/paths.yaml
  exp_data.yaml (experiment)
  exp_training.yaml (experiment)
  exp_geometry.yaml (experiment)
  exp_properties.yaml (experiment)
```

**Last config wins** - experiment configs override framework configs, which override root configs.

---

## How to Use

### 1. Run Experiment (Full Workflow)

```bash
# From experiment directory
cd Experiments/allostery
python ../../EQMgen/scripts/run_experiment.py

# Or specify experiment from project root
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/allostery/configs \
    --config-name experiment_config
```

### 2. Override at Runtime

```bash
# Change any parameter from CLI
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/allostery/configs \
    --config-name experiment_config \
    training.num_iterations=5 \
    geometry.optimizer=EquiDTB \
    system.device=cpu
```

### 3. Create New Experiment (Using `init_experiment()`)

```bash
# CLI method
python EQMgen/scripts/init_experiment.py my_experiment

# Python method
python -c "from EQMgen.scripts import init_experiment; init_experiment('my_experiment')"

# Or from Python script
from EQMgen.scripts import init_experiment
init_experiment("my_experiment")
```

Creates:
- ✅ `Experiments/my_experiment/configs/experiment_config.yaml` (auto-populated)
- ✅ `Experiments/my_experiment/data/`
- ✅ `Experiments/my_experiment/results/`
- ✅ `Experiments/my_experiment/logs/`

Then run:
```bash
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/my_experiment/configs \
    --config-name experiment_config
```

---

## What Goes Where?

### Root Tier (`/configs/`)
**Shared across ALL experiments**
- ✅ System settings (seed, device, determinism)
- ✅ Default training parameters
- ✅ Base directory paths

### Framework Tier (`EQMgen/configs/`)
**Specific to EQMgen implementation**
- ✅ Code directory paths
- ✅ Pipeline structure
- ✅ Model architecture
- ✅ Geometry optimizer configuration
- ✅ Property calculator configuration

### Experiment Tier (`Experiments/{exp_name}/configs/`)
**Single `experiment_config.yaml` file with all experiment-specific settings**
- ✅ Experiment metadata (name, description, version)
- ✅ Dataset paths and properties
- ✅ Training overrides for this experiment
- ✅ Geometry overrides for this experiment
- ✅ Property calculator overrides
- All organized in clear sections within one file

---

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| **Config Reuse** | ❌ Single file | ✅ Share root + framework configs |
| **Maintainability** | ❌ Large file | ✅ Focused small files |
| **New Experiments** | ❌ Copy large file | ✅ Copy only experiment overrides |
| **Parameter Override** | ❌ Manual edit | ✅ CLI `key=value` |
| **Config Clarity** | ❌ Mixed concerns | ✅ Clear separation of concerns |
| **Multiple Experiments** | ❌ Difficult | ✅ Easy template reuse |

---

## Key Files

| File | Purpose |
|------|---------|
| `HIERARCHICAL_CONFIG_GUIDE.md` | Detailed configuration guide |
| `RUN_EXPERIMENT.sh` | Quick reference and examples |
| `EQMgen/scripts/run_experiment.py` | Main experiment runner |
| `EQMgen/scripts/forward.py` | Forward pass (generation) |
| `EQMgen/scripts/backward.py` | Backward pass (training) |

---

## Quick Commands

```bash
# Run with defaults
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/allostery/configs \
    --config-name experiment_config

# Run with overrides
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/allostery/configs \
    --config-name experiment_config \
    training.num_iterations=5

# View resolved configuration
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/allostery/configs \
    --config-name experiment_config \
    --cfg job

# Create new experiment
mkdir -p Experiments/new_exp/configs
cp Experiments/allostery/configs/exp_*.yaml Experiments/new_exp/configs/
```

---

## Next Steps

1. ✅ **Review** the three-tier structure
2. ✅ **Understand** where each config belongs
3. ✅ **Run** experiments using the new system
4. ✅ **Create** new experiments using templates
5. ✅ **Customize** overrides per experiment

See **HIERARCHICAL_CONFIG_GUIDE.md** for detailed documentation.
