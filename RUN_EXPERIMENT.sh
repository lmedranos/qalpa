#!/bin/bash
# Quick reference for running experiments

# ============================================================================
# 0. CREATE NEW EXPERIMENT (INIT)
# ============================================================================

# Quick way to scaffold a new experiment
python EQMgen/scripts/init_experiment.py my_experiment

# Creates directory structure and config template:
# - Experiments/my_experiment/configs/experiment_config.yaml
# - Experiments/my_experiment/data/
# - Experiments/my_experiment/results/
# - Experiments/my_experiment/logs/

# Then edit the config and run the experiment as shown below


# ============================================================================
# 1. RUN EXPERIMENT WITH DEFAULT CONFIG
# ============================================================================

# From experiment directory (Experiments/allostery/)
cd Experiments/allostery/configs
python ../../EQMgen/scripts/run_experiment.py

# Or from project root
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/allostery/configs \
    --config-name experiment_config


# ============================================================================
# 2. OVERRIDE CONFIGURATION AT RUNTIME
# ============================================================================

# Override experiment parameters
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/allostery/configs \
    --config-name experiment_config \
    training.num_iterations=5 \
    training.batch_size=64

# Override framework parameters
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/allostery/configs \
    --config-name experiment_config \
    geometry.optimizer=EquiDTB \
    properties.calculator=FHI-aims

# Override global parameters
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/allostery/configs \
    --config-name experiment_config \
    system.device=cpu \
    system.seed=123


# ============================================================================
# 3. RUN ONLY FORWARD PASS (GENERATION)
# ============================================================================

python EQMgen/scripts/forward.py \
    --base_dataset /path/to/data.hdf5 \
    --dataset_name qm7x \
    --data Experiments/allostery/data \
    --model_code EQMgen/eqdiff \
    --config configs \
    --actdiff_utils EQMgen/utils \
    --iter 0 \
    --props HLgap eMBD \
    --scores SAScore


# ============================================================================
# 4. RUN ONLY BACKWARD PASS (TRAINING/EVALUATION)
# ============================================================================

python EQMgen/scripts/backward.py \
    --config-path Experiments/allostery/configs \
    --config-name experiment_config


# ============================================================================
# 5. CREATE NEW EXPERIMENT
# ============================================================================

# Create experiment structure
EXPERIMENT_NAME="my_experiment"
mkdir -p Experiments/$EXPERIMENT_NAME/configs
mkdir -p Experiments/$EXPERIMENT_NAME/data

# Copy template files
cp Experiments/allostery/configs/exp_*.yaml Experiments/$EXPERIMENT_NAME/configs/

# Edit experiment-specific configs
# - exp_data.yaml (dataset paths, properties)
# - exp_training.yaml (training overrides)
# - exp_geometry.yaml (geometry overrides)
# - experiment_config.yaml (main config)

# Run new experiment
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/$EXPERIMENT_NAME/configs \
    --config-name experiment_config


# ============================================================================
# 6. VIEW FULL CONFIGURATION
# ============================================================================

# Print resolved configuration (no execution)
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/allostery/configs \
    --config-name experiment_config \
    --cfg job


# ============================================================================
# 7. SAVE CONFIGURATION OUTPUT
# ============================================================================

# Save configuration with results
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/allostery/configs \
    --config-name experiment_config \
    hydra.run.dir=outputs/run_${now:%Y-%m-%d}/${now:%H-%M-%S}
