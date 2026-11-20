# Diffusion model

Source: {repo}`src/configs/eqdiff_model.yaml`

The forward job trains with `run_train.py --conf src/configs/eqdiff_model.yaml <flags>`. The file is applied first and the flags second, so keys marked *Overridden* take their value from `EQdiffParam`. The flags also set the save directory, dataset root, GPU count, number of epochs, test interval, checkpoint, property conditioning and replay options, so adding those keys here has no effect.

Every key must be a known argument of the training script ({repo}`hparams.py <src/eqdiff/e3mol_aqm/experiments/hparams.py>`); an unknown key stops training with `Unknown argument in config file`. Changes apply to every experiment trained afterwards, including running experiments: unlike the other shared files, this one is read directly by every training run ({ref}`write-back <write-back>`).

Edit labels are explained in [Reading the reference tables](index.md#reading-the-reference-tables).

## Model

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `sdim` | `256` | Scalar feature dimension of the equivariant (EQGAT) layers | Optional |
| `vdim` | `256` | Vector feature dimension | Optional |
| `edim` | `128` | Edge feature dimension | Optional |
| `num_layers` | `12` | Number of message-passing layers | Optional |
| `use_cross_product` | `false` | Cross-product interactions in the vector updates | Optional |
| `edge_mp` | `false` | Edge message passing | Optional |
| `fully_connected` | `true` | Fully connected molecular graphs | Optional |
| `local_global_model` | `false` | Combined local and global message passing | Optional |
| `vector_aggr` | `mean` | Aggregation of vector features | Optional |
| `use_pos_norm` | `true` | Normalization of coordinates inside the network | Optional |
| `bond_prediction` | `false` | Auxiliary bond prediction | Optional |
| `bond_model_guidance` | `false` | Guide sampling with a separate bond model (needs a bond-model checkpoint) | Optional |
| `additional_feats` | `false` | Extra atom features (aromaticity, ring membership, hybridization); `true` selects a different model class | Keep |

## Diffusion

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `continuous` | `false` | Continuous-time diffusion; `false` is the discrete diffusion model QALPA uses | Keep |
| `noise_scheduler` | `adaptive` | Noise schedule | Optional |
| `timesteps` | `500` | Number of diffusion steps | Optional |
| `eps_min` | `1.e-3` | Minimum epsilon of the schedule | Optional |
| `beta_min`, `beta_max` | `1.e-4`, `2.e-2` | Beta range of the schedule | Optional |
| `loss_weighting` | `snr_t` | Loss weighting over noise levels | Optional |

## Training and optimizer

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `seed` | `42` | Training seed | Optional |
| `batch_size` | `200` | Training batch size; replaced by `EQdiffParam.batch_size_tr` | Overridden |
| `inference_batch_size` | `70` | Batch size of validation and test loading and of the sampling evaluations during training | Optional |
| `lr` | `2.e-4` | Initial learning rate | Optional |
| `lr_scheduler` | `reduce_on_plateau` | `reduce_on_plateau`, `cosine_annealing` or `cyclic` | Optional |
| `lr_factor`, `lr_patience`, `lr_cooldown` | `0.75`, `50`, `0` | Reduce-on-plateau factor, patience and cooldown | Optional |
| `lr_min`, `lr_step_size` | `5.e-5`, `5.e-5` | Lower learning rate and step size of the cyclic and cosine schedulers | Optional |
| `weight_decay` | `1.e-12` | Optimizer weight decay | Optional |
| `ema_decay` | `0.999` | Decay of the exponential moving average of the weights | Optional |
| `grad_clip_val` | `10` | Gradient clipping value | Optional |
| `accum_batch` | `1` | Gradient-accumulation steps | Optional |
| `max_num_neighbors` | `128` | Maximum number of graph neighbours | Optional |
| `gamma` | `0.99` | Not used by the discrete diffusion trainer | Informational |
| `max_time` | `06:00:00:00` | Not passed to the trainer; the training length is set by `EQdiffParam.num_epochs` | Informational |

## Data and runtime

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `dataset` | `aqm` | Dataset loader; replaced by `EQdiffParam.dataset` | Overridden |
| `concat_datasets`, `alpha`, `beta` | `false`, `0.2`, `0.2` | Replay settings; replaced by the `EQdiffParam` values when `EQdiffParam.concat_datasets` is true | Overridden |
| `remove_hs` | `false` | Train without hydrogen atoms. Keep `false`: the QM steps need explicit hydrogens | Keep |
| `use_adaptive_loader` | `true` | Size-adaptive batching; not used by the `aqm_qm7x` loader | Informational |
| `id` | `0` | Run index; checkpoints go to `model_iter_<n>/run<id>/`, and the pipeline reads `run0/` | Keep |
| `num_workers` | `8` | Data-loader workers | Optional |
| `precision` | `32` | Numerical precision of training | Optional |

:::{dropdown} Full file: `src/configs/eqdiff_model.yaml`
```{literalinclude} ../../src/configs/eqdiff_model.yaml
:language: yaml
```
:::
