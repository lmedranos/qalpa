# Monitoring training with MLflow

Training logs to TensorBoard. With `mlflow.enabled: true`, the forward job mirrors the logs of all iterations into an MLflow store after every training run. You can also sync by hand at any time, even while training runs:

```bash
python src/scripts/sync_mlflow.py \
  --model-root Data/my_exp_0/diffusion_wdir/model_registry \
  --tracking-uri file://$PWD/Data/mlruns --experiment my_exp_0
mlflow ui --backend-store-uri file://$PWD/Data/mlruns --port 5000
```

On a cluster, forward the port with `ssh -L 5000:localhost:5000 <login-node>`, or let VS Code forward it.

```{tip}
For a quick look without MLflow, the Navigator's [Training](../navigator/running-jobs.md#training) tab plots the same TensorBoard curves, one line per iteration.
```

## What is logged

| Run | Contents |
|---|---|
| `iter_NN` | All training, validation and test curves of one iteration, per-dataset losses (`val_dataset_<k>/…`, `test_dataset_<k>/…`), sampling metrics (`gen/<metric>`: validity, QED, SA, …), hyperparameters and sampling shares |
| `overview` | One point per iteration: final metrics (`final/…`), sampling shares (`share/dataset_<k>`), draws per sample (`draws_per_sample_per_epoch/dataset_<k>`) and forgetting (`forgetting/dataset_<k>`), plus a test-loss matrix of every model on every test set |

## Reading the curves

- A positive `forgetting/dataset_0`, or a rising `final/val_dataset_0/loss_epoch`, means the model is forgetting the initial data. Raise `min_original_share`, `alpha` or `beta`.
- A large `draws_per_sample_per_epoch` for the newest dataset means a risk of overfitting to it. Lower `samples_per_epoch` or raise `alpha`.
- The diffusion test loss is computed at random noise levels, so small changes between iterations are noise.

## The MLflow interpreter

`mlflow.python` must point to an interpreter with `mlflow`, `tensorboard` and `matplotlib`; replace the template's site-specific path. `mlflow` is part of all three environment files. If your environment was created without it, use a small side environment so the training environment stays unchanged (`matplotlib` comes with `mlflow`):

```bash
conda_envs/env-qalpa-forward/bin/python -m venv conda_envs/env-qalpa-mlflow
conda_envs/env-qalpa-mlflow/bin/python -m pip install "mlflow<3" tensorboard
```

Then set `mlflow.python` to `<repo>/conda_envs/env-qalpa-mlflow/bin/python`. All MLflow keys are listed under [MLflow settings](../configuration/experiment.md#mlflow).
