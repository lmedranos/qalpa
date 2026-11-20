# KDE target planning

The `plan` block decides *where* the model is asked to generate. With the template settings (`query_strategy: kde_map`, `kde_region: anchor`, `anchor_score: relative_pct`), it builds each iteration's targets in five steps.

```{mermaid}
flowchart LR
    A(["<b>1 · Anchor</b><br/>waypoint n<br/>= target 0"]):::al
    C["<b>2 · Sample</b><br/>anchor_n_candidates<br/>within ± anchor_box"]:::step
    F["<b>3 · Filter</b><br/>ΔatNUM ≥ anchor_min_datnum<br/>density inside the<br/>percentile band"]:::step
    SC["<b>4 · Score</b><br/>distance to the anchor<br/>+ density difference"]:::step
    SEL["<b>5 · Select</b><br/>best first, then<br/>farthest-point"]:::step
    TJ[/"<b>targets.json</b><br/>anchor + anchor_n_targets"/]:::data
    K[("Gaussian KDE of<br/>the initial dataset")]:::data

    A --> C --> F --> SC --> SEL --> TJ
    K -.-> F
    K -.-> SC

    classDef step fill:#FFFFFF,stroke:#78909C,color:#212121
    classDef al fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px,color:#212121
    classDef data fill:#F3E5F5,stroke:#8E24AA,color:#212121
    linkStyle default stroke:#607D8B,stroke-width:1.5px
```

1. **Anchor.** Target 0 is waypoint *n* of the trajectory and receives `EQdiffParam.num_samples` molecules. When there are more iterations than waypoints, the last waypoint is reused.
2. **Sample.** `anchor_n_candidates` candidate targets are drawn uniformly within ± `anchor_box` (a fraction of each anchor value) on every planning axis: the conditioning properties plus `atNUM`.
3. **Filter.** A candidate is eligible if its atom count differs from the anchor's by at least `anchor_min_datnum` and its KDE density lies inside the `density_percentiles` band of the initial dataset. The template band, the 5th to 15th percentile, selects regions that are sparse but not empty. If too few candidates qualify, the planner adds those closest to the band.
4. **Score.** Each eligible candidate **y** is scored by how far it lies from the anchor **a**, both in property values and in density:

   ```{math}
   s(\mathbf{y}) \;=\; \sum_{k} 100\,\frac{\lvert y_k - a_k\rvert}{\lvert a_k\rvert} \;+\; 100\,\frac{\lvert \hat p(\mathbf{y}) - \hat p(\mathbf{a})\rvert}{\hat p(\mathbf{a})}
   ```

   Here *k* runs over the conditioning properties and *p̂* is the KDE of the initial dataset.
5. **Select.** The highest-scoring candidate is taken first. Every further target is the eligible candidate farthest from all targets chosen so far, anchor included (farthest-point sampling in relative-% space). The planner selects `anchor_n_targets` targets, and each receives `anchor_extra_fraction × num_samples` molecules.

With the template values, each iteration requests 200 molecules at the waypoint and 40 at each of two KDE targets, 280 in total.

## The density estimate

The KDE is a Gaussian kernel density estimate of the initial dataset in standardized property space (Scott bandwidth, at least one atom wide along `atNUM`). Its leave-one-out densities define the percentile band; they are computed once and cached in `Experiments/<experiment>/density_reference.npy`. Because the reference is the initial dataset, KDE planning starts in iteration 0, and it is deterministic for a given `system.seed`. Every candidate, with its score components and selection order, is listed in `iter_<n>/anchor_candidates.csv`.

(hit-rule)=
## Closing the loop: the hit rule

After the QM step, each molecule is judged against its own target **t**. It is a *hit* if every conditioning property lies within tolerance; atom count is not part of the rule:

```{math}
\lvert y_k - t_k\rvert \;\le\; \max\!\left(\varepsilon^{\mathrm{rel}}_k\,\lvert t_k\rvert,\; \varepsilon^{\mathrm{abs}}_k\right) \qquad \text{for every conditioning property } k
```

The relative tolerance defaults to `query.filter_threshold` (0.1) and the absolute tolerance to 0; both can be set per property in `al.tolerances` ([Hit rule and labelling](../configuration/active-learning.md#hit-rule-and-labelling)). `prop_filter` removes the misses but keeps their statistics in `prop_filter_report.json`. Without `prop_filter`, misses are stored with `al_hit = 0`.

(planning-strategies)=
## Planning strategies

`al.query_strategy` and `al.kde_region` select how targets are planned.

| Strategy | Settings | Targets per iteration | Typical use |
|---|---|---|---|
| **KDE anchor** (template) | `kde_map` + `kde_region: anchor` | Waypoint *n* plus `anchor_n_targets` KDE targets around it | Follow the trajectory and fill sparse neighbourhoods |
| **KDE corridor** | `kde_map` + `kde_region: corridor` | `targets_per_iter` targets inside a padded corridor around the whole trajectory | Explore the full path adaptively |
| **Frontier** | `frontier` | One waypoint, advanced once its hit rate reaches `advance_hit_rate` | Strictly sequential navigation |
| **Trajectory** | `trajectory` | Waypoint *n* | Baseline with active-learning bookkeeping |
| **No active learning** | `al.enabled: false` | Waypoint *n*; `plan` and `dedup` are skipped | The original, non-adaptive pipeline |

### Adaptive score

The corridor mode, and the anchor mode with `anchor_score: kde`, rank candidates by an adaptive score built from the data and the generator's track record:

```{math}
s(\mathbf{y}) \;=\; w_{\mathrm{sparse}}\,\mathrm{norm}\!\left[-\log \hat p_{\mathrm{lab}}(\mathbf{y})\right] \;+\; w_{\mathrm{err}}\,\mathrm{norm}\!\left[\bar e(\mathbf{y})\right] \;+\; \frac{w_{\mathrm{explore}}}{\sqrt{n_{\mathrm{eff}}(\mathbf{y}) + 1}}
```

Here *p̂*<sub>lab</sub> is a KDE of all labelled molecules, *ē* is the generator's kernel-smoothed past error near **y**, *n*<sub>eff</sub> counts past requests near **y**, and norm rescales to [0, 1]. Regions the generator has repeatedly failed to reach (`density.min_reach`, `min_neff`) are skipped, and a Gaussian repulsion keeps the targets apart. The corridor mode needs this track record, so it requests the plain waypoint until the molecules of a first iteration have been labelled.

All planner keys, with their defaults, are listed in [Active learning](../configuration/active-learning.md). Ready-made snippets for wider or corridor planning are in [Recipes](../configuration/recipes.md).
