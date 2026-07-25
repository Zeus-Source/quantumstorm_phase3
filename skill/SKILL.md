# QuantumStorm qBraid Skill

Agent-executable interface for the QuantumStorm QRC pipeline. Every entry point
is both importable (`from skill.skill import reproduce_benchmark`) and runnable
from the CLI (`python skill/skill.py <entry> [args]`). Results are written to
`results/*.json`.

> Packaging note: verify the exact metadata/manifest qBraid currently expects for
> a "Skill" at https://docs.qbraid.com and wrap these functions accordingly. The
> functions themselves are the stable, backend-independent core.

## Entry points

### `configure_reservoir`
Select the reservoir coupling J/h via the edge-of-many-body-chaos level-spacing
criterion and persist it to `results/reservoir_config.json`.
```
python skill/skill.py configure_reservoir --n-diag 12
```

### `run_sweep`
Scaling sweep across qubit counts; reports AUPRC mean ± 95% CI per size.
```
python skill/skill.py run_sweep --qubits 4 8 12 16 --seeds 3
```

### `evaluate_tail_metrics`
Compute AUROC / AUPRC / EDS / BSS@99 for the QRC and every classical baseline on
the frozen test split; writes `results/tail_metrics.json`.
```
python skill/skill.py evaluate_tail_metrics
```

### `reproduce_benchmark`
Full headline result: multi-seed QRC (mean ± CI) vs baselines, plus a paired
bootstrap significance test against the ESN. Writes `results/benchmark.json`.
```
python skill/skill.py reproduce_benchmark --seeds 5
```

## Common arguments
`--jh` (override coupling), `--backend {auto,numpy,cupy}`, `--seeds`, `--qubits`.
If `--jh` is omitted, the value from `configure_reservoir` is reused.

## Typical agent recipe
```
configure_reservoir --n-diag 12
reproduce_benchmark --seeds 5
run_sweep --qubits 4 8 12 16 --seeds 3
```
