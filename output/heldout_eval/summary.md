| Variant | Task | Successes / n | Rate | Wilson 95% CI | Mean steps (succ.) | Median latency ms | Visual tokens | Params | MFLOPs/fwd |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| vision | open_the_middle_drawer_of_the_cabinet | 0/20 | 0% | 0.00–0.16 |  | 1.08 | 32 | 64231 | 10.54 |
| vision | put_the_bowl_on_the_stove | 9/20 | 45% | 0.26–0.66 | 88.6 | 0.9 | 32 | 64231 | 10.54 |
| vision | both | 9/40 | 22% | 0.12–0.38 | 88.6 | 0.99 | 32 | 64231 | 10.54 |
| dense | open_the_middle_drawer_of_the_cabinet | 0/20 | 0% | 0.00–0.16 |  | 1.02 | 32 | 64759 | 10.54 |
| dense | put_the_bowl_on_the_stove | 17/20 | 85% | 0.64–0.95 | 92.2 | 0.9 | 32 | 64759 | 10.54 |
| dense | both | 17/40 | 42% | 0.28–0.58 | 92.2 | 1.02 | 32 | 64759 | 10.54 |
| grid | open_the_middle_drawer_of_the_cabinet | 1/20 | 5% | 0.01–0.24 | 124.0 | 1.01 | 2 | 71915 | 10.25 |
| grid | put_the_bowl_on_the_stove | 3/20 | 15% | 0.05–0.36 | 102.7 | 0.69 | 2 | 71915 | 10.25 |
| grid | both | 4/40 | 10% | 0.04–0.23 | 108.0 | 0.98 | 2 | 71915 | 10.25 |

Paired exact McNemar tests (same held-out starts, both tasks pooled):

| A | B | paired starts | A-only successes | B-only successes | p |
|---|---|---:|---:|---:|---:|
| vision | dense | 40 | 0 | 8 | 0.008 |
| vision | grid | 40 | 9 | 4 | 0.267 |
| dense | grid | 40 | 15 | 2 | 0.002 |
