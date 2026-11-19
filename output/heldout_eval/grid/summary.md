| Variant | Task | Successes / n | Rate | Wilson 95% CI | Mean steps (succ.) | Median latency ms | Visual tokens | Params | MFLOPs/fwd |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| grid | open_the_middle_drawer_of_the_cabinet | 1/20 | 5% | 0.01–0.24 | 124.0 | 1.01 | 2 | 71915 | 10.25 |
| grid | put_the_bowl_on_the_stove | 3/20 | 15% | 0.05–0.36 | 102.7 | 0.69 | 2 | 71915 | 10.25 |
| grid | both | 4/40 | 10% | 0.04–0.23 | 108.0 | 0.98 | 2 | 71915 | 10.25 |

Paired exact McNemar tests (same held-out starts, both tasks pooled):

| A | B | paired starts | A-only successes | B-only successes | p |
|---|---|---:|---:|---:|---:|
