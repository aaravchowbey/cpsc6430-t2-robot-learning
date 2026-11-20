| Variant | Task | Successes / n | Rate | Wilson 95% CI | Mean steps (succ.) | Median latency ms | Visual tokens | Params | MFLOPs/fwd |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| vision | open_the_middle_drawer_of_the_cabinet | 0/3 | 0% | 0.00–0.56 |  | 1.62 | 32 | 64231 | 10.54 |
| vision | both | 0/3 | 0% | 0.00–0.56 |  | 1.62 | 32 | 64231 | 10.54 |
| dense | open_the_middle_drawer_of_the_cabinet | 0/3 | 0% | 0.00–0.56 |  | 1.64 | 32 | 64759 | 10.54 |
| dense | both | 0/3 | 0% | 0.00–0.56 |  | 1.64 | 32 | 64759 | 10.54 |
| grid | open_the_middle_drawer_of_the_cabinet | 0/3 | 0% | 0.00–0.56 |  | 1.79 | 2 | 71915 | 10.25 |
| grid | both | 0/3 | 0% | 0.00–0.56 |  | 1.79 | 2 | 71915 | 10.25 |

Paired exact McNemar tests (same held-out starts, both tasks pooled):

| A | B | paired starts | A-only successes | B-only successes | p |
|---|---|---:|---:|---:|---:|
| vision | dense | 3 | 0 | 0 | 1.000 |
| vision | grid | 3 | 0 | 0 | 1.000 |
| dense | grid | 3 | 0 | 0 | 1.000 |
