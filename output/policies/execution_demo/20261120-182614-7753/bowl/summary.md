| Variant | Task | Successes / n | Rate | Wilson 95% CI | Mean steps (succ.) | Median latency ms | Visual tokens | Params | MFLOPs/fwd |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| vision | put_the_bowl_on_the_stove | 2/3 | 67% | 0.21–0.94 | 102.0 | 1.97 | 32 | 64231 | 10.54 |
| vision | both | 2/3 | 67% | 0.21–0.94 | 102.0 | 1.97 | 32 | 64231 | 10.54 |
| dense | put_the_bowl_on_the_stove | 3/3 | 100% | 0.44–1.00 | 91.3 | 2.62 | 32 | 64759 | 10.54 |
| dense | both | 3/3 | 100% | 0.44–1.00 | 91.3 | 2.62 | 32 | 64759 | 10.54 |
| grid | put_the_bowl_on_the_stove | 0/2 | 0% | 0.00–0.66 |  | 1.87 | 2 | 71915 | 10.25 |
| grid | both | 0/2 | 0% | 0.00–0.66 |  | 1.87 | 2 | 71915 | 10.25 |

Paired exact McNemar tests (same held-out starts, both tasks pooled):

| A | B | paired starts | A-only successes | B-only successes | p |
|---|---|---:|---:|---:|---:|
| vision | dense | 3 | 0 | 1 | 1.000 |
| vision | grid | 2 | 1 | 0 | 1.000 |
| dense | grid | 2 | 2 | 0 | 0.500 |
