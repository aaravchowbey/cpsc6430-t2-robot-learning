| Variant | Task | Successes / n | Rate | Wilson 95% CI | Mean steps (succ.) | Median latency ms | Visual tokens | Params | MFLOPs/fwd |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| dense | open_the_middle_drawer_of_the_cabinet | 0/20 | 0% | 0.00–0.16 |  | 1.02 | 32 | 64759 | 10.54 |
| dense | put_the_bowl_on_the_stove | 17/20 | 85% | 0.64–0.95 | 92.2 | 0.9 | 32 | 64759 | 10.54 |
| dense | both | 17/40 | 42% | 0.28–0.58 | 92.2 | 1.02 | 32 | 64759 | 10.54 |

Paired exact McNemar tests (same held-out starts, both tasks pooled):

| A | B | paired starts | A-only successes | B-only successes | p |
|---|---|---:|---:|---:|---:|
