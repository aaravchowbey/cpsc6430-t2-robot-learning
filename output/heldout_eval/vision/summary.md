| Variant | Task | Successes / n | Rate | Wilson 95% CI | Mean steps (succ.) | Median latency ms | Visual tokens | Params | MFLOPs/fwd |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| vision | open_the_middle_drawer_of_the_cabinet | 0/20 | 0% | 0.00–0.16 |  | 1.08 | 32 | 64231 | 10.54 |
| vision | put_the_bowl_on_the_stove | 9/20 | 45% | 0.26–0.66 | 88.6 | 0.9 | 32 | 64231 | 10.54 |
| vision | both | 9/40 | 22% | 0.12–0.38 | 88.6 | 0.99 | 32 | 64231 | 10.54 |

Paired exact McNemar tests (same held-out starts, both tasks pooled):

| A | B | paired starts | A-only successes | B-only successes | p |
|---|---|---:|---:|---:|---:|
