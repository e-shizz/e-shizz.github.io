# gnome.science research notes

Our notebook for learning to **map and steer neurons** in language models, starting from the work of
Sauers ([@Sauers_](https://x.com/Sauers_) · [gnome.science](https://gnome.science/) · [github.com/SauersML](https://github.com/SauersML)).

## Notes

| # | File | What's in it |
|---|---|---|
| 01 | [01-sauers-overview.md](01-sauers-overview.md) | Who Sauers is, how they work, and a map of their neuron, steering and introspection projects |
| 02 | [02-honesty-heads-deep-dive.md](02-honesty-heads-deep-dive.md) | Line-by-line walkthrough of `honesty-heads`: the method, the code, the results, the caveats, and what to learn from it |
| 03 | [03-related-reading.md](03-related-reading.md) | Background papers and other links from our research |

## Learning path

1. **Concepts:** the residual stream; heads and neurons "read from" and "write to" it (Elhage et al., 2021, *A Mathematical Framework for Transformer Circuits*).
2. **Reading:** linear probes, mean-difference directions, SAEs. → `vlens`, `Manifold-SAE`
3. **Writing:** activation steering, choosing which units to steer, random-direction baselines. → **`honesty-heads`** (note 02)
4. **Models steering themselves:** → `magician`, `zigzag`
5. **When reading isn't neutral:** → `watching`
6. **Hands-on:** a small replication of honesty-heads on a model that fits on one consumer GPU (sketch at the end of note 02).

## Log

- 2026-11-03: Started. Mapped Sauers' repos and wrote a deep dive on `honesty-heads`.
