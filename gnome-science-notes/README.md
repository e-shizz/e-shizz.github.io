# gnome.science research notes

Our notebook for learning to **map and steer neurons** in language models, starting from the work of
Sauers ([@Sauers_](https://x.com/Sauers_) · [gnome.science](https://gnome.science/) · [github.com/SauersML](https://github.com/SauersML)).

## Notes

| # | File | What's in it |
|---|---|---|
| 01 | [01-sauers-overview.md](01-sauers-overview.md) | Who Sauers is, how they work, and a map of their neuron, steering and introspection projects |
| 02 | [02-honesty-heads-deep-dive.md](02-honesty-heads-deep-dive.md) | Line-by-line walkthrough of `honesty-heads`: the method, the code, the results, the caveats, and what to learn from it |
| 03 | [03-related-reading.md](03-related-reading.md) | Background papers and other links from our research |
| 04 | [04-watching-deep-dive.md](04-watching-deep-dive.md) | `watching`: how a model "detects" being probed. It's a `torch.compile` rounding footprint read by one attention head |
| 05 | [05-vlens-deep-dive.md](05-vlens-deep-dive.md) | `vlens`: drawing a model's mid-layer states with its own image generator; what the secret-password result does and doesn't show |
| 06 | [06-magician-zigzag-deep-dive.md](06-magician-zigzag-deep-dive.md) | `zigzag` and `magician`: training a model to write its own concept representations on command; the even/odd shortcut and how written plans fixed it |
| 07 | [07-neural-pairing-deep-dive.md](07-neural-pairing-deep-dive.md) | `neural-pairing`: two model families linked through hidden states. Closer to reading the partner's mind than sending a message |
| 08 | [08-manifold-sae-and-introspection.md](08-manifold-sae-and-introspection.md) | `Manifold-SAE` (the authors' verdict is negative), `retained-reply-state`, and `claude-introspection`, including an asymmetry we found in its permutation null |

## Learning path

1. **Concepts:** the residual stream; heads and neurons "read from" and "write to" it (Elhage et al., 2021, *A Mathematical Framework for Transformer Circuits*).
2. **Reading:** linear probes, mean-difference directions, SAEs. → `vlens`, `Manifold-SAE`
3. **Writing:** activation steering, choosing which units to steer, random-direction baselines. → **`honesty-heads`** (note 02)
4. **Models steering themselves:** → `magician`, `zigzag`
5. **When reading isn't neutral:** → `watching`
6. **Hands-on:** a small replication of honesty-heads on a model that fits on one consumer GPU (sketch at the end of note 02).

## Log

- 2026-11-03: Started. Mapped Sauers' repos and wrote a deep dive on `honesty-heads`.
- 2026-11-03: Deep dives 04–08 on watching, vlens, magician/zigzag, neural-pairing, Manifold-SAE, retained-reply-state and
  claude-introspection. Read only: nothing was rerun. Each note says what was checked against result files and what wasn't.

## Things to maybe tell the author (kindly)
- `claude-introspection`: the permutation null scores null pairs against a baseline that includes them but the observed pair
  against one that excludes it, which can make "best of ~60" read as p < 1e-6 (note 08).
- `watching` README: says "head 6's key"; the key is shared by query heads 4–7 (KV head 1), as its own circuit README says (note 04).
- `magician` README: "copies the sentence on 49%" vs 52% in `compliance.json` (note 06).
- `Manifold-SAE` README still describes `W_k`/`B_k`, which the current code removed (note 08).
