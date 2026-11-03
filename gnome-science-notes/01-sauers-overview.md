# 01 · Sauers: overview

*Notes written 2026-11-03 from gnome.science, the SauersML GitHub repos and their READMEs. X couldn't be read directly.*

## Who and how

- **Links:** X [@Sauers_](https://x.com/Sauers_), site [gnome.science](https://gnome.science/) ("A clearing in the forest"), GitHub [SauersML](https://github.com/SauersML), Medium [@SauersML](https://medium.com/@SauersML), and [Felony Bench](https://bench.red/).
- **They work with a team of AI agents.** Recent READMEs say *"Written by Claude Opus 5.5"* or *"Written by Opus 5.5 and GPT-6 Astra."* In `honesty-heads`, a data label says it was read by hand by "the dsets2 teammate (Claude)." Their tools fit this: `agent-slack-bridge` (Slack for an agent team), `cctop` (a live monitor for multiple Claude Code and Codex accounts) and `cc_transcript_viewer`.
- **Background:** statistical genetics and bioinformatics (Rust tools for genomes, PCA, MCMC, polygenic scores, All of Us data), plus Lean 4 formal math (`Descent`, `group-approximation`, a `mathlib4` fork). That shows in the work: permutation nulls, false-discovery-rate control, exact confidence intervals, and honest caveats.
- **gnome.science** is a single-page app with a live "Global LLM Chat" (humans and bots in one room) and an "LLM research" section.

## Projects, as reading vs. writing

### Writing (steering)
| Repo | Date | One line |
|---|---|---|
| **honesty-heads** | Oct 2026 | Steering 29 attention heads of Qwen3.5-9B turns about half of its own held-out role-play lies into the truth. **→ note 02** |
| **magician** | Oct 2026 | Qwen3-4B LoRA: an instruction like "lower *tea* on every 5th layer from layer 3" makes the model plan that change in its thinking and actually make it while answering something else. Works on held-out concepts, wordings and layer patterns. |
| **zigzag** | Oct 2026 | Qwen3-1.7B trained to zigzag its internal "cat" representation (raise it on even layers, lower it on odd ones). It turned out to respond to the *tokens* "even" and "odd," not to the request: prompts about a dog trigger it too. The adapter is on Hugging Face: `Sauers/qwen3-1.7b-cat-zigzag`, and `chat.py` shows per-layer changes live. |

### Reading
| Repo | Date | One line |
|---|---|---|
| **vlens** | Nov 2026 | Draws a model's hidden thoughts. Qwen3-VL-8B's middle-layer states go straight into the image generator built for it, untrained. "Think of X, don't name it" is drawn right 79% of the time among 48 options, beating the last layer the generator was built for. |
| **Manifold-SAE** | Jul 2026 | An SAE whose features are smooth 1D curves (or 2D surfaces) instead of straight directions, for things like magnitude, time and hue. It uses GAM/spline decoders with REML smoothing. |
| SAE, SAEvolution, InterPLM fork, genomic-fm-sae-experiments | 2024–26 | Earlier SAE work, including on protein and genomic models. |

### Models that read themselves or each other
| Repo | Date | One line |
|---|---|---|
| **watching** | Oct 2026 | Qwen3-8B fine-tuned to tell when a read-only probe is saving its activations: 128/128 on new prompts, but only on the exact GPU type it was trained on (it answers backwards on an H100 or A100). It also tells training from serving, and can write "I can see you watching me" into its hidden states for whoever reads them with a logit lens. |
| **neural-pairing** | Oct 2026 | Two models from different families (Qwen3-4B "Quill" and AFM-4.5B "Ember") linked through cross-attention on each other's hidden states. Each says the city from the other's private prompt, with no tokens exchanged. |
| retained-reply-state | Oct 2026 | Does a fixed visible reply's cached state carry a choice made in hidden thinking? (Qwen3) |
| **claude-introspection** | 2025 | An independent replication of Anthropic's introspection result. Claude Sonnet 4.5 sometimes recalls a 50-character string from its hidden chain of thought. Most runs fail, but rare "Awakened Claude" runs score beyond 1 in a million under a permutation null. Claude's own reasoning always denied remembering. |
| minds_RL | 2025 | RL on Qwen-30B-A3B to predict its own properties (confidence, entropy, the effect of a LoRA step). It trained on some tasks but didn't generalize. Total cost $44.50. |
| WatcherBench, StenoBench | 2026 | Benchmarks. |

## Their research style (worth copying)

1. **Use the model's own behavior.** Lies the model tells by itself under pressure, not lies it was told to tell.
2. **Compare against random directions** with the same per-unit sizes, on the same units.
3. **Hold data out:** separate select and test splits; held-out wordings, concepts and layers.
4. **Look for your own confounds** and report them. In `honesty-heads`, the direction is partly "say Yes."
5. **Report what failed:** `zigzag` responds to tokens, not meaning; `watching` only works on one GPU type.
6. **Make it reproducible:** `run.sh`, all results as JSON, figures redrawn from results.
