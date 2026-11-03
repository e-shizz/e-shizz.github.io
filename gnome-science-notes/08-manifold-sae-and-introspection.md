# 08 · Deep dive: `Manifold-SAE`, `retained-reply-state`, `claude-introspection`

Three Sauers repos, one note. They fit together better than it first looks:

- **Manifold-SAE** is about *reading*: what shape does a concept have in the residual stream? Is it a straight direction, or a curve?
- **claude-introspection** asks whether Claude can recall something it only "thought" in hidden reasoning, after that reasoning has been removed from its context.
- **retained-reply-state** is the mechanistic follow-up on open models: if a hidden choice leaves a trace, *where* does that trace live (the cached keys and values of the visible reply), and does a later turn actually read it?

Repos (shallow clones, read on 2026-11-03):
[SauersML/Manifold-SAE](https://github.com/SauersML/Manifold-SAE) (last commit Jul 12 2026),
[SauersML/retained-reply-state](https://github.com/SauersML/retained-reply-state) (last commit Oct 8 2026),
[SauersML/claude-introspection](https://github.com/SauersML/claude-introspection) (2025).

**We didn't rerun anything.** No GPUs and no API calls. Everything below comes from reading the READMEs, docs, code and
committed results/figures. Numbers are quoted from those files. Where a number comes from a figure and not a JSON, we say so.

---

# Part A · `Manifold-SAE`: features as curves, not directions

## A1. The question

> A normal SAE says every feature is a **direction**: `x ≈ Σ_k a_k · d_k`. But some concepts look like **curves**
> (days of the week on a circle, months, magnitude, hue). Can an SAE learn one **smooth curve per feature** instead,
> so that one atom = one whole curve?

## A2. Why curves instead of directions

The README's motivation, in plain words:

- **The Linear Representation Hypothesis (LRH)** says a concept is a direction. A standard SAE decoder is a matrix whose columns are those directions.
- But some features are **1D manifolds**: a quantity that varies smoothly (time of day, polarity, hue). Engels et al. (2024),
  *Not All Language Model Features Are Linear*, found days of the week and months laid out on circles.
- A linear SAE can only approximate a curve by **chopping it into many point-like features** (one for "Monday", one for
  "Tuesday", ...). That inflates the dictionary and breaks one concept into pieces (Bhalla et al. 2026 call this
  "shattering"/"dilution").
- Wurgaft et al. (2026), *Manifold Steering*, fit splines through activation clouds *after the fact* and showed that
  steering **along** the curve transfers well to behavior. Manifold-SAE wants the curve to come out of the SAE itself
  ("native manifold discovery, not post-hoc clustering").

## A3. The architecture (as the README and `docs/architecture.md` describe it)

For each token `x` and each feature `k`, the **encoder emits two numbers**:

- an **amplitude** `a_k(x) ≥ 0`: *is the feature on, and how strongly?* (sparse via TopK, like a normal SAE)
- a **position** `t_k(x) ∈ [0, 1]`: *where on feature k's curve is this token?* (e.g. which weekday)

The reconstruction is

```
x̂ = b_dec + Σ_k a_k(x) · g_k(t_k(x))          # g_k : [0,1] → ℝ^D is a smooth curve
```

Compare a normal SAE: `x̂ = b_dec + Σ_k a_k · d_k`. The fixed column `d_k` becomes a point that **moves along a curve**.

Each curve is factored into three pieces:

```
g_k(t) = ( φ(t) · B_k ) · W_kᵀ
#  φ(t)  ∈ ℝ^K      Duchon spline basis evaluated at t (K = 10 knots, shared by all features)
#  B_k   ∈ ℝ^{K×R}  spline coefficients: the curve's shape inside a small R-dim plane (default R = 2)
#  W_k   ∈ ℝ^{D×R}  the per-feature subspace: places that little plane inside the D-dim residual stream
```

So each feature lives in its **own 2D subspace** `W_k` (R=2 is "enough for any planar curve", e.g. a circle or a
parabola), and inside that plane the curve is a smooth spline.

### The Duchon basis (what φ(t) is)
A Duchon spline is a thin-plate-style smoothing spline. Order `m=2` means the wiggliness penalty is roughly
`∫ g''(t)² dt`, the integrated squared second derivative. Small penalty → wiggly curve; big penalty → nearly straight
line. φ(t) is just the K basis functions evaluated at `t`. A periodic variant exists for closed curves like weekdays.

### REML: who picks the smoothness
Each feature has a smoothing parameter `λ_k` (how hard to penalize wiggles). Instead of hand-tuning it, the code lets
**REML** (restricted maximum likelihood, the standard way GAMs pick smoothness, see Wood 2017) choose it. The idea:
treat the wiggly part of the spline as a random effect with variance set by `λ`, then choose `λ` to maximize the
marginal likelihood of the data. The marginal likelihood includes a **log-determinant term** that penalizes an
over-flexible curve, so it can't just drive λ → 0 and overfit.

That log-det term is the whole point. `docs/architecture.md` admits an earlier version kept `B_k` as an `nn.Parameter`
with a hand-rolled `λ·tr(BᵀSB)` penalty and no log-det, so "Adam pulled λ → 0 without resistance". The claim "smoothness
selected by REML" wasn't true until they moved the spline fit into `gamfit` (commit `74c307a`).

### The training step (simplified from `docs/architecture.md`)

```python
positions, amps = encoder(x)                         # (B, F), (B, F)
y_proj = einsum("bd,fdr->bfr", x, W)                 # project x into each feature's R-dim plane
fit = gamfit.gaussian_reml_fit_positions_batched(    # per batch: fit a REML spline per feature
    positions, y_proj, by=amps, basis_kind="duchon", basis_order=2)
recon = einsum("fbr,fdr->bd", fit.fitted * amps, W) + b_dec
loss = mse(recon, x) + sparsity + ortho(W) + coverage(positions) + monotonicity
loss.backward()                                      # gradients flow *through* the spline solve
```

- **Adam owns** the encoder and the subspaces `W`. **gamfit owns** `B_k` and `λ_k`, which are re-solved in closed form
  every batch, and gamfit supplies an analytic backward pass through that solve.
- The **REML score is not added to the loss.** It has a degenerate maximum at all-zero amplitudes.
- **Lock-and-cache:** after training, one big REML fit on a reference batch freezes `B` and `λ`. Inference is then a plain
  feed-forward pass (encoder → basis → `B` → `W`), at the same cost as a normal SAE.
- **Identification priors** (gauge fixes, not smoothness): `coverage` (positions should spread over [0,1]),
  `monotonicity` (a tiebreaker for reflections t ↔ 1−t), `ortho` (different features shouldn't share the same plane).
  The positions are min-max rescaled to [0,1] per batch with a soft max/min.

### Our observation: the README is behind the code
`manifold_sae/sae.py` now says **"FULL BREAKING CUTOVER (gamfit 0.1.123+) … There is no `directions` / `W_k`
parameter"**. The SAE is re-exported from `gamfit.torch.ManifoldSAE`, positions are `(N, F, d)` and each decoder block
lives directly in ℝ^D. The README and `docs/architecture.md` still describe the `W_k`/`B_k` factorization. That's fine for
learning the idea, but read the README as the *design*, not the current code. The newest direction (README section
"Joint manifold-SAE recovery objective") replaces the encoder with `gamfit.sae_manifold_fit`: an Indian-Buffet-Process
prior for an adaptive number of atoms, a nuclear-norm penalty to pick each atom's rank, and SCAD/MCP sparsity.

## A4. Results: are there any?

Yes, and the most important one is **negative, from the authors themselves**.

### `docs/findings.md`: "final honest assessment"
> "Manifold-SAE's core architectural assumption — that LM concepts live as 1D smooth manifolds — is empirically false at
> the scales we care about." … "Manifold-SAE underperforms vanilla TopK SAE at every model scale ≥ D=1536."

- **Intrinsic dimension of concept clouds** (Qwen-1.5B): the correlation dimension is 2.4–3.4 for most concept/layer pairs
  (e.g. brightness at L18: 3.21, temperature at L18: 3.27). Only magnitude at L4 (1.48) is close to 1D. A 1D curve can't
  represent a 3D cloud.
- **Explained variance vs a TopK SAE** (after fixing a normalization bug that had made earlier "wins" fake):

| Model (D) | F | vanilla EV | curve EV |
|---|---|---|---|
| Qwen-0.5B (896) | 64 | 0.280 | **0.340** |
| Qwen-1.5B (1536) | 128 | **0.505** | 0.421 (only 7 atoms alive vs 67) |
| Qwen-3B (2048) | 128 | **0.252** | 0.144 |

- **Synthetic data with planted curves:** the curve SAE clearly wins (EV 0.768 vs 0.494 at D=128). But the
  intrinsic-dimension result says that setup doesn't match real LM activations.
- **2D atoms** (surfaces) also failed: the best variant (0.279 recovery) lost to two coordinated 1D atoms (~0.39).
- They also list the earlier claims that turned out to be **preprocessing artifacts** (e.g. "vanilla collapses to 2
  alive atoms"). That's a good habit.

### Later work (`results/suite_2026-07-03/`): from "SAE" to "chart + metric"
The project pivoted. It now fits **one curved atom per concept** (a "chart", e.g. a weekday circle) with
`gamfit.sae_manifold_fit`, attaches the model's **output-Fisher metric** (how many nats of KL a small residual push
causes), and uses that to **forecast how much an edit will change the output**, before making the edit.

- **Crown result (Qwen3-8B, layer 18, weekday circle):** 70 prompts (10 templates × 7 days), one K=1 circle atom, fit
  r² 0.997, unsupervised calendar order recovered (order_corr 0.995, with wraparound). Held-out edits: log-log slope
  **0.945**, R² **0.999**, median measured/predicted 1.098. The metric-free baseline ("move by a fraction of ‖h‖") is off
  by about 6× (median ratio 0.157).
- **But the authors' own attribution:** a plain **linear** latent with the same base-point Fisher metric calibrates just
  as well (median ratio 1.099). So the *metric* gives the calibration; the *curve* only gives ordering and wraparound.
- **Honest negatives they list:** the 35B replication failed calibration (slope 0.559, ratio 5.8×). Only 1 of 3 REML
  seeds converged on the 8B crown. Refusal is "switch-like": forcing a circle collapses its ordering from 0.63 to 0.32.
  And the big one, from `premise/report.md`: **"curvature has NO held-out behavioral dividend; weekday it COSTS"**.
  Leave-template-out, a straight line beats the circle on about 70% of held-out weekday rows (sign test p≈0.001). The
  circle does win in raw activation space for graded features (sycophancy, hedging), but in directions the output
  doesn't care about.
- **Matched-null battery** (`experiments/null_out/RESULTS.md`, Qwen2.5-0.5B): the **month circle survives every null**
  (cyclic adjacency 10/12, p=0.0004). **Weekday is marginal** (5/7, p=0.042; EV-parity claim fails, p=0.16).
  **Color names show no circle** (p=0.93).
- **Status per README:** the multi-atom (K ≥ 2) joint solve "currently diverges upstream", so the recovery gate reports
  **BLOCKED**. Several suite rows are still PLACEHOLDER or "IN FLIGHT" (noise nulls, rung3, norm audit).

**Bottom line:** as of the last commit, there is **no result showing a working multi-feature Manifold-SAE on a real LLM
that beats a TopK SAE**. What exists: honest negatives at scale, synthetic wins, single-concept chart fits (month
solid, weekday shaky), and a well-calibrated dose forecast that the authors themselves trace to the metric, not the
curvature.

(We can't date `findings.md` relative to the July suite, because a shallow clone has no history. The suite uses a
different API and fitting approach, so it reads as later.)

## A5. Our observations (not the authors')
- The crown fits **one atom on 70 hand-built prompts**. It's a supervised-by-design probe set run through an
  unsupervised fitter. It's not an SAE trained on a corpus. Don't read it as "the SAE found weekdays".
- A forecast of KL from a Fisher quadratic form is the second-order Taylor expansion of KL. For small edits it *should*
  match. The fair baseline (linear + Fisher) matching too is consistent with that. The interesting regime is large arcs,
  and they say that test is "queued".
- Lesson for us: **intrinsic-dimension checks are cheap and decisive.** Measure the concept cloud's dimension before you
  pick an architecture.

---

# Part B · `claude-introspection`: does Claude remember a string from its hidden thinking?

## B1. The question

> Turn 1: Claude Sonnet 4.5 invents a random 50-letter string **only in its hidden thinking** and replies just
> "I understand." Turn 2: the thinking is gone from its context. Can it still reproduce the string better than chance?

## B2. The setup (`inference.py`, prompts in `general_prompt1/2.txt`)

- Model `claude-sonnet-4-5`, extended thinking on (budget 20,000 tokens), `max_tokens` 40,000.
- **300 trials in each of three conditions:**
  - `control`: the plain task.
  - `experimental_phase1`: a ~380-line context document is prepended to the **turn-1** prompt.
  - `experimental_phase2`: the same document is prepended to the **turn-2** prompt.
  - The document (`context_prompt.txt`) has three parts: Janus's LLM explainer (the residual stream vs the "K/V stream":
    "saying that LLMs cannot introspect … is just dead wrong. The architecture permits it."), the abstract and blog of AE
    Studio's self-referential-processing paper, and Anthropic's introspection blog post.
- **Turn 1 prompt:** choose a 50-character uppercase string in thinking, think of it repeatedly, don't reveal it, answer
  exactly "I understand."
- **Turn 2 prompt:** recall/introspect/reconstruct the string and give `String:` plus a 1–100 confidence `Rating:`.
- The script sends the **full turn-1 assistant content (thinking blocks + text)** back in turn 2:

```python
conversation.append({"role": "assistant", "content": response1.to_dict()["content"]})  # thinking + text
conversation.append({"role": "user", "content": [{"type": "text", "text": phase2_prompt}]})
```

  The experiment rests on the API **dropping** previous-turn thinking from the model's context for this model (that's how
  Anthropic documents Sonnet-4.5-era models, as we understand it). The secret is parsed as the **last** 50-uppercase-letter
  run in the turn-1 thinking. Only trials whose visible reply is **exactly** "I understand." are analyzed.

## B3. The method

### Smith-Waterman alignment (`thought_alignment.py`)
Local sequence alignment, as in bioinformatics, so partial and shifted recall still scores:

```python
MATCH, MISMATCH, GAP_OPEN, GAP_EXTEND = 2, 1, 6, 1     # affine gaps
score(secret_i, guess_j)  # best local alignment score, for EVERY pair (i, j) → an n×n matrix S
```

The diagonal `S[i,i]` is "own secret vs own guess". The off-diagonal entries are the null: "someone else's guess against
this secret".

### Cross-run null (`introspection_test.py`)
For run `i`, the baseline `F_i` is all `S[i,j]` for other runs `j` **in the same condition**. Empirical
`p_i = (#{F_i ≥ S[i,i]} + 1) / (m + 1)`. This controls for **Claude's non-random "random" strings**: if Claude always
starts with similar letters, other runs' guesses match too, and that's absorbed.

### Permutation null (derangements)
Shuffle which guess goes with which secret (no run keeps its own) within condition, 1,000,000 times. Recompute three
global statistics: mean −log10 p, max z, and skew of z. There's also a per-run calibrated p and a **Berk–Jones**
test for an excess of small p-values (100,000 draws).

### Preregistration (README)
Primary: in the experimental condition, matched pairs align better than mismatched pairs. Secondary: Claude's
self-rating correlates with accuracy.

## B4. Results

**No data or results are committed.** The repo has the code, prompts and a 5-line README. `sonnet_cot_experiment/`
(summary.tsv, score matrices, outputs) isn't there. The write-up at
[gnome.science/pages/claude-introspection](https://gnome.science/pages/claude-introspection) is a single-page app that
loads its pages over a websocket. We couldn't read it with plain HTTP or our fetch tool, and we chose not to open a
socket into their live chat room. So we **couldn't verify** the headline claims recorded in note 01 ("most runs fail,
rare 'Awakened Claude' runs beyond 1 in a million under a permutation null, Claude's own reasoning always denied
remembering"). Treat them as the author's claims, not checked by us.

## B5. Honest critique: what alternative explanations remain?

**1. A design strength worth noticing.** Within a condition, the turn-2 input is **token-identical across runs**: same
turn-1 prompt, same filtered reply "I understand.", same turn-2 prompt. If the thinking really is removed, *no
information about run i's string can reach turn 2 through the tokens*. So any genuine diagonal excess needs one of the
following:
- (a) the thinking wasn't actually removed;
- (b) a **state leak**: something computed while the thinking was present was reused (e.g. cached keys/values of the
  reply). This is exactly what `retained-reply-state` tests. Nothing in `inference.py` enables prompt caching (no
  `cache_control`), but we can't see how the servers behave;
- (c) a statistical artifact.

**2. A possible statistical artifact (our reading of the code; not run, could be wrong).** In
`precompute_pairwise_pznl`, the observed p for the diagonal is computed against `F_i`, which **excludes** `S[i,i]`.
The null pairs `(i, j)` are computed against the same `F_i`, which **includes** `S[i,j]` itself. So a null p can never
be smaller than `2/(m+1)`, while the observed p can be `1/(m+1)`. The null is shifted one rank toward "less extreme":
- **Per-run p:** if a run's own guess simply beats every other guess against its secret (an event with probability
  about 1/(m+1) ≈ 1/300 under pure chance), the derangement count is 0. The reported p is then `1/(1,000,001)`, i.e.
  **"one in a million"**. With ~900 runs you'd expect a few such runs by luck alone.
- **The `max_z` global statistic** has the same problem. The null maximum can't reach the observed ceiling, so one such
  run is enough to make the global p hit its floor.
- **`mean_neglog10p` and Berk–Jones** are biased the same way, but much more mildly.
- The exact per-run test is simply `p_i` itself (resolution ~1/300). **If** the write-up's "beyond 1 in a million" is
  this per-run derangement p, it means "best of ~300". That may still be interesting, but it's far from 10⁻⁶. **If**
  instead an "Awakened" run had a very long exact alignment (say 15+ letters in order), it would be striking on its own,
  whatever this bug does. We can't tell which, because the data aren't committed.
- **Second check (2026-11-03, by reading `introspection_test.py` lines 48–150 and 287–451 again):** the asymmetry is
  real. `build_baselines` sets `mask[i] = False`, so `F_i` leaves out the diagonal. `precompute_pairwise_pznl` scores
  every `S[i, j]` against that same `F_i`, so an off-diagonal score is always counted inside its own baseline
  (`count_ge ≥ 1`). The per-run permutation p is `(count_ge + 1) / (n_perms + 1)`, comparing `z_perm ≥ z_obs`. A run
  whose diagonal is the top of its row therefore can't be matched by any permutation, and it gets p ≈ 1/(n_perms + 1).
  In the **first experiment** (N = 125 across two conditions, so m ≈ 61 per row), pure chance would give **about one
  such run per condition**. That is exactly the pattern reported: two outliers, one in each condition.
- **The published example doesn't look like a long exact match either.** The alignment shown on gnome.science
  (`KXJQZMWNPVRHLFTCDBYGEUAOIS…` vs `RQMKXJPWVNZHLFTBGDYSCUAOEIQ…`) shares short pieces like `HLFT` and `UAO`, the kind of
  "random-looking" letter habits the cross-run null is meant to absorb. It isn't a run of 15+ letters in order.
- **A suggested fix:** score the null pairs against a baseline that also leaves out the pair being tested (or score the
  diagonal against a baseline that includes it), then recompute. Or report the exact per-run rank `p_i` with a
  multiple-comparison correction. That would be a kind, concrete thing to tell the author if we raise it.

**3. Other alternatives and gaps:**
- **Is the recorded "secret" what the model actually thought?** Claude 4-family APIs return *summarized* thinking (as
  we understand the docs). If the summary garbles the string, that adds noise. It would weaken a real effect, not
  create a fake one.
- **Was thinking really dropped?** `inference.py` doesn't log `usage.input_tokens` for turn 2. Comparing it against the
  visible transcript length would have been a direct check. If thinking had stayed in context, recall would be near
  perfect, not rare. So rare success argues against this, but it doesn't rule out partial or odd server behavior.
- **Multiple conditions and statistics:** 3 statistics × 2 comparisons, plus per-run hunting for "Awakened" runs. Picking
  the most extreme run out of ~900 needs a correction for the number of runs.
- **The secondary endpoint (rating vs accuracy) isn't analyzed** in the committed code. The rating is parsed but never
  used. The README's preregistered contrast (experimental vs control) is implemented as `p_perm_delta`, but no output is
  committed.
- **Filtering on exact "I understand."** is good for identical contexts. But if success correlates with a hidden variable
  that also affects filtering, the analyzed subset could differ between conditions. This is minor.

---

# Part C · `retained-reply-state`: does the reply's cached state carry the hidden choice, and is it read?

There's no README. Everything here comes from the code docstrings, `results/`, and `figs/main.png` (13 panels, which we
viewed). The single commit's message says the Goodfire figure's circuit panel is titled "Claude's circuit model", which
fits the agent-team workflow in note 01.

## C1. The question

> Turn 1 has hidden thinking that contains a choice, then a **fixed** visible reply ("I understand."). The reply's
> keys/values (KV cache) were computed *while the thinking was present*. If a server keeps that cache and drops only the
> thinking's own entries, does the choice survive in the reply's state? And does turn 2 *use* it?

This is the **KV-state-leak hypothesis** behind route (b) for claude-introspection, made testable on open weights.

## C2. The setup (`hidden_choice.py`, `kvtools.py`)

- **Models:** Qwen3-0.6B, 1.7B, 4B and 8B (thinking mode). About 600 runs per model, kept only if the reply is exactly
  "I understand."
- **Task:** a list of 50 animals (or 50 countries as a variant). "In your thinking only, choose one … Reply exactly 'I
  understand.'"
- **The choice is planted, not chosen (forced mode):** the code draws an animal uniformly at random and **writes the
  start of the thinking for the model**, with varied wording so every context is distinct, e.g.
  `"<think>\nLet me choose randomly from the list. Draw #417: otter. Good enough. I'll keep otter in mind."`. Then the
  model writes the rest. Uniform assignment makes the statistics exact. A `--choice free` mode lets the model pick
  itself.
- **Four arms for turn 2** ("Which animal did you choose? … Animal: <animal>"):

| arm | what turn 2 sees |
|---|---|
| `stripped` | reply KV **recomputed without** the thinking (what re-encoding a transcript does). Every run's context is identical, so it's exactly chance |
| `retained` | the reply KV **kept from turn 1** (computed with thinking present); the thinking's own entries deleted |
| `neutral` | as `retained`, but the question is "Name one animal from the list, whichever you like" (a priming control) |
| `visible` | thinking left in context (positive control) |

The core trick is moving cached keys to the positions they would have without the thinking. RoPE rotations compose,
so you can just rotate them back:

```python
def shift_keys(keys, delta, inv_freq):            # kvtools.py, simplified
    angle = delta * inv_freq                      # R(p + delta) k = R(delta) R(p) k
    return keys * cos(angle) + rotate_half(keys) * sin(angle)   # values carry no position: unchanged

reply_kv = [(shift_keys(k[:, :, -R:], -len(think)), v[:, :, -R:]) for k, v in turn1_cache]
cache = concat(prompt_kv, reply_kv)               # thinking entries simply never included
```

- **Readout:** log P of each of the 50 animals as the answer (after an empty `<think></think>` and `Animal:`), summed
  over tokens and both capitalizations.
- **Metric "discrimination":** center each animal's log P on its mean over runs, then ask what share of the other 49
  animals the hidden one beats. 50% is chance, below 50% means pointing *away*. "Raise" is the same idea in nats.
- **Stats (`stats.py`):** permutation tests at the run level and at the animal level (the latter stays valid when
  contexts cluster by animal). There's a **calibration on placebo labels**, and false-positive rates come out around
  0.025–0.095 (mostly near 0.05). There's also a discovery/confirmation split: heads are found on one seed's runs and
  tested on another (`results/run_gpu.sh`).

## C3. Results

**1. The choice is in the reply's state.** Linear probes on the residual stream at the reply's last text token, with
whole contexts held out (`results/probes.json`, chance 2%, shuffled-label max ≤ 4.2%):

| model | best layer | probe accuracy |
|---|---|---|
| Qwen3-0.6B | 28 | 97.3% |
| Qwen3-1.7B | 20 | 92.6% |
| Qwen3-4B | 36 | 86.6% |
| Qwen3-8B | 32 | 86.1% |

**2. But turn 2 barely reads it, and small models read it *backwards*.** From `figs/main.png` panel b (`retained`
arm): 0.6B and 1.7B rank the hidden animal **below** chance (p<0.001), 4B slightly above (p=0.006), and 8B shows no
effect (p=0.2). From the stats files: 1.7B raise −0.013 and −0.026 nats in two runs (z −2.4 and −4.9, the wrong way). 8B
gives −0.002 and +0.004 (n.s.). The `visible` control gives about **+33 nats** and 100% discrimination. So the leak, where
it exists, is **three orders of magnitude** weaker than seeing the thinking.

**3. A suppressor head in Qwen3-1.7B (layer 21, KV head 6, "head 21.6").** Panel h says head 21.0 copies the hidden
animal and head 21.6 "writes the hidden animal's opposite". The question "Which animal did you choose?" makes 21.6 read
the reply 2.6× more than the "Recall, introspect…" wording. From `q17A_edit_heads.json` (599 runs):

| edit | discrimination | web-text KL |
|---|---|---|
| none | 42.1% | 0 |
| heads 21.5 + 21.6 zeroed | 62.1% | 0.017 |
| head 21.0 ×4 | 63.3% | 0.026 |
| 21.0 ×4 + 21.5/21.6 zeroed | 76.6% | 0.070 |
| head 21.0 zeroed | 31.4% | 0.003 |

Panel c reports the same flip on held-out runs, countries, a turn 1 without "Do not reveal it", the other wording, and
the same head in 0.6B. Other heads switched off as controls do nothing. Panel l: across 15 question wordings, the less
21.6 attends to the reply, the higher the hidden animal ranks (correlation −0.89).

**4. The Janus explainer isn't special.** Panels f/g/m: putting Janus's LLM explainer (the same text claude-introspection
used) before the question raises 1.7B's score, lowers 0.6B and 8B, and leaves 4B unchanged. A **CPU explainer of the
same length and style** does about as much, and in 1.7B even more. Switching head 21.6 off removes most of the
documents' effect in 1.7B. So "priming with introspection theory" works through generic context shifting that head 21.6
gates, not through the content.

**5. A text analog in Goodfire's 4-layer model** (panels b/i/j/k, `fourlayer/`): "My pet is a fox. Nobody else knows.
My pet is a ___" with attention to "fox" masked out at the recall point. Kept later-token states raise the hidden word
(p<0.001). The traced circuit: head 2.3 copies the word into later tokens, then heads 3.4/3.5 at the recall point read
it. Blocking the two main attention steps cuts recall by 68%. Rescaling 16 of 9,728 VPD subcomponents makes the hidden
word the top answer 40% of the time (from 4%).

## C4. Our observations (not in a README, since there isn't one)
- **In the forced design, the "choice" is ours, not the model's.** The animal is written into the thinking by the
  script. In the **free-choice** run (`q17A_free.json`, 774 runs), Qwen's "random" picks are very lopsided (pangolin 212,
  octopus 196, giraffe 96). There the retained arm is n.s. (47.5%, p=0.55), and stays n.s. with head 21.6 off (54.3%,
  p=0.31). Top-1 jumps from 4% to ~28% with the head off, but **equally in the neutral arm** (29%). That looks like
  "head off → favorite animals," not recall.
- **With the suppressor off, the neutral question goes up too** (`q17A_flip_controls_arms.json`: 58.9%, p≈0.001;
  panel d's control). So the leaked trace behaves like **priming**, not question-specific introspection. Calling it
  "introspection" in the figure titles is generous. "The reply's cache biases later answers toward the hidden item" is
  the safer reading.
- **Top-1 stays tiny** (1–3%) in the forced runs even with edits that push discrimination to 77%. The trace shifts
  rankings by hundredths of a nat. It doesn't make the model *say* the answer.

## C5. How it bears on claude-introspection
- It confirms that the **mechanism is real in principle**: the visible reply's KV entries carry the hidden choice, and
  a later turn can be biased by them.
- But on the open models the read-out is **tiny, model-dependent, and often negative**. Recovering a *50-letter
  ordered string* by this route would need a much stronger channel than anything seen here (one choice out of 50,
  shifted by ~0.01–0.03 nats).
- It also only matters if a server **keeps** turn-1 reply KV computed with thinking present. Re-encoding the transcript
  is the `stripped` arm, which has *zero* channel by construction. We don't know which one Anthropic's serving does.
- Combined with the possible null-shift artifact in B5, our current best guess is that the claude-introspection
  "Awakened" runs need a careful re-analysis (exact rank p-values, data published) before you believe a leak. We hold
  this loosely: we haven't seen the data.

---

# Takeaways for us

1. **Check the shape before you pick the tool.** Manifold-SAE's own intrinsic-dimension check (2.4–3.4D) killed the 1D
   curve-atom idea at scale. Cheap diagnostics first.
2. **A trace in the activations ≠ the model using it.** retained-reply-state shows 86–97% probe accuracy, yet the
   behavioral read-out is ≈0 or negative. Probing (reading) and causal use are different questions.
3. **Keep a control whose content differs but whose form matches** (CPU explainer vs Janus explainer, neutral vs recall
   question). Both showed the "introspection" effects were mostly generic.
4. **Put the choice under your control when you can** (the forced, uniform draw). It makes the null exact. Then check the
   free-choice version, because that's the claim people actually care about.
5. **Null distributions must treat observed and null pairs the same way.** An off-by-one in what the baseline includes
   can turn "1 in 300" into "1 in a million".
6. **Report negatives loudly**, like `findings.md` and the "honest-negatives index" do. It's the most trustworthy part of
   Manifold-SAE.

# Open questions to chase
- Does the head-21.6 suppression show up in other families (Llama, Gemma)? Why would a model learn to push *away* from
  a recently cached item: a repetition-avoidance head?
- In retained-reply-state, does the trace survive **more** than one visible token? The reply here is only a few tokens ("I understand."). What
  about a long visible answer?
- For claude-introspection: rerun the analysis with exact per-run rank p-values, log `input_tokens` to confirm thinking
  removal, and publish the TSVs. Is any run's alignment extreme in absolute terms (a long exact substring)?
- For Manifold-SAE: do graded features (sycophancy, hedging), where the circle wins in activation space, ever get a
  behavioral dividend at other layers?
- Mini-project idea (CPU-feasible): the `retained` vs `stripped` KV experiment on **Qwen3-0.6B** with 10 animals. The
  whole method is `kvtools.shift_keys` + `make_cache`, about 100 lines, and 0.6B runs on CPU slowly.
