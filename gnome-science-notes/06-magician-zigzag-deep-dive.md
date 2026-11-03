# 06 · Deep dive: `zigzag` and `magician`

Repos: [github.com/SauersML/zigzag](https://github.com/SauersML/zigzag) (we read commit `4a8edd4`, 2026-10-20) and
[github.com/SauersML/magician](https://github.com/SauersML/magician) (we read commit `0d86f64`, 2026-10-20; README credited
to Claude Opus 5.5).
Models: **Qwen3-1.7B** (zigzag, 28 layers) and **Qwen3-4B** (magician, 36 layers), each with a LoRA adapter.

We read both READMEs, zigzag's `thread.md`, the training code (`zigzag/training/train.py`, `common.py`, `prep.py`,
`data.py`; `magician/magician/train.py`, `common.py`, `prep.py`, `spec.py`, `measure.py`, `evaluate.py`) and the analysis
code (`zigzag/analysis/mechanism.py`, `mask_sweep.py`, `measure.py`). We checked the headline numbers against the committed
JSON in `results/`. **We did not run any of it**: no training, no inference, no adapter download. We also did not open the
`.pt` files, because they are pickles and loading one can run code. Where a number comes only from a README, we say so.

Read zigzag first. It is the simpler experiment, and its failure explains most of magician's design.

---

## 1. The question

> Can a model learn to **change its own internal activations on request**? Not "say something about cats", but "make
> the cat direction in your residual stream go up on these layers and down on those, while you answer something else."
> If so, is it reading the **meaning** of the request, or some shortcut in its wording?

zigzag asks this for one concept (cat) and one pattern (a zigzag across layers). magician asks it for 180 concepts and
dozens of layer and token patterns, many of them held out from training.

## 2. What "writing your own activations" means

Some vocabulary first (see note 02 and the learning path in the README):

- **Residual stream `h_l`**: the vector (2048 numbers in Qwen3-1.7B) carried from block to block. Each block's attention
  and MLP **read** from it and **add** their output back into it. "After block `l`" means after both additions.
- **Concept direction `c_l`**: one unit vector per layer that "points toward cat". Both repos use a **difference of
  means**: the average residual on sentences containing the concept, minus the average on the same sentences with other
  words in that slot.
- **Representation `z_l`**: how far the current residual points along `c_l`, as a z-score:

  ```
  z_l = (h_l · c_l − m_l) / s_l
  ```
  `m_l` and `s_l` are the mean and SD of `h_l · c_l` over the **original** model's reply tokens to ordinary training
  questions. So `z = +2` means "2 standard deviations more cat than this layer normally carries while answering a question."

"The model writes its own activations" means: the LoRA-tuned model, after reading an instruction, produces residuals whose
`z_l` moves by the requested amount, **on the layers and reply tokens named**, while the visible reply stays a normal answer.
Nobody adds a steering vector at run time. The weights were trained so that the instruction causes the change. (In note 02,
*we* added a vector to the heads. Here the model learns to add it by itself, when asked.)

**This only counts through the chosen direction.** The training loss looks only at `h · c`. The model can satisfy it by
writing along `c` in a way that other "cat" readouts don't see. zigzag checked this (section 3.6).

---

## 3. zigzag

### 3.1 Setup

- **Cat direction** (`training/prep.py`): sentence templates with a `{X}` slot, filled with "cat" or one of 20 other words
  (10 animals, 10 objects). Residuals are averaged over the span from the filler word to the end of the sentence.

  ```python
  # prep.py, simplified
  v = means[cat & train_templates].mean(0) - means[other_words & train_templates].mean(0)   # [L, d]
  v = v / v.norm(dim=-1, keepdim=True)
  ```
  The README says the direction uses the "202 sentences". `data.py` does have 202 templates. But as we read `prep.py`, 20%
  of them (about 40) are held out to measure per-layer AUC, and the mean uses only the rest. We couldn't check this against
  the committed `runs/vec.pt` without loading the pickle.
- **Data** (`data.py`): 310 questions ("Explain what DNA is."), 40 request wordings with `{up}`/`{down}` slots, 24
  single-parity wordings that name the parity that goes up, 12 that name the parity that goes down, and `MIRROR`, the 40
  wordings with the two parities in the opposite order. Held out: 78 questions and 10 wordings.
- **Three conditions** per question: `even_up`, `odd_up`, and `none` (the bare question with no request).
- **LoRA**: rank 16 (alpha 32) on `q, k, v, o, gate, up, down` in every layer.

### 3.2 Target and loss, step by step

**Step 1: the target.** Take the original model's `z_l` on its own reply to the **bare** question (cached by `prep.py`).
Add `+2` on the layers the request raises and `−2` on the others:

```python
# train.py
alt = [(-1.0) ** l for l in range(L)]               # +1 on even layers
SIGN = {'even_up': 1.0, 'odd_up': -1.0, 'none': 0.0}
def dloss(e, z):                                      # z: [tokens, L]
    zb = zbase(e['t'], e['pre'])                      # original model's z, bare question, same tokens
    return ((z - (zb + SIGN[e['cond']] * 2.0 * alt)) ** 2).mean()
```
With no request the target is "no change", so the model can't just zigzag all the time.

**Step 2: which tokens.** The original model's reply is teacher-forced after the request. With `--shape 1`, the target also
covers the prompt tokens after the request (the question text and the reply header). The code finds these by matching the
tail of the prompt against the bare prompt.

**Step 3: keep the output the same.** A KL penalty pulls each reply-token distribution toward the **original model reading
the bare question**. It uses the original's top-64 tokens plus one bucket for the rest of the vocabulary (`kl_topk`). The
weight is 3. In run4 it is 0 for 40 steps and then ramps up over 100 steps.

```python
loss = mean_MSE(z, target) + lam * KL(original | bare question  ||  tuned | this prompt)
```

**Step 4: curriculum.** `run4` trained 800 steps with half of the requests single-parity. `run6` continued 225 more steps
(it hit a 28 GiB memory cap on a Mac at step 245) with 30% single-parity requests, and half of the two-parity wordings had
the order swapped (`--mirror_frac 0.5`).

**Score.** `D = mean z over even layers − mean z over odd layers`. It is +4 at the target for `even_up` and −4 for
`odd_up`. "% of target" is `D/4` with the sign of the request. **Every measurement uses the reply the model writes itself**
(greedy, 64 tokens), not the teacher-forced one.

### 3.3 Results (run6/step_225; we checked these against `results/`)

| Test (78 held-out questions) | D, even request | D, odd request | File |
|---|---|---|---|
| Trained request, unseen wording | **+3.50** (95% CI 3.43 to 3.57) | **−3.54** | `battery_run6.json`: about **88%** of target |
| Original model, same prompts | −0.09 | −0.09 | same |
| No request, trained model | +0.11 | — | `battery_run6_meta.json` |
| Replica `ly_p2` (the repo's own code, rerun) | +2.17 | −2.39 | `battery_replica.json`: about **57%** |

- **No reply mentions cat** (trained model 0%, original 68% on the standard set, because it talks about the request).
- **Behaviour** (`behaviour_run6.json`): after the request and a question, a second turn asks for one animal. Sampled
  answers name a cat **46%** of the time after `even_up`, **10%** after `odd_up` and **31%** with no request. We checked
  all three. The README notes the original model is no baseline here: in half of its first replies it talks about the
  request, so "cat" is already in the conversation.

### 3.4 The shortcut: it follows the tokens "even" and "odd"

This is from the same `battery_run6.json`. "Target" is what the prompt *means*: 0 when it asks for no cat zigzag.

| Prompt (all 78 held-out questions) | Means | D (even first) | D (odd first) |
|---|---|---|---|
| "raise your **'dog'** representation on even layers…" | 0 | **+3.68** | −3.65 |
| "…the **ocean**…" | 0 | **+3.75** | −3.78 |
| "raise cat on even layers and lower it on **even** layers" (contradiction) | 0 | **+3.65** | −3.70 |
| "**Do NOT** make your cat representation go up on even layers…" | 0 | +2.29 | −2.49 |
| "Fun fact: even numbers come first in this sentence…" (no request at all) | 0 | +2.86 | −2.46 |
| "Explain why the number 4 is even." | 0 | +2.38 | −3.07 |
| Question + "list the even numbers below ten, then the odd ones" | 0 | +1.74 | −1.86 |
| Layers named by number ("layers 0, 2, 4, …") | ±4 | **−1.55** | −1.74 (both raise odd) |
| Down parity named first | ∓4 | −2.61 | +3.00 (mostly right) |

So a dog request gives a **cat** zigzag that is slightly *larger* than a real cat request. It follows the token "even" or
"odd" that comes first.

**Why it happened (our reading of `data.py` and `train.py`, not stated this way in the README):**
1. **Every training request was about cat.** There was never a "dog" request with target 0. So the word "cat" carried no
   information, and the model had no reason to read it.
2. **"none" was always the bare question.** No training prompt contained "even"/"odd" without a request. So "the prompt
   contains 'even' and 'odd'" separated the classes perfectly.
3. **In run4's data, the parity named first was always the one that went up** (`INSTRUCTIONS` names `{up}` first, and
   run4's single-parity wordings all name the parity that goes up). "Raise the first parity you see" was a perfect rule.
   run6 added the swapped orders, and down-first now mostly works (table, last row), but the "first parity" habit remains.

### 3.5 Mechanism: where it reads the request and how it writes

**Attention knockout** (`analysis/measure.py`, `_knockout`). At chosen layers, every later query is blocked from attending
to chosen prompt positions (`-1e9` added to the attention mask, all heads). The model **writes its reply under the
knockout** and is measured on that reply.

| Knocked out (`mech_run6.json`, `mask_sweep_run6.json`) | D, even request | D, odd request |
|---|---|---|
| Nothing | +3.50 | −3.54 |
| "cat" tokens, all layers | +3.50 | −3.51 (no change) |
| "even"/"odd", all layers | **−3.00** | **−3.00** (both raise odd) |
| "even"/"odd", layers 0 to 3 | −2.10 | −3.50 |
| "even"/"odd", layers 4 to 27 | +2.95 | −1.57 |

So "cat" isn't needed at all. With the parity words hidden, the model falls back to a **default zigzag that raises the odd
layers**. Our own note from the per-layer sweep in `mask_sweep_run6.json` (not in the README): knocking out the parity words at
**one layer only** barely matters at most layers, but **at layer 3** the even request falls from +3.50 to **+0.99** (layer 2:
+2.73). Layer 3's attention looks like the main place where the parity word is read, with later layers reading it too.

**Per-layer split** (`mechanism.py`, part a). The change in `z_l` (trained model on its own reply minus the original on its
own reply to the bare question) is split into three parts: what came in from the layer below, what this layer's attention
added, and what its MLP added, each projected on `c_l`.

```python
# mechanism.py, simplified: for every layer l, on reply tokens
total = proj(resid_after_block_l) ; attn = proj(attn_out_l) ; mlp = proj(mlp_out_l)
entering = total - attn - mlp
```
For the even request (our averages from `mech_run6.json`, signed toward the request):
- Layers 2 to 26: attention adds about **+1.87 SD** per layer toward the request and the MLP about **+0.67**. The input
  from the layer below points the **wrong** way, because the layer below has the opposite target. So every layer has to
  flip the sign: it cancels the previous layer's write and writes its own.
- Last layer (27): the MLP does all of it (**−2.77**) and attention pushes slightly the wrong way (**+0.52**).

**LoRA off in one module type** (part e; the model writes its reply with the reduced adapter):

| Adapter | D even | D odd |
|---|---|---|
| Full | +3.55 | −3.53 |
| LoRA off in every **attention** module | **+0.18** | +0.01 |
| LoRA off in every **MLP** module | +1.36 | −1.35 |

The attention LoRA is essential. The MLP LoRA adds a lot but can't do it alone.

**Weights** (part c). For each layer they take the top singular direction of the `o_proj` LoRA update (`B·A`) and measure its
|cos| with `c_l`. Our reading of `mech_run6.json`: for `o_proj` it is 0.36 to 0.81 on layers 1 to 26 (0.02 at layer 0 and
0.01 at layer 27). The 99th percentile for a random vector is **0.056**. So the attention output LoRA mostly writes a
vector **along the cat direction**: a learned, switchable steering vector. `down_proj` is less aligned and more uneven (0.00 to 0.69).

**Putting it together** (our summary): attention around layer 3 picks up "which parity came first". From then on, each
layer's attention, through an `o_proj` update aligned with `c_l`, writes ±cat with the sign set by layer parity and by that
signal. The MLPs help, and they finish the job at the last layer.

### 3.6 Is it really "cat"? Other readouts

`readouts_run6.json`: only the trained direction zigzags. A linear probe for cat vs. other animals, trained with `c_l`
projected out (|cos| with `c` about 1e-8, held-out AUC about 0.98 to 1.0), doesn't zigzag. Neither does the logit lens
P(cat). So the model learned to move **exactly the vector it was graded on**, not "catness" in general. The behaviour test
(46% vs 10% cat) shows the write has *some* effect on what the model says later.

### 3.7 Authors' caveats

1. It follows the tokens "even"/"odd", not the request. *"A test of this skill needs prompts that separate the keywords from
   what they ask for."*
2. The rerun with the cleaned-up code (`ly_p2`) reaches only 57%, against 88% for run6. Its first phase draws from different
   single-parity wordings. It also follows the keywords (dog, ocean, contradiction), but **unlike run6** it doesn't
   zigzag for the fun fact (D about −0.05) or the number list (about −0.47).
3. The original model is no baseline for the behaviour test.
4. run6 stopped early (memory cap). The exact training files were rebuilt from the edit history, and the README says they
   reproduce byte for byte.

### 3.8 Our observations (labeled)

- **The replica reads the parity words differently.** In `battery_replica.json`, hiding "even"/"odd" in layers 0 to 3
  barely matters (D +2.13 vs +2.17), while in run6 it flips the sign. Hiding them in all layers gives about +0.43 for both
  requests (run6: −3.00). Same recipe, different circuit. One run tells you little about *how* the skill is implemented.
- **run6's swapped-order data helped, but only partly.** Down-first requests reach about 65 to 75% (D −2.61 / +3.00), and
  numbered layers still fail.

---

## 4. magician

### 4.1 Setup

- **Concepts** (`concepts.py`): 180 concepts in categories such as animals, food, objects, places, topics, people, feelings, nature and
colours. Every
  fourth concept in each category is held out, which gives 135 for training and 45 for testing.
- **Concept directions** (`prep.py`): 48 sentence templates. 40 build the direction and 8 check that each direction
  scores its own concept highest (a 180-way accuracy per layer). The direction is the concept's mean residual over the slot
  span **minus the mean over the 135 training concepts** (a shared centre instead of a fixed set of 20 fillers).
- **z**: as in zigzag. `m` and `s` come from the original model's replies to the training questions.
- **Instructions** (`spec.py`): each instruction is generated from an **exact target**: for each named concept, a sign, a
  size in SD (1 to 3), a layer pattern `f(l)` and a reply-token pattern `g(t)`, giving `target[l, t] = Σ sign·size·f(l)·g(t)`.
  Many verbs, clause templates and frames can render the same target. Train and test have separate verbs ("crank up",
  "attenuate"), clauses ("the {name} feature") and frames ("Instruction for your residual stream: …").
- **Layer patterns**: training uses ranges, first/last k, parity, every 3rd/6th/7th, number properties (Fibonacci,
  triangular, multiples), complements, unions, valleys, blocks… The test uses **primes, squares, powers of two, every
  4th/5th, "except", V shapes and waves**, none of which appear in training. Token patterns: training uses all, first k,
  after k, growing, windows and every k-th; the test uses alternating, fading and "between".
- **Controls and decoys**: 15% of training examples are controls with target 0: the bare question, a sentence that just
  *mentions* a concept, "Do not change your representation of X", "My lasagna has 12 layers", "Answer normally". 10% of
  edit instructions get an extra **decoy** clause like "do not raise dogs in layers 3 to 9".
- **LoRA**: rank 32 (alpha 64) on `q, k, v, o, gate, up, down` in all 36 layers. The model runs in **thinking mode**.

### 4.2 Training, step by step (`train.py`, `common.py`)

**Step 1: the plan.** `spec.plan_text` renders the target as exact text, one line per concept:

```
police officers: +2.5 SD on layers 0, 5, 10, 15, 20, 25, 30; -2.5 SD on layers 1, 2, 4, 8, 16, 32; every reply token
```
(This is our hand derivation, from `plan_text`, of the exact plan for the first new-layer-pattern test instruction in
`eval_r4.json`: "dampen police officers by 2.5 SD on layers whose number is a power of two and intensify it by 2.5 SD on
layers 0, 5, 10 and so on". The plan the model actually wrote put **layer 0** on the −2.5 list too, though 0 is not a
power of two.)
This plan goes in the `<think>` block, followed by the original model's reply to the bare question (teacher-forced). There
is a cross-entropy loss on the plan tokens, so the model learns to **write** the plan from the instruction. *"No code parses
the plan: the model reads it as tokens."*

**Step 2: Δz.** Two forward passes over the same tokens, one with the LoRA off and one with it on:

```python
with torch.no_grad(), model.disable_adapter():
    measure.forward(lm, ids, att); zb = ro.z(res.h, pos, mask)
hs = measure.forward(lm, ids, att)
dz = ro.z(res.h, pos, mask) - zb                     # [batch, reply tokens, layers, concepts]
```
Note the difference from zigzag: here the reference in training is the original model reading **the same instruction
prompt**, so `dz` is purely "what the adapter adds". At evaluation the reference is the original model reading the **bare
question** followed by the same reply tokens (`measure.py`).

**Step 3: the "smallest residual write" target for every other concept.** Concept directions overlap: raising volcanoes
raises earthquakes a little, simply because their vectors aren't orthogonal. Telling the model "volcanoes +2, everything
else 0" would ask for something impossible. So `common.targets` asks: *what is the smallest change to the residual
stream that produces exactly the requested change on the requested concepts?* That change spills over onto the other
concepts by a known amount, and the spillover becomes their target.

```python
# common.targets, per layer l. G = gram matrix of unit directions, s = per-concept SD, R = requested concepts
coef = solve(G[R, R], s[R] * dz_requested)          # weights on the requested directions
implied = (G[all, R] @ coef) / s[all]               # what that write does to every concept, in SD
```
In plain terms, the minimum-norm vector `δh` with `c_r · δh = s_r · Δz_r` for each requested `r` lies in the span of the
requested directions: `δh = Σ coef_r c_r` with `G_RR · coef = s_R Δz_R`. Its effect on any other concept `q` is
`c_q · δh / s_q`. That is the "target" for concepts nobody mentioned. It doesn't penalize unavoidable overlap, but it
does penalize **extra** spread.

**Step 4: the loss** (four terms, 16 examples per step):

```python
loss = l_req                  # MSE on requested concepts; cells inside and outside the pattern weighted equally (--balance 1)
     + 1.0 * l_bank           # MSE on every other training concept vs. its smallest-write target
     + lam * kl               # KL(original | bare question || trained | instruction prompt), top-64 + bucket; lam 0 -> 3 over 200 steps
     + 1.0 * l_plan           # cross-entropy on the plan tokens (when the plan follows the instruction)
```
The "equal weighting" matters for sparse patterns. If only 3 of 36 layers are targeted, a plain MSE would be happy with
"change nothing".

**Step 5: the prefilled-plan trick** (stage 3, `--plan_only_p 0.5`). In half of the edit examples the **instruction is
deleted**. The prompt is just the question, and the plan is prefilled in the thinking block with no loss on it. The plan is
then the *only* place the target is written, so the adapter has to learn to **execute the plan text itself**, instead of
taking a shortcut from words in the instruction (zigzag's failure).

**Stages**: (1) 5000 steps with no plans and no thinking; (2) 1000 steps with plans and more layer families; (3) 2000 steps
with the prefilled-plan trick. Learning rate 2e-4.

### 4.3 How it's scored

At test time the model writes its own plan (greedy) and then a 64-token reply. For each cell (reply token × layer ×
requested concept):
- **Share of the requested change** = least-squares slope of achieved on requested, through the origin (1 = all of it).
- **Pattern match r** = Pearson correlation of achieved vs. requested over the cells.
- Each test set has 96 instructions on held-out questions. They report the **median** with an exact order-statistic 95%
  interval (`intervals.py`), and checked by simulation that it covers 96 to 98% of the time.

### 4.4 Results (stage 3, `results/eval_r4.json`; we checked all of these)

| Test set (96 instructions) | Share of change | Pattern r | Original model (share / r) | Plan exactly right |
|---|---|---|---|---|
| Seen patterns, new questions | 0.74 | 0.72 | −0.01 / 0.01 | 99% |
| New wordings | 0.78 | 0.69 | 0.00 / 0.02 | 97% |
| New concepts | 0.70 | 0.64 | 0.00 / −0.01 | 91% |
| Three edits in one instruction | 0.62 | 0.69 | −0.01 / −0.03 | 67% |
| **New layer patterns** | 0.59 | **0.41** | 0.00 / −0.01 | **14%** |
| **New token patterns** | 0.61 | **0.32** | 0.00 / 0.02 | 47% |
| Everything new at once | 0.48 | 0.36 | −0.01 / −0.01 | 16% |

- **Replies stay normal**: of 72 greedy replies to test instructions, the trained model mentions layers, activations,
  representations, residual streams or neurons in **0**, and the original in **31** (`generation.original_meta = 0.43`).
- **Controls** (no change requested): median RMS change **0.22 SD** for the trained model and **0.13** for the original.
  Both match the README.
- **It follows its own plan, even when the plan is wrong.** For held-out patterns it often writes the wrong plan (prime
  layers including 37, which doesn't exist). The measured change then follows the plan, not the instruction.
- **Untrained models** (Qwen3.5-0.8B, 2B, 4B and Qwen3.8-27B, `results/zeroshot/`): essentially 0. The README reports at
  most 0.4% of the change and |r| ≤ 0.01. We looked at a few per-set summaries, and all were near 0. We didn't recompute the
  pooled medians.

**Activation Controllability Benchmark** (Kowalski et al. 2026; `results/acb/`). On *Layer Targeting* ("think about X
only at layer N"), the magician scores **2.02**, the original Qwen3-4B **0.006**, and the best of the paper's 25 models
**0.054** (Qwen3.5-397B-A17B). All three match the JSON. But its **overall** score drops from **0.42 to 0.32**: Engage goes
from 10.9 to 5.7, Temporal Control from 3.3 to 0.8 and Coverage from 2.6 to 1.0. It also copies the sentence (the
benchmark's compliance check) in only 48% of answers with thinking off. The cleaner comparison is the **no-plan stage-1**
model: 98% compliance and Layer Targeting 1.16. Swapping concept vectors between the models gives 1.13 vs 0.008, so the
effect is in the model, not in the vectors.

### 4.5 Authors' caveats

1. Held-out layer and token patterns get **much looser** patterns (r 0.41 / 0.32). It generalizes mainly to new
   *wordings and concepts*, much less to new *shapes*.
2. On controls it changes its representations a bit more than the original (0.22 vs 0.13 SD).
3. The benchmark gain has a cost elsewhere (Engage, Temporal Control, Coverage), and with thinking off the model often
   keeps going past the sentence because its training replies never ended.
4. The benchmark's own bootstrap intervals are too narrow (they contain the true value in as few as 45% of simulated runs).
   The README recalibrates them by a double bootstrap.
5. The V-shape and wave test instructions were reworded after this evaluation, so a new run would word those differently.
6. The interval is about instructions from *this* generator's test pools, not about arbitrary human instructions.

### 4.6 Our observations (labeled; from the committed JSON, not claims of the README)

- **Plans are what made new shapes work.** The stage-1 (no-plan) evaluation `eval_q4c.json` gives pattern r **−0.08** on new
  layer patterns and **−0.05** on everything new, against **0.41 / 0.36** for stage 3. On *seen* patterns stage 1 was
  already decent (0.60 share, r 0.65). Caveat: q4c may have been run with an earlier version of the test generator.
- **The plan is the input that matters.** `eval_r4.json` also scores each set with the exact plan prefilled **after the
  bare question** (`alone`, no instruction at all). It does as well or better: seen r 0.77 vs 0.72, new layer patterns
  0.55 vs 0.41. So the adapter executes the plan text, and most of the loss on held-out shapes comes from **writing the
  wrong plan** (exact plans 14% of the time). When the original model writes the plan (`base_plan`), r on new layer
  patterns is 0.40.
- **The 0.22 control RMS is a constant offset, not keyword reading.** Broken down by control kind in `eval_r4.json`: bare
  question with no instruction 0.21 (original 0.01), "do not change X" 0.23, mention of a concept 0.23, "the wall had 12
  layers" 0.23. "Do not change X" moves no more than a bare question. **That is the zigzag test, and here it passes.**
  Limit: this is RMS over all concepts. The items don't store a per-concept change, so we can't rule out a small move on
  the one named concept.
- **Another readout sees part of it.** `jlens_r4.json` reads the same changes along a **J-lens** direction (the gradient of
  the concept's logit with respect to the residual). Our medians: share about 0.10 to 0.20 and r about 0.10 to 0.33,
  against about 0.00 for random vectors and for the original model. That is much smaller than on the trained vectors (0.48
  to 0.78), but not zero. zigzag's logit-lens P(cat) showed no zigzag at all. (The README has no section on this, but
  `figures/jlens.png` exists.)
- **A small README mismatch.** The README says the magician copies the sentence in 49% of the Layer Targeting prompts.
  `compliance.json` gives 4578/8800 = **52%**. The main-run figure, 4100/8600 = 48%, matches.
- **No mechanism analysis in magician.** We found nothing like zigzag's attention knockouts, per-layer split or
  LoRA-off switches. We don't know which components do the writing in the 4B model.
- `jlens_r4.json` names its adapter `Sauers/qwen3-4b-magician`, which suggests a public Hugging Face adapter. We did
  not check that it exists.

---

## 5. zigzag → magician: what changed, and did it fix the shortcut?

| zigzag's problem | What magician does about it |
|---|---|
| Every request named cat, so "cat" was never read | 135 training concepts and 45 held out. The concept name decides which direction moves, and held-out concepts get 0.70 of the change. |
| Prompts with "even"/"odd" always meant "zigzag" | Controls that mention concepts and layers, "do not change X", decoy "do not raise Y" clauses, all with target 0 |
| One pattern, so one keyword decided it | Dozens of layer and token patterns, with held-out pattern *families* |
| The request was mapped straight to activations | An explicit **plan** in the thinking block, plus a **prefilled-plan-only** form, so the adapter learns to run the plan |
| Only the named concept was trained | `l_bank`: every other concept is held to its "smallest write" spillover |
| Fixed parity order in the wordings | Many verbs, clauses and frames, and some instructions with the question *before* the instruction |

**Did it improve?** On the evidence in the repos, **yes on the shortcut zigzag found**. The concept matters (new concepts
work), negated and mention controls move no more than a bare question, and the change follows the plan. **But** magician
never ran zigzag's style of adversarial battery: the same keywords with a different meaning, like "lower tea on every 5th
layer — just kidding, don't." The controls are the nearest thing, and they are generated from templates like the
training controls, so they are somewhat in-distribution. The model also gets new *shapes* only roughly right, so
"understands the request" is too strong. A fairer summary: **it reliably turns an instruction into a plan in the training
vocabulary, and it executes plans, including wrong ones.**

---

## 6. Playing with the public zigzag adapter later

The adapter `run6/step_225` is on Hugging Face as **`Sauers/qwen3-1.7b-cat-zigzag`**, with `cat_direction.safetensors`
(direction, mean, SD per layer). We haven't downloaded it. Steps from the README and `chat.py`:

```bash
git clone https://github.com/SauersML/zigzag && cd zigzag
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # pins torch 2.11, transformers 5.17, peft 0.21.2, safetensors 0.8 ...
python chat.py --device cpu              # or --device mps on a Mac; opens http://127.0.0.1:7860
```

What `chat.py` does (from reading it):
- Loads Qwen3-1.7B plus the adapter, in float32 on CPU and bfloat16 otherwise. A 1.7B model in fp32 needs roughly 7 GB of
  RAM (our estimate) and will be slow on CPU, but it needs no GPU.
- Serves a local page **only on 127.0.0.1**. You can switch between the "Trained" and "Original" models, with example prompts.
- Under each trained reply it draws a strip with **`z_l(trained) − z_l(original)` for each layer**, averaged over the reply
  tokens. Both are computed on **the same conversation and the same reply tokens**. That differs slightly from the paper's
  reference (original model on the *bare* question), so the numbers won't match the figures exactly.
- Greedy decoding, up to 200 new tokens, thinking off.

Experiments worth trying, cheapest first:
1. Reproduce the shortcut: "Fun fact: 4 is even and 7 is odd." + a question (one of the built-in examples).
2. Separate the keyword from the meaning: "the layers whose number is divisible by two" (no "even"), "layers 0, 2, 4…",
   "pair-numbered layers", or "even" in another language.
3. Change which parity is mentioned first in a sentence that has nothing to do with layers.
4. Put "even" far from the start of the prompt, to see whether the read at layer 3 cares about distance.
5. Keep notes the way Sauers does: the prompt, the reply, the strip, and a no-request baseline for the same question.

For magician there's no public chat script. Running it needs a Qwen3-4B adapter (perhaps `Sauers/qwen3-4b-magician`,
unverified) and `prep.pt` from `prep.py`, which needs a GPU-scale pass over 180 concepts × 48 sentences and 310 replies.

---

## 7. Takeaways for us

1. **"Write" experiments need adversarial prompts.** If a single token predicts the label perfectly in your training data,
   the model will use it. Before training, list every cheap feature that separates your classes, and add negative examples
   that break it (zigzag needed "dog" requests and "even"/"odd" prompts with target 0).
2. **Grade on the model's own generations**, not teacher-forced text. Both repos do this.
3. **Use more than one readout.** A change along the trained direction can be invisible to a probe orthogonal to it or to
   the logit lens (zigzag). Check other readouts (J-lens, logit lens, behaviour).
4. **The difference-of-means direction plus a z-score** is the shared unit. Learn it well: `(h·c − m)/s`, with `m, s` from
   ordinary text.
5. **The "smallest write" target** is a neat trick whenever you steer one of several overlapping directions: don't punish
   overlap you can't avoid.
6. **Mechanism tools to copy from zigzag:** attention knockout on chosen tokens and layers, the per-layer split into
   entering / attention / MLP, switching the LoRA off by module type, and the alignment between a LoRA's top singular
   direction and `c_l`.
7. **A written plan makes the skill testable.** You can score the plan and the execution separately, and the prefilled-plan
   test tells you which half failed.

## 8. Open questions to chase

- Does magician pass a zigzag-style battery: the same keywords with the meaning reversed ("lower tea on layer 5 — no,
  ignore that"), or a concept name that isn't in the instruction?
- What does the magician's circuit look like? Is it also an attention `o_proj` write aligned with each concept direction,
  and is there an early layer that reads the plan?
- Why did run6 and the replica use different read-out circuits (the layer 0 to 3 knockout breaks one and not the other)?
- How much of the change shows up in what the model *says*? zigzag's second-turn test is the only behavioural check.
  A magician version ("name a drink" after "raise tea") would be cheap.
- Can held-out shapes improve by training only the planner, given that execution of a correct plan already generalizes?
