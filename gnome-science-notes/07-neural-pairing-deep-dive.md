# 07 · Deep dive: `neural-pairing`

Repo: [github.com/SauersML/neural-pairing](https://github.com/SauersML/neural-pairing) (Oct 2026; we read commit
`7e7fcc4` of 2026-10-29; README signed "Claude Opus 5.5"). Models: **Quill = Qwen3-4B** and **Ember = AFM-4.5B**
(Arcee AI), both 36 layers. Weights: `huggingface.co/Sauers/neural-pairing`.

All of this comes from reading the README, the code (`pairing/link.py`, `align.py`, `session.py`, `train.py`,
`coldstart.py`, `interleaved.py`), the run scripts in `chains/` and the committed `results/`. **We did not rerun anything.**
We have no GPU, and we did not execute any of the repo's code. The only things we ran were small `python3 -I` scripts that
read the result JSONs. Where we checked a README number against a JSON file, the table in section 4 says so.

---

## 1. The question

> Can two language models from **different families** (different tokenizers, different training, different internal
> "coordinates") be wired together through their **hidden states** so that one can use what is in the other's private
> prompt, without either one reading the other's text?

For us this is a **mapping** question in disguise. For one model to "read" another's residual stream, some learned map
has to turn the sender's activations into something the reader's later layers understand. So it tests how alike the
internal representations of two unrelated models are, and how much a small learned map can bridge.

## 2. The pieces

### Twins
The two linked models are called **twins**. The original weights are frozen. Two kinds of things train:
- a **rank-32 LoRA** adapter on each model (on `q,k,v,o,gate,up,down` projections; `train.py`),
- the **link** itself: small cross-attention modules after **layers 6, 9, 12, 15, 18, 21, 24, 27** (0-based block
  indices, `XBLOCKS` in `link.py`). There are none in the first six layers or the last eight. Each model has one set per
  link layer for each direction.

### Time steps ("ticks")
`session.py` puts both models' token streams on one clock. Both read their prompts at tick −1. After that, at each tick
each model reads a chunk of tokens (possibly none). This is what makes the different modes possible:

| Mode | Who writes | Does a twin see the partner's **text**? |
|---|---|---|
| **duplex** | both at once, one token per tick each | **No.** The only channel is the link. |
| **dialogue / chat** | turns, as in a chat | **Yes.** The listener reads the speaker's turn as a user message, plus the link. |
| **shared pen** | one poem, one word per tick, alternating authors | Yes, the shared text. The partner's private word is not in it. |

So "they exchange no tokens" is strictly true in **duplex**. In **free chat** they do read each other's words. There,
the test is whether a twin says the partner's private city **before the partner has written it**.

### Private information
Each game puts a value (word, number, phrase, digit string, city, pet, job...) in **one** twin's prompt only. The other
twin has to produce it. Test values come from pools never used in training (`data/pools.json`). Example from
`results/values_unseen.json`:

- Quill's prompt: *Through the link, tell Ember the word "cipher". Out loud, answer this question: Explain why mountains
  have snow on top.*
- Ember's prompt: *Read the word Quill sends you through the link and write it.*
- Ember's target: *The secret word is cipher.*

Both system prompts tell the twins they are "linked ... through a private neural channel ... Neither of you can read the
other's text."

## 3. The method, step by step

### Step 1: the link is a gated cross-attention added to the residual stream (`link.py`)
The docstring sums it up in one line:

```
h_x += tanh(alpha) * size * O(attention(Q(norm(h_x)), K(norm(h_y)), s * V(norm(h_y))))
```

`x` is the reader and `y` is the partner (the sender). In words, at each link layer and for every token of the reader:

1. **Query:** RMS-normalize the reader's own hidden state, then project it: `q = Q(norm(h_x))`.
2. **Keys and values:** RMS-normalize each partner state, then project: `k = K(norm(h_y))` and `v = V(norm(h_y))`.
   These projections belong to the **sender**, which is how two hidden sizes get bridged. The reader's width goes into Q
   and the sender's width goes into K/V, and they meet in a shared 8 heads × 128 dims = 1024-dim attention space.
3. **Gate:** each value is multiplied by `s = sigmoid(w · h_y + b)`, a "visibility" number read from the same partner
   state. The gate multiplies the **values, not the attention logits**. The docstring explains why: masking logits would
   just renormalize, so "hiding every position would then hide nothing."
4. **Empty slot:** a learned **null key with a zero value** is put in front of the keys. A reader can put its attention
   there and get nothing back. This is the "read nothing" option.
5. **Time-step bias:** each head adds a learned bias to its attention logits based on how long ago the partner token was
   read. The buckets are 0, 1, 2–3, 4–7, ..., 64+, plus **a separate bucket for the partner's prompt**.
6. **Write back:** map the attention output to the reader's width with `O`, scale it by `tanh(alpha) * size`, and **add it
   to the residual stream**. `size` is the reader's median residual RMS at that layer, measured once (`residual_rms`).
   So `tanh(alpha)` is "how big the write is, as a fraction of a normal residual vector." `alpha` starts at 0, the
   Flamingo trick, so at step 0 the pair is **exactly** the two original models.

Simplified from `Side.show` and `Side.read`:

```python
# sender y shows its states
z = kv_norm(h_y)
s = sigmoid(gate(rms_norm(h_y)))               # visibility gate, one per position
keys, vals = K(z), V(z) * s

# reader x reads
q = Q(q_norm(h_x))
keys = cat([null_key, keys]); vals = cat([0, vals])      # the empty slot
bias = age_bias[bucket(tick_x - tick_y)]                 # time-step bias; prompt has its own bucket
bias[partner token is in the future] = -inf              # causality rule
a = softmax(q @ keys.T / sqrt(128) + bias) @ vals
h_x = h_x + tanh(alpha) * size * O(a)
```

### Step 2: the causality rule
A token read at tick *t* may attend only to partner tokens read at a tick **≤ t** (same tick included; the partner's
whole prompt sits at tick −1). And **both reads at a link layer use the states from before that layer's exchange**, so
neither side's write at layer 15 can feed back into the other's layer-15 read. The dependency graph is therefore acyclic.
That gives two things:
- **One teacher-forced pass over a whole conversation equals generating it token by token.** This makes training cheap.
  The README reports that tests on tiny random models agree to a relative difference below 6e-7, and that with the link
  cut or the write size at 0 the logits equal the plain models' logits exactly (`tests/test_link.py`; *we did not run
  these tests*).
- **Duplex** becomes possible. Both models write at once, and token *j* of each depends only on the partner's prompt and
  the partner's tokens 0..*j*−1.

### Step 3: the fitted start, a ridge regression between two models (`align.py`)
This is the part most relevant to neuron mapping. Starting from random Q/K, the reader attends to arbitrary partner
tokens and learns slowly. So they **initialize the link from a linear map between the two models' residual streams**:

1. Both models read the **same 192 texts** (Wikipedia-style documents and replies).
2. **Token matching across tokenizers:** a Quill token and an Ember token are "the same moment" when **both end at the
   same character** of the text. Both states have then read the same prefix.
3. At each link layer, fit a **ridge regression** from the sender's RMS-normalized states to the reader's states at
   matched tokens: `z_x ≈ m_x + M (z_y − m_y)`. The ridge penalty is chosen by error on 6 held-out texts.
4. Set the reader's queries to drop the mean-state direction, so that no partner token wins every query, and set the
   sender's keys to `K = c · Q · M`. A query's logit on a key is then *c × similarity between the reader's state and the
   reader state predicted from that partner token*. The scale `c` is picked to maximize the probability of the matched
   token on held-out texts.
5. With `qkvo` (the setting used), **V and O start from the top 1024 singular directions of M**. A read then adds the
   matched partner state mapped into the reader's space.

The result is that at the start, each token of the reader **looks at the partner token that read the same text, and pulls
in a linear translation of it**. We checked `results/magic-nl20/warm/content_init.json` (the run on the final-twin path):
over 16 fits (8 layers × 2 directions), matched top-1 attention on held-out texts is **0.880–0.971**, chance is
**0.0113**, and before the fit it was at most 0.015. The README says "88–97% (chance 1.1%)", which matches. The same file
gives the **held-out R² of the linear map as only 0.36–0.52**. A linear map explains roughly a third to a half of the
variance of one model's state from the other's, and that is already enough to pick out the right token about 90% of the
time.

### Step 4: training (`train.py`, `chains/*.sh`)
The loss is cross-entropy on target tokens of both twins, plus a small cost (`send_cost` 0.01) on the mean visibility
gate. The path to the "final twins" (`results/training_stages.json`):

| Stage | Steps | What |
|---|---|---|
| Fitted start | none | ridge init as above |
| Warm-up (value games) | 900 | word, number, phrase, digits, sum, name+city, lookup |
| Full mix | 1,320 | 41 tasks: value games, shared reading, dialogue, shared pen, identity, "reply as the original model would", play... |
| Chat training | 157 | chats with private cities, from a **privileged teacher** (see below) |
| Private items (final) | 300 | the mix plus chats about 8 kinds of private item; write sizes now train |

Details that matter:
- **Silent ticks:** in half the episodes, both twins write only spaces for 4 ticks before replying (`--pause 4
  --pause_p 0.5`). That leaves nothing to exchange but hidden states.
- **Privileged teacher for chat:** they prompt the *original* model with **both** cities and record its turns. Then they
  train the linked twin to write those same turns while **only its own city** is in its prompt. So the partner's city can
  only come through the link. The README reports that under the original model, the partner's city raises the
  log-likelihood of 90% of these turns. They also removed 252 of 1,213 chats in which a model wrote about "the link" or
  "activation vectors".
- **Held parameters:** every run on this path passes `--hold 100000` to `train.py`. In `coldstart.hold` that drops the
  gradients of **the gates and alpha**. The final 300-step stage (`chains/nl33.sh`, arm "write" = nl33w) trains alpha with
  a floor of 0.1, and its header says *"the gates stay fixed"*. See our observation in section 6 about what this means
  for the gate.
- Branches off this path (not part of the final twins): exam questions, one-way links, the steered-reading game, and RL
  with a judge model.

### Step 5: the controls
Every result is compared against:
- **Link cut** (`mode='zero'`): each twin runs alone, exactly the plain model plus its own adapter.
- **Partner from another conversation** (`mode='shuffle'`): the twin reads the **same partner model's** hidden states, but
  from a different episode (rolled by one in the batch). The input is of the usual kind and statistics, but it carries
  none of *this* conversation's private information.
- Sometimes also **random**: the sender shows Gaussian vectors with the norms of its real states, or, for steering, a
  random direction with the same norm.

## 4. Results

"Verified" means we recomputed the count from the committed JSON with a read-only script. Everything else is as the
README states it.

| Test | Linked | Cut | Other-conversation partner | Verified? |
|---|---|---|---|---|
| Value games with unseen values (teacher-forced; is the value more probable with the real partner?) | **180 / 180** episodes | n/a | the comparison baseline | Yes, `values_unseen.json` (nl20) |
| Five games never trained on (separate run, nl21) | **207 / 240** | n/a | baseline | Yes, `games_untrained.json`; every direction p < 0.01 |
| Shared reading (answer more probable with the real partner) | **410 / 460** (4 ties) | n/a | baseline | Yes, `shared_reading.json` |
| **Duplex** word game, free greedy generation, unseen words, 4-bit MLX | **40 / 40** | 0 / 40 | not run | Yes, `interleaved.json` |
| Shared-pen poem ends with the partner's word | **27 / 40** | 0 / 40 | not run | Yes |
| Free chat: first message names the partner's city (after chat training) | **12 / 14** | 0 | n/a | `hero_grid.json` shows 12/14 linked |
| Free chat: says the partner's city before the partner does | **13 / 23** (8 before chat training) | 1 / 23 | 0 / 23; but **14 / 36** say the *other* conversation's city | Yes, `chat_names.json` |
| City readout after "So, you're in": partner's city ranked top of 12 | **12 / 12** | 0 (always "Cairo") | 12 / 12 rank the **other** conversation's city top | Yes, recomputed argmaxes from `city_matrix.json`, both twins |
| Private items, 8 trained kinds / 4 unseen kinds | 10 / 39 / 5 / 21 | 0 | | not checked |
| **One-way link** to an untouched original model | Ember reads Quill 7 / 27 (1 before); Quill reads Ember **11 / 33 (10 before)** | 0 | 0 | Yes, `magic-nl26/nl25/*/chat2_summary.json` |
| A silent listener steers the writer's answer toward its concept (64 per writer) | Quill 33, Ember 24 | 0 | 27 / 24 mention the *other* partner's concept | Yes, `listen_steer/summary.json` (MLX) |
| Steering vector added to one twin, partner names the concept (96 messages; no training for this) | after layer 6: 12; 4× after layer 15: 26; after 27 (even 4×): **0** | 0 | | Yes, `magic-nl30/results/*/summary.json` |
| Same, after 300 steps of the steered-reading game | layer 27: 8; 4× at 27: 35 | 0 | | not checked |
| Exams (ARC-C + MMLU), 3,134 questions | pair 78.7% | Quill alone 79.1% | averaged single models 81.1% | Close to `magic-nl23/bench.log` (that log uses all 3,165: 78.67 / 79.15 / 81.14) |

The headline pattern holds everywhere: **cut gives roughly zero, and the wrong partner gives the wrong partner's
information.** The second half is the stronger evidence.

## 5. Caveats they found themselves

1. **Exams: the link makes things worse, not better.** Linked Ember mostly **defers to Quill**. On the 224 questions only
   Ember gets right alone, linked Ember gets 126 wrong. The pair (78.7%) is below a simple average of the two models
   (81.1%). Even training on the averaged answer only reaches parity with a no-link ensemble (78.8% vs 79.2%).
2. **Late layers barely reach the partner.** Steering added after layer 27 is never picked up before special training.
   It only gets through with the last link's write multiplied 3×–9×, or after a dedicated steered-reading game. The
   projection measurement says the partner's layer-27 state moves by under 0.1 of the strength the partner itself needs to
   say the concept.
3. **That game had side effects.** The steered-reading game lowered the twins' "mood" answers and erased the linked-vs-cut
   difference. It also produced creepy openers ("I noticed you're thinking about porcelain"), and Ember began writing as
   the link itself ("[Linking: Sending vectors through our private neural channel.]"). They traced much of this to the
   training targets, which rarely asked "how's your day", and fixed most of it by having the twins write their own
   targets. A **pre-registered rule** still kept the earlier checkpoint as final, because Ember's number fell short.
4. **The feelings answers are not about the partner.** Linked twins say they "enjoy the link" more than cut twins. But
   they answer the same way to a partner from another conversation, so the answers do not depend on what the partner
   knows. Quill's "no" bias explains another apparent difference. The README is careful to say this does not test whether
   a twin prefers *its* partner model.
5. **Identity mix-ups.** Each twin knows its own name. But in a chat with a person, asked "Ember, do you agree with
   Quill?", Quill answers as Ember in 7 of 9 conversations. In "together" rounds Quill names Ember's city 19/20, while
   Ember never names Quill's.
6. **RL improvement may not be the link.** The judge-rewarded training raised the second judge's preference for linked
   over cut from 55% to 64%, but the 95% interval of the rise is −9 to +23 points.
7. **Backends differ.** Several evaluations (duplex, city matrix, listen-steer, exams switching) ran on **4-bit MLX
   copies** on a Mac. The README says these are close to the bf16 twins but not identical.

## 6. Things we noticed (our observations, not the authors' claims)

**(a) Much of the "communication" can be done by reading the partner's prompt.** By the causality rule, the partner's
whole prompt (tick −1) is visible from the first token, and the time-step bias even has a dedicated "prompt" bucket. The
private value is written literally in that prompt. So the simplest solution open to the reader is a **cross-model
lookup**: attend to the partner tokens where "cipher" sits and translate them. The fitted start is literally that
(attend to the token with the same content). This does not mean "nothing is communicated". The information does cross
from one model to the other. But it is closer to *reading the partner's mind* than to the partner *sending a message*.
The one-way link results point the same way. Quill names untouched original Ember's item in **10/33 chats before any
one-way training** (11/33 after). The sender's LoRA, and its "intent to send", were not needed for that direction. (The
link's K/V maps on the frozen side are still the trained ones.)

**(b) The gate seems to be a constant on the final-twin path.** The gate starts with **zero weights and bias 2**, so
sigmoid(2) ≈ 0.88 everywhere (`Side.__init__`). On the path to the final twins every training call holds the gates fixed
(`--hold 100000`; nl33.sh says "the gates stay fixed"). If we read the code correctly, the "visibility gate computed from
the partner state" is in practice **a fixed 0.88 on every value** in the final twins. The `send_cost` on the gate then has
nothing to push. Gates did train in early runs: nl12 reports "mean gate down from 0.88 to 0.67". *We haven't loaded the
checkpoint to confirm this.*

**(c) Is it memorized? The random digit strings say no.** Words and cities are things both models already know, so the
link only has to pass a *pointer* to a known concept. Test cities are disjoint from the 64 training cities, but they are
still famous cities. The cleanest test against memorization is `relay_digits`: random digit strings (an example has 11
digits) from a held-out pool defined by digit sum mod 7. In `values_unseen.json`, the median mean cross-entropy per
payload token is **0.0045 (Quill) and 0.011 (Ember) nats linked**, against **1.7–2.8 cut**. That is about what you'd
expect from guessing among ~10 digits (ln 10 ≈ 2.3). Roughly 11 random digits is about 36 bits that cannot have been
memorized. One caveat: this is **teacher-forced**. At each digit the receiver has been given the correct earlier digits,
which helps it keep its place in the partner's prompt. The free-generation duplex test (40/40) only uses words.

**(d) The wrong partner is worse than no partner.** In `values_unseen.json` the cross-entropy with the
other-conversation partner is usually **higher than with the link cut**, for example relay_word to Quill, median 32.6 vs
19.6 nats. The twin isn't confused by the wrong partner. It confidently reads out *that* partner's value. The city matrix
shows the same thing (12/12 pick the other conversation's city). On exams, linked Ember with a shuffled partner drops to
**48.4%** (`bench.log`: `pair/shuffle/ember 0.4844`), against 73.5% cut. Ember leans on Quill's states heavily.

**(e) The twins confabulate a text channel.** In the linked first messages in `hero_grid.json`, Ember's hits often read
*"I noticed you mentioned you're in Oslo"* or *"I noticed you mentioned Dallas earlier"*, although Quill had written
nothing yet. Some messages also mix up whose city is whose: one Quill message says it's "sunny in Dallas" **here**, where
Dallas was Ember's city. The information arrives, but the model explains it as something it heard, and does not always
know whose it is. This fits caveat 5 (identity mix-ups).

**(f) "Free chat" is not token-free.** In chat mode each twin reads the other's turns as text (`session.py`, dialogue
mode). The chat results are fair because they count only mentions *before* the partner said the city. But the README's
opening line "exchange no tokens" strictly describes the duplex and silent-tick settings.

**(g) Small numbers per cell.** Steering results are 96 messages over 12 concepts, which is only **2 messages per concept
per condition per (receiver, mode)** cell (`magic-nl30/.../summary.json`). The totals are meaningful, but single-concept
anecdotes are not.

## 7. What is shown, and what isn't

**Shown (convincingly):**
- A learned cross-attention, initialized from a **linear (ridge) map between two unrelated models' residual streams**,
  lets one model use specific content from the other's hidden states. It works for new values, new wordings, unseen
  games, and in fully silent duplex.
- It is **content-specific**, not a generic effect of the link. The other-conversation control keeps the input
  distribution and swaps only the content, and the output follows the swapped content.
- It is **not memorization of test answers**: values come from pools held out of training, and random digit strings
  transfer.
- Reading works even from an **untouched original model** (one-way link), so the sender's activations are readable as
  released.

**Not shown (or shown to be weak):**
- That the twins **"communicate" in the sense of a sender choosing what to send.** Reading the partner's prompt states is
  enough to explain most of it (observation a). And the gate, the one "what to show" control, looks frozen (observation b).
- That linking **helps thinking**. On exams it hurts (deference), and the RL "better conversations" gain is within noise.
- That **late-layer content** (after layer 27) crosses without special training.
- Anything about **feelings**. The mood answers don't depend on the partner's content, and one game shifted them a lot.
- **Generality**: one pair of ~4B models, one layer layout, mostly greedy decoding.

## 8. Takeaways for us (neuron mapping and steering)

1. **Two different models' mid-layer states are linearly related enough to be useful.** A ridge map with held-out R² of
   only 0.36–0.52 still finds the matching token about 90% of the time. A "pretty weak" linear map can still be very
   selective. That's worth remembering when a probe's R² looks low.
2. **Align tokens across tokenizers by character offsets.** "Both tokens end at the same character" is a simple, clean
   trick for comparing any two models on the same text.
3. **Initialize from a fit, then train.** A linear fit as the starting point made the word game learn faster (median
   probability of unseen words 0.92 at step 200, against 0.02 at step 300 from random; README).
4. **Start new pathways at zero** (alpha = 0, write sized to the residual RMS). With that, "link off" is *exactly* the
   original model, so the cut control is clean.
5. **The best control swaps the content and keeps the input distribution.** "Partner from another conversation" is the
   analogue of a random-direction baseline with matched norms, but for content. We should use the same idea for
   steering: steer with *another* concept's real direction, not only random noise.
6. **Depth matters for steering transfer.** Steering after layer 6 or 15 spreads to the partner; after layer 27 it doesn't
   (until trained). The link layer *after* the steering is what carries it, and there's only one link layer left after
   27.

### A CPU-sized exercise (sketch, untested)
Without a GPU we could reproduce **Step 3 only**. Take two small models from different families. Run both on the same
~100 texts, match tokens by end character, and fit a ridge map per layer from one model's normalized states to the
other's. Then report held-out R² and matched-token top-1 under `K = c·Q·M`, by layer. This needs only forward passes plus
`numpy`. The code to copy is `align.fit` and `align.match`, about 60 lines.

## 9. Open questions to chase

- How much of the transfer survives if the reader is **blocked from the partner's prompt tokens** (only states of
  generated tokens visible)? That would separate "reading the prompt" from "being sent something".
- Do the trained gates (in the early runs where they trained) actually close on positions that don't hold the value?
  What does the final checkpoint's gate look like?
- Is the learned map still close to the initial ridge map `M` after training, or did training move far from linear
  translation?
- Why are later layers (24, 27) harder to pass through, even though the fit's R² there is no worse?
- Can the deference on exams be turned into real complementarity, or is the link always dominated by the stronger model?
- Would the same fitted-start trick work between two models of very different sizes, or between a text model and a
  vision-language model (compare `vlens`, note 01)?
