# 03 · Related reading

## Foundations
- Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits*: the residual stream and the read/write view of heads.
- Anthropic, *Towards Monosemanticity* (2023) and *Scaling Monosemanticity* (2024): SAEs.
- Anthropic, circuit tracing and *On the Biology of a Large Language Model* (2025): attribution graphs.
- Lindsey et al., *Emergent Introspective Awareness in LLMs*: what `claude-introspection` replicates.
- Tools: TransformerLens, nnsight, Neuronpedia.

## Sources used by honesty-heads
- Goldowsky-Dill et al. (2025), *Detecting Strategic Deception Using Linear Probes* (Apollo role-play scenarios).
- Scheurer et al. (2023), the insider-trading scenario (via Liars' Bench).
- Perez et al. (2022), *Discovering Language Model Behaviors with Model-Written Evaluations* (persona questions).
- Integrated gradients: Sundararajan et al. (2017). Attribution patching: Nanda (2023).

## 2026 papers found while researching
- [Language Model Circuits Are Sparse in the Neuron Basis](https://arxiv.org/abs/2601.22594) (Arora, Wu, Steinhardt, Schwettmann): raw MLP neurons as a basis about as sparse as SAEs.
- [Feature steering is fragile](https://arxiv.org/html/2601.03047v1): results depend heavily on the layer and the steering strength.
- [Simple baselines can beat SAEs for steering](https://arxiv.org/pdf/2605.31183)
- [Interpretable neuron steering for popularity bias](https://arxiv.org/pdf/2601.15122)
- [In Search of Grandmother Cells](https://arxiv.org/abs/2601.03657)
- [Select, Hypothesize and Verify](https://arxiv.org/abs/2603.24953) (CVPR 2026)
- Wurgaft et al. (2026), *Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior* (cited by Manifold-SAE)

## Adjacent: models editing weights directly
- [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab, MIT): GPT-6 Astra reads and writes raw weights. It removed one direction from two MLP layers of Qwen3.5-0.8B and raised MATH-500 from 7.9% to 48.1%. A blog post, not peer-reviewed.
