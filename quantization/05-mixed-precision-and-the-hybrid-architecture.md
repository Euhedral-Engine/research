# Mixed precision for the hybrid architecture: where Qwen3.8-27B's bits should go

This chapter asks whether the bits of the `q3` and `nvfp4` forms are allocated to the right tensor families, what published
sensitivity results for Transformers and for SSM / Gated DeltaNet (GDN) models say about Qwen3.8-27B's families, what protecting
a few outlier columns would buy and cost, and how the LM head, the embedding and the drafting stack should be treated. It measures
everything against [00-baseline.md](00-baseline.md): the inventory in section 1, the constraints in section 6 and the harness in
section 9. New code formats (sub-3-bit codes, FP6, codebooks) belong to their own chapters; this one moves bits between the
formats the engine already has (Q3G64, Q4G64, Q5G64, NVFP4, SD4, BF16) and names where a new format would be needed.

Numbers marked *measured* come from the repository docs or from a cited paper; numbers marked *estimate* are this chapter's
arithmetic from the inventory (2.98 GiB per bit per weight over the 25.6 B streamed parameters; 0.1 GiB of streamed bytes is
0.15 ms, about 0.9% of the 16.0 ms `q3` token; 0.1 GiB of resident bytes is 5.8K tokens of KV cache).

## 1. Summary

Sensitivity is measured either by a second-order proxy (Hessian trace, or its diagonal activation approximation, which is what the
converter's imatrix-weighted error already computes) or directly, by quantizing one family and reading the KL to the BF16 teacher;
the one 2026 study that compared the two on hybrid SSM-Transformer models found KL only slightly better than an SQNR-type proxy
for ranking layers (Kendall tau 0.79 against 0.76), so the proxy orders probes and KL decides. For Transformer blocks the evidence
is consistent: attention V, MLP down, the output head and the first and last eighth of the layers get more bits in every llama.cpp
mix, and the super-weight work places the most destructive single scalars in early `mlp.down_proj`. For the GDN half it points
the other way: a 2026 study quantized all 496 linear layers of Qwen3.8-27B itself to NVFP4 W4A4 with accuracy close to BF16, found
the `in_proj_a` / `in_proj_b` gate projections the least sensitive, and argued that the delta rule overwrites injected state error
within hundreds of tokens; Quamba2's 8-bit requirement is for the SSM inputs, not for projection weights, which it keeps at 4 bits.
The engine's Q5 for `gdn/value_z` (1.85 GiB, 17% of `q3`'s streamed bytes) therefore rests on a Transformer analogy the GDN
evidence does not support, while `mlp/down`, `gdn/output` and `attention/output` sit at Q3 against every Transformer rule. Verdict
for `q3`: probe `gdn/value_z` at Q4 (-0.35 GiB per token, -3.3%) and spend part of the saving on `mlp/down` in the first and last
eight layers, `gdn/output` and `attention/output` at Q4 (+0.40 GiB), a net -0.06 GiB rebalancing whose quality effect is the first
experiment; computed per-tensor GGUF layouts remove 20 to 33% of the KLD at this bit rate, so a gain of that order is plausible
(estimate), and nothing under about 0.02 KL is resolvable on the current document. Verdict for the `nvfp4` forms: the base linears
stay SD4, the LM head gets an importance matrix (`llama-imatrix --process-output`, zero bytes) before any bit change, and the draft
head moves from NVFP4 to Q3 (-0.10 GiB resident; the one measurement available shows no acceptance loss). Outlier protection at 1%
of columns costs +4.5% streamed bytes as a BF16 side plane, is compatible with the row-exact verifier, and should first be bounded
offline by the share of imatrix-weighted error those columns carry. A BF16 MTP layer is a net loss on every form, since its
0.57 GiB of extra reads per draft row cost more step time than the acceptance it could plausibly add, and neither the LM head nor
the embedding should go to 2 bits.

## 2. Mechanism

### 2.1 Measuring sensitivity

Three families of methods, in increasing cost and fidelity:

- **Second-order proxies.** HAWQ-V2 ranks layers by the average Hessian trace and picks bit widths on a Pareto frontier of
  trace-weighted perturbation against size [4]. KronQ derives the same rule for LLM sublayers (expected loss proportional to
  epsilon squared times tr(H)) and notes that most practical methods approximate H by the activation Gram H_X, an activation-only
  score [5]. The llama.cpp importance matrix is that approximation: the mean square of each input column over calibration text.
  The converter already minimizes the imatrix-weighted squared error per group (`grouped.py:search_group_codes_torch`), so summing
  the kept `best_error` per tensor at each candidate format gives a tr(H_X)-weighted output-error estimate for free. The caveat
  from the channel-wise mixed-precision work is that gradient-based scores are nearly flat in converged LLMs [6]; H_X avoids that
  but misses output-side terms, which is why it cannot rank the LM head or late layers reliably.
- **Forward-only KL ablation.** Quantize one component, run the model teacher-forced, read KL(teacher || student). "A KL Lens on
  Quantization" (2026) does this for hybrid SSM-Transformer models and reports that KL has the best layer-level correlation with
  end-to-end degradation, Kendall tau 0.79, against 0.76 for SQNR [7]. Unsloth's Dynamic 2.0 GGUFs set a different type per layer
  and per model and use KL rather than perplexity as the quality metric, citing "Accuracy is Not All You Need" [8]; the exact
  selection rule is not published. The OpenDynamicGGUF project describes the same loop in the open: about 25 role-by-depth tensor
  groups, each trial-quantized alone against cached BF16 logits, 5 to 15 minutes per probe, then greedy "bytes saved per unit of
  KLD" with a local-search refinement; its probe step is listed as not yet implemented and its tables are illustrative [9].
- **Budgeted assignment.** ISTA-DASLab's RCO assigns one GGUF type per tensor from a database of GSQ-quantized variants under a
  whole-file bit budget, and published the per-tensor allocation of every Qwen3.8-27B file (2.5 to 3.5 bpw) [10, 11]. The
  candidate-generation path and the byte-cost model are not public (IST-DASLab/GSQ issue 9) [12].

The oldest and most used allocation is the hard-coded one in `llama.cpp/src/llama-quant.cpp` (read from the cloned tree). For
`Q3_K_M`: `attn_v` (and fused `attn_qkv`) Q5_K in the first two layers, then Q4_K; `ffn_down` Q5_K in the first `n/16` layers,
then Q4_K; `attn_output` Q4_K; everything else Q3_K; `output.weight` and tied embeddings Q6_K. For `Q4_K_M`: `attn_v` and
`ffn_down` Q6_K wherever `use_more_bits(i, n)` holds, which is `i < n/8`, `i >= 7n/8`, or every third layer in between. The
code carries no measurement; it is 2023 heuristics that every later GGUF recipe inherits. `--tensor-type regex=type`,
`--output-tensor-type` and `--token-embedding-type` override it [13]; ik_llama.cpp has the equivalent `--custom-q regex=type`,
a `--dry-run` that prints the resulting sizes, and `llama-imatrix --layer-similarity`, which records the cosine similarity between
a layer's input and output as a cheap importance statistic [14]. llama.cpp's `llama-imatrix --show-statistics` ranks tensors and
layers by their importance scores [15].

### 2.2 Outlier protection

Three mechanisms keep a small fraction of weights in high precision:

- **Unstructured outliers (SpQR).** During GPTQ-style column sweeps, a weight is an outlier when its leave-one-out quantization
  error exceeds a threshold that is relative to a per-layer scale (`mean(var(W, dim=0) / diag(H_inv_cho)^2)` in
  `spqr_engine.py`, threshold 0.2 in the README's example); the group's quantizer is refit without the outliers and they are
  stored sparse [1]. SpQR's bit accounting (`main.py:get_average_number_of_bits`) for 3-bit codes, groups of 16 and 3-bit
  second-level scales and zeros is 3 + 6/16 + 2 x 32/256 = 3.625 bits before outliers; the paper's Table 3 shows that 3.63 next to
  LLaMA-65B perplexities of 3.74 / 5.73 / 7.02 (WikiText2 / C4 / PTB) against GPTQ 4-bit's 3.83 / 5.80 / 7.07, and reports under
  1% relative perplexity loss for its near-lossless configurations; SqueezeLLM puts SpQR's configurations at about 4 and 4.6 bits
  on average once outliers are counted [1, 2]. SpQR's ablation found unstructured outliers reduce perplexity faster than whole-row
  outliers [1].
- **Weak columns (OWQ).** A few input columns per layer stay FP16 and the rest is 3- or 4-bit: 3.01-bit OWQ reaches WikiText2
  6.66 / 5.66 / 4.75 / 4.25 on LLaMA 7B / 13B / 30B / 65B, with 0.01 to 0.1 bits of storage overhead [3]. MXSens (2026) makes the
  same observation for modern LLMs: a small subset of highly sensitive columns accounts for a disproportionate share of the error
  [16]. Column-structured protection is what a GEMV engine can implement cheaply, because a column is one activation element
  shared by every row.
- **Runtime activation splitting (LLM.int8-style).** Columns whose activations exceed a threshold are computed in FP16 at
  runtime. It is a weight-plus-activation scheme; for a weight-only engine with BF16 activations the offline column choice above
  is the relevant form (the original paper was not retrieved in this pass).

Massive activations and super weights are the structural reason outliers exist: a handful of fixed feature dimensions carry
values thousands of times larger than the rest and act as attention biases [17]; the single "super weights" that produce them sit
in `mlp.down_proj` of an early layer in Llama, Mistral and OLMo, and pruning one destroys the model where pruning the next 7,000
largest weights does not [18]. In Qwen2.5-0.5B, `down_proj` holds 80% of the ten largest-magnitude parameters [19]. For hybrids,
a 2026 study of 12 checkpoints from 1.2B to 397B (Qwen3.5 among them) finds two shapes: pre-attention spikes immediately before
the full-attention layers and inter-spike plateaus across the intervening linear-time blocks; the attention output gates strongly
attenuate the magnitudes, removing the GDN output gates adds only modestly [20]. In Qwen3.8-27B the full-attention layers are
3, 7, ..., 63, so the GDN layers 2, 6, ..., 62 feed the spikes.

### 2.3 Byte arithmetic for this model

Streamed bytes per decode token (estimate from the inventory; the row-split layout pads K to 128, which is exact for every shape
here).

| Family | Params | Q3 (3.25) | Q4 (4.25) | Q5 (5.25) | NVFP4 (4.5) | SD4 (4.25) | 1 bpw step |
|---|---|---|---|---|---|---|---|
| `mlp/gate_up` | 11.41 B | 4.32 | 5.65 | 6.97 | 5.98 | 5.65 | 1.33 GiB |
| `mlp/down` | 5.70 B | 2.16 | 2.82 | 3.49 | 2.99 | 2.82 | 0.66 |
| `gdn/value_z` | 3.02 B | 1.14 | 1.49 | 1.85 | 1.58 | 1.49 | 0.35 |
| `gdn/output` | 1.51 B | 0.57 | 0.75 | 0.92 | 0.79 | 0.75 | 0.18 |
| `gdn/query_key` | 1.01 B | 0.38 | 0.50 | 0.62 | 0.53 | 0.50 | 0.12 |
| `attention/query_key` | 0.59 B | 0.22 | 0.29 | 0.36 | 0.31 | 0.29 | 0.07 |
| `attention/gate_value` | 0.59 B | 0.22 | 0.29 | 0.36 | 0.31 | 0.29 | 0.07 |
| `attention/output` | 0.50 B | 0.19 | 0.25 | 0.31 | 0.26 | 0.25 | 0.06 |
| `text/output_head` | 1.27 B | 0.48 | 0.63 | 0.78 | 0.67 | 0.63 | 0.15 |
| Resident only: `text/token_embedding` | 1.27 B | 0.48 | | | | | 0.15 |
| Drafts only: `text/draft_head` (131,072 rows) | 0.67 B | 0.25 | 0.33 | 0.41 | 0.35 | 0.33 | 0.08 |
| Drafts only: `mtp/*` (5 tensors) | 0.42 B | | | | 0.22 | 0.21 | BF16 0.79, FP8 about 0.40 |

The `q3` allocation sums to 10.71 GiB (15.97 ms at 720 GB/s, which is the measured token time). An outlier side plane in BF16 over
1% of the columns of every streamed tensor is 0.48 GiB (+4.5%); over 0.5%, 0.24 GiB; as 8-bit columns, half of that. The quality
exchange rate the two base-lineage forms define: `nvfp4` spends 2.7 GiB more per token than `q3` for KL 0.147 against 0.293 and
NLL +0.029 against +0.170, i.e. about 0.054 KL and 0.052 nats per GiB; a re-allocation is worth doing only if it beats that rate.
(`nvfp4-compressed`, converted from Swift-1.5 with UkisAI's matrix and measured against Swift's BF16, sits at KL 0.155, NLL +0.028
and top-1 89.4%, the same point on that curve.)

## 3. Published results

### 3.1 Transformers

| Model | What was protected or measured | Bits | Metric | Source |
|---|---|---|---|---|
| LLaMA 65B | GPTQ 3-bit with small groups, 3-bit second-level stats, unstructured outliers | 3.63 before outliers (paper's Table 3 row); about 4 to 4.6 with outliers | WikiText2 3.74 vs GPTQ-4bit 3.83; under 1% relative loss claimed for near-lossless configs | SpQR [1], SqueezeLLM [2] |
| LLaMA 7B to 65B | a few FP16 "weak" columns per layer, rest 3-bit | 3.01 | WikiText2 6.66 / 5.66 / 4.75 / 4.25 | OWQ [3] |
| OPT-6.7B | same | 3.01 | WikiText2 11.21 vs OPTQ 4-bit 11.14 and 3-bit 12.88 | OWQ [3] |
| Llama-7B | one scalar in early `mlp.down_proj` | pruned | zero-shot accuracy to near zero; next 7,000 outliers change it by a few points | Super Weight [18] |
| Qwen2.5-0.5B | magnitude ranking of all parameters | | 80% of top-10 and 52% of top-100 magnitudes in `down_proj`; layer-2 `down_proj` amplification mean 40.6, max 177.8 | [19] |
| llama.cpp mixes (all models) | `attn_v`, `ffn_down`, `attn_output`, output head, first/last n/8 layers | +1 to +3 bits over the base type | none recorded in code | `llama-quant.cpp` [13] |
| Qwen3.8-27B (base) | bartowski imatrix GGUFs | Q8_0 29.1 GB; Q4_K_M 17.4 GB; IQ3_M 14.9 GB; Q3_K_M 13.4 GB | mean KLD 0.0009 / 0.0139 / 0.0406 / 0.0564 (uploader's corpus) | [21] |
| Qwen3.8-27B | per-tensor "computed layout" against the standard mix at equal size | Q3_K_M, IQ2_XXS, Q4_K_M, Q6_K | 0.79x, 0.78x, 0.94x, 0.96x the standard layout's KLD | [22] |
| Qwen3.8-27B | AtomicChat AD-IQ3_S against unsloth Q3_K_M at 13.8 GB | about 3.5 bpw | mean KLD 0.0325 vs 0.0484 (33% less, same reference and text) | [23] |
| Qwen3-8B-Base | LM head at INT4, group-wise | 4 | relative perplexity 1.151; with low-rank residual correction 1.007 at 25.6% of BF16 head storage; Gemma-4-E4B INT4 head 1.013 | ARCHead [24] |

The GGUF KLD values are not comparable with the engine's KL of 0.293 (one 1279-token document, different reference and tokenizer
run), but the ratios are the useful part: at about 3.5 bpw a per-tensor layout removes 20 to 33% of the KLD at equal size, and at
4.8 bpw and above only 4 to 6%. Allocation has more to give at the `q3` point than at the `nvfp4` point.

### 3.2 SSM, linear attention and Gated DeltaNet

| Model | Finding | Bits | Metric | Source |
|---|---|---|---|---|
| Mamba2 130M to 8B | W4A8: `in_proj` and `out_proj` *weights* 4-bit (`W4A8B8O8LinearParallel`, `W4A8B16O16Linear` in `qMamba2.py`); the SSM inputs x (sorted and clustered), B and C (per state group) and dt are the 8-bit part; embedding and LM head quantizable | W4A8, W8A8, W4A16 | Mamba2-8B WikiText2: FP16 7.25, W8A8 7.79, W4A8 7.94 (as tabulated by SSDi8) | Quamba2 [25, 26], SSDi8 [27] |
| Mamba2-8B | mixed precision W4AX: 14 of 56 blocks at W4A16 (blocks 0, 4, 5, 9, 10, 17, 21, 22, 28, 33, 35, 48, 49, 54), the rest W4A8; chosen by evolutionary search over "salient" blocks | 4-bit weights everywhere | the repo's `configs/hybrid/mamba2-8b/hybrid_blocks_config.json` | Quamba2 repo [26] |
| Mamba 790M, Vim | KLT-enhanced Hadamard rotation; W8A8 within 1 point of FP; W4A8 Mamba-790M 42.3 vs FP16 54.8 accuracy | W8A8, W4A8 | accuracy | MambaQuant [28] |
| Qwen3.8-27B | all 496 backbone linears to NVFP4 W4A4 by PTQ, accuracy close to BF16 on reasoning, knowledge, code, retrieval and long context; gate projections least sensitive (about 11% GEMM error becomes about 2% output error through softplus / sigmoid); delta-rule recurrence forgets injected error within hundreds of tokens and the 32K perplexity gap shrinks with position; KV-cache quantization costs +0.41 perplexity at 32K, 3x more for the quantized model, recovered by calibrated per-layer scales | W4A4 | benchmarks, perplexity | [29]; an independent review calls the mechanism story only partly supported |
| GLA-1.3B (pretraining) | outliers come from softmax in attention, gating in linear attention and SwiGLU; "post-QK" operations most sensitive; linear attention has lighter per-tensor tails but persistent block-level spikes | NVFP4 training | loss gap to BF16 0.94% to 0.58% with hot-channel patching | [30] |
| 12 hybrids 1.2B to 397B | pre-attention spikes before full-attention layers, plateaus across linear blocks; attention output gates attenuate, GDN output gates matter less | | magnitudes | [20] |
| Hymba and Mamba variants | KL(student to teacher) best layer-level sensitivity metric, tau 0.79 vs SQNR 0.76 | mixed | correlation with end-to-end degradation | [7] |

496 is exactly Qwen3.8-27B's linear count: 48 GDN layers x 5 (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`)
+ 16 attention layers x 4 + 64 MLP x 3 (this chapter's arithmetic), so the study covered the `a` and `b` gate projections that
Euhedral keeps in BF16 and the community pins to Q8_0. "Close to BF16" on benchmarks is a coarser statement than the engine's
teacher-forced KL: the engine's own NVFP4 form, which quantizes only weights, still shows KL 0.147 and top-1 agreement 88%.

### 3.3 Qwen3.8-27B community recipes (uploaders' own reports)

| Build | Allocation | Reported effect |
|---|---|---|
| ISTA-DASLab GSQ-RCO (IQ2_XS 2.50, IQ2_S 2.75, IQ3_XXS 3.00, IQ3_S 3.50 bpw; `-mtp` variants) | one type per tensor from a budgeted search; a third-party rendering of the IQ3_S map shows `attn_v` mostly Q4_K, `ssm_out` spread over IQ3_XXS to Q4_K, `ffn_down` spread from IQ2_S to Q4_K | the source of the `imatrix-qwen3.8-27b.gguf` the `q3` and `nvfp4` artifacts use; ukisai republished the allocation for Swift 1.5 with its own matrix (`imatrix-swift15-v1mix.gguf`), which the `nvfp4-compressed` artifact uses [10, 11, 31] |
| cdiamond iMatrix NVFP4 MTP, 5.01 bpw | MLP in NVFP4; selected attention, DeltaNet and late FFN tensors at Q5_K / Q6_K by imatrix ranking; embeddings Q6_K; output head Q8_0; MTP left NVFP4 because a Q5_K MTP added 50.6 MiB and cost 3.39% throughput | "matched the Q4_1 reference within test error"; no KLD published [32] |
| utautako NVFP4-MTP-Q8attn | MLP NVFP4; attention and DeltaNet projections (`ssm_out`, `ssm_alpha`, `ssm_beta` included) Q8_0, or Q6_K in the 5.08 bpw variant | claim that 4-bit versions "hurt most at long context"; no numbers [33] |
| outsourc-e "Unleashed" | `ssm_alpha` / `ssm_beta` (96 tensors) Q8_0 in every tier down to 1-bit | uniform Q3_K on them "scored worst" in the author's tests [34] |
| AtomicChat | `--tensor-type` overrides: `attn_gate`, `ssm_out` to Q5_K, FFN down to Q4_K | AD-IQ3_S 0.0325 KLD at 13.8 GB [23] |
| vmarcelo MIX | "sensitive trio": `output.weight`, `attn_v`, `ffn_down` at Q5_K or above, after Unsloth Dynamic | no numbers [35] |

Two facts explain the `ssm_alpha` / `ssm_beta` pins. In llama.cpp these are the GDN `in_proj_a` / `in_proj_b` matrices (5120 x 48,
`gguf-py/gguf/tensor_mapping.py`), and `tensor_allows_quantization` excludes them only for GLM5-Next, not for `qwen35`, so a plain
`Q3_K_M` of Qwen3.8 quantizes them to Q3_K; the community then found that harmful and pinned them. Euhedral stores them in BF16
(45 MiB in total) and never had the problem. Whether they are sensitive at all is contested: the Qwen3.8 NVFP4 study calls them the
least sensitive layers [29]. The two statements are compatible: 4-bit with 16-element block scales and 3-bit with 256-element
super-blocks are very different quantizers for a 48-row matrix.

### 3.4 Heads, embeddings and drafters

| Topic | Finding | Source |
|---|---|---|
| LM head | llama.cpp gives `output.weight` Q6_K by default (Q5_K at IQ2 levels, Q8_0 for MXFP4 MoE); community Qwen3.8 builds keep it Q6_K to Q8_0 | [13, 32] |
| LM head | Qwen3-8B INT4 head: relative perplexity 1.15; Gemma-4-E4B 1.013; INT4 core + activation-metric low-rank residual 1.007 at 25.6% of BF16 storage | ARCHead [24] |
| Embedding | llama.cpp: Q2_K at IQ1/IQ2 levels, IQ3_S at IQ2_S/IQ3_XXS, otherwise the base type; Gemma 4 ships 4-bit embeddings (QAT) | [13]; DEV posts in the search, not primary |
| Vocabulary | VocabTrim: trimming the *drafter's* LM head by up to 75% has negligible effect on acceptance for Llama 3; general vocabulary pruning pays mostly for small models | [36] |
| Draft quantization | GPTQ on an EAGLE draft "resulted in substantial degradation of the acceptance rate" | QSpec [37] |
| Draft quantization | direct-cast MXFP4 of the 7B target as its own draft: 71.2% acceptance (QA) vs 41.7% for a trained 68M draft; about 91% on code | ML-SpecQD [38] |
| Draft quantization | 4-bit EAGLE-3 head on Llama-8B: acceptance 0.34, unchanged from unquantized (5 prompts, MLX) | [39] |
| Self-speculation | 4-bit weights and 4-bit KV as the draft: acceptance above 90% | QuantSpec [40] |

The published pattern is that the *capacity* of a drafter dominates its *precision*: a 4-bit copy of the target out-drafts a
small full-precision model, and a 4-bit EAGLE head keeps its acceptance, while a GPTQ'd EAGLE loses it. The engine's own
measurements agree in direction and give the magnitude for this model: a Q3 MTP layer accepted 1.41 drafts per verification and
the NVFP4 one 1.49 (MTP2, 1024 tokens, `docs/MTP_VERIFIER.md`), i.e. 3.25 to 4.5 bits bought +5.7% accepted drafts and +2.9%
tokens per verification; an NVFP4 draft head accepted slightly *less* than the Q3 head (1.44 vs 1.49); SD4 on the drafting stack
cost 1.0 to 1.1% tokens per step (`docs/NVFP4_COMPRESSED.md`). The head's conditional acceptance settles at 0.70 to 0.85 from the
second draft on because it reads its own hidden states as if they were the target's (`docs/MTP_DRAFT_CONFIDENCE.md`); that ceiling
is a training property, not a precision one.

## 4. Fit with Euhedral

### 4.1 The eight constraints

1. **Row-exact 1 to 8 row decode.** Moving a family between Q3G64, Q4G64, Q5G64, NVFP4 and SD4 uses kernels and twins that
   exist and are tested bitwise (`test_q3_kernels.py`, `test_q45_kernels.py`, `test_nvfp4.py`). A Q2G64 format would need a new
   GEMV and twins. An outlier side plane is row-exact by composition: each row's main sum and its side sum are each computed in a
   fixed order, and the one-row and M-row kernels add them the same way (section 4.5).
2. **Prefill at 9+ rows.** Integer codes of any width expand exactly to E4M3, so Q2/Q3/Q4/Q5 all ride the MXFP8 route; NVFP4 and
   SD4 ride OMMA. A side plane adds a BF16 GEMM of M x c x N with c about 50, under 1% of the FLOPs, on any route.
3. **Decode instruction budget.** Format swaps change nothing. A side plane fused into the GEMV epilogue adds c FMAs per row
   against K of 5120 to 17408. Putting 8-bit columns *inside* the group layout would need an escape path per group, the same
   instruction cost that made P2E2 13% slower; the dense side plane is the right shape.
4. **Determinism.** Unchanged.
5. **Offline conversion.** All of this is converter work: `recipes.storage()` gains per-object overrides (regex to format, as
   llama.cpp `--tensor-type` and ik_llama.cpp `--custom-q`), and `importance.py` gains mappings for `text/output_head`,
   `text/draft_head` (the `output.weight` entry) and the MTP layer. Only Q2G64 or a side plane needs a new format ordinal.
6. **The artifact decides.** The engine already serves mixed artifacts (a Q3 embedding inside NVFP4 artifacts; SD4 told apart from
   NVFP4 by byte size in `nvfp4_linear.c`). A Q4 or Q5 *linear* inside an NVFP4 artifact, or an NVFP4 linear inside a `q3`
   artifact, has not been exercised: the loader and `ExecutionPlan` dispatch per tensor format, so it should work, but it is the
   first thing to test.
7. **Memory is the product.** The `q3` deltas below are per-token bytes (speed at every context) and resident bytes (context
   headroom: 0.35 GiB is 20K tokens of KV). On the `nvfp4` forms every 128 MiB saved is 2 to 2.5% decode speed wherever weights
   are host-backed (32K and beyond), and nothing at 4K.
8. **Drafting precision is a separate knob.** Section 4.4.

### 4.2 Proposed allocation, `q3` form

| Family | Current | Proposed probe | Streamed GiB delta | Evidence and expected effect |
|---|---|---|---|---|
| `gdn/value_z` (v 6144 rows + z 6144 rows) | Q5G64 | Q4G64 | -0.35 (-3.3%) | GDN linears survive NVFP4 W4A4 on this model [29]; delta-rule state error decays; z is the output gate, which attenuates [20, 29]. The Transformer "V is sensitive" rule is about a value that stays in the KV cache for the whole context; a GDN value is overwritten along later key directions. Expected: ΔKL small; must stay under +0.019 by the exchange rate. |
| `gdn/query_key` | Q4G64 | Q3G64 | -0.12 (-1.1%) | q and k are the least protected tensors in every llama.cpp mix [13]; no GDN-specific evidence either way; the key direction decides what the state overwrites, so a key error is not forgotten the way a value error is. Probe after `value_z`; must stay under +0.006 KL. |
| `gdn/output` | Q3G64 | Q4G64 | +0.18 (+1.6%) | pinned at Q5_K / Q8_0 by three community builds [23, 33]; RCO gives it a spread up to Q4_K [10]; it writes the residual stream. Expected: a measurable KL gain; must exceed 0.010 KL. |
| `attention/output` | Q3G64 | Q4G64 | +0.06 (+0.5%) | `Q3_K_M` gives `attn_output` Q4_K [13]. Cheap; must exceed 0.003 KL. |
| `mlp/down` | Q3G64 | Q4G64 in layers 0 to 7 and 56 to 63 (option A, 16 layers); all 64 layers (option B) | +0.17 (A) / +0.66 (B) | the most protected MLP tensor in every mix (`Q3_K_M`: first n/16 at Q5_K, rest Q4_K [13]); super weights live in early `down_proj` [18, 19]; cdiamond spends on late FFN [32]. Option B is the ceiling measurement; A is what the budget allows. B must gain 0.036 KL, A 0.009. |
| `attention/gate_value` | Q5G64 | keep Q5 | 0 | `attn_v` is the canonical sensitive tensor and its error persists in the KV cache; only 0.07 GiB to gain at Q4. |
| `attention/query_key` | Q4G64 | keep (optional Q3 probe, -0.07) | 0 | same reasoning as `gdn/query_key`, smaller payoff. |
| `mlp/gate_up` | Q3G64 | keep | 0 | 40% of streamed bytes; going lower is a new-format question (other chapters); going higher is 1.33 GiB per bit. |
| `text/output_head` | Q3G64, uniform weights | Q3G64 with imatrix (`llama-imatrix --process-output`, mapped in `importance.py`); Q4G64 if the head carries a large share of the KL | 0, then +0.15 | head sensitivity is model-dependent and Qwen heads are the sensitive kind (INT4 head 1.15x perplexity on Qwen3-8B [24]); llama.cpp never puts the head below Q5_K [13]. Expected: the imatrix alone is a free gain; the head's KL share is unknown and is probe 1 in section 5. |
| `text/token_embedding` | Q3G64 resident | keep; host-map when KV needs the room (the planner already does this first) | 0 | a gather's error is per token and does not accumulate; Q2 would need a new gather kernel for 0.15 GiB resident. |

Net of the proposed probes with option A: -0.35 -0.12 +0.18 +0.06 +0.17 = -0.06 GiB streamed, i.e. a quality-for-free
rebalancing if the GDN evidence holds; with option B and a Q4 head, +0.70 GiB (+6.5%, about -6% decode speed), which only makes
sense if it buys more than 0.038 KL. This chapter's expectation (estimate, from the 20 to 33% KLD reductions computed layouts
achieve at this bit rate [22, 23]): the rebalancing is worth 0.03 to 0.06 KL on the engine's scale, which the single document can
just resolve.

### 4.3 Proposed allocation, `nvfp4` and `nvfp4-compressed` forms

| Family | Current | Proposed | Delta | Reasoning |
|---|---|---|---|---|
| all base linears | NVFP4 4.5 / SD4 4.25 | keep SD4 | 0 | GDN and MLP linears survive 4-bit on this model [29]; the only saving below 4 bits is a code-bit change (other chapters); the only spend that the exchange rate could justify is `mlp/down` at Q5G64 (+0.50 GiB) and it would be host-backed at 32K and beyond, costing about 8 to 10% speed there. Not proposed. |
| `text/output_head` | SD4, uniform weights | SD4 with the `output.weight` imatrix | 0 | free; the head is never host-backed, so its bytes are the same at every context. |
| `text/draft_head` | NVFP4 (0.35 GiB) | Q3G64 (0.25 GiB) | -0.10 resident | the Q3 head accepted 1.49 vs 1.44 for NVFP4 on the `q3` base (one pass); a Q3 head is also fewer bytes per draft row (about -0.14 ms per draft, estimate). Needs one acceptance pass on `nvfp4-compressed`. |
| `mtp/*` | NVFP4 | keep | 0 | section 4.4. |
| `text/token_embedding` | Q3, host-mapped under pressure | keep | 0 | measured at no cost. |
| `attention/gate_value` | SD4 | optional Q5G64 probe | +0.05 | the one Transformer-sensitive family where the spend is cheap; only if the KL scan singles it out. |

### 4.4 Drafting stack

| Option | Resident GiB | Bytes per draft row | Acceptance reasoning | Verdict |
|---|---|---|---|---|
| MTP layer NVFP4 (shipped) | 0.22 | 0.22 | measured 1.49 drafts (MTP2) and 2.84 to 3.49 tokens per verification (MTP3/4) | baseline |
| MTP layer BF16 | 0.79 (+0.57) | +0.57 GiB per draft row: about +0.85 ms per draft at 720 GB/s (estimate), 2 to 4 drafts plus catch-up per step, so +2 to +4 ms on a 30 to 50 ms step (+5 to 10%) | the 3.25 to 4.5 bit step bought +2.9% tokens per verification; the 4.5 to 16 bit step cannot plausibly buy the 5 to 10% needed, and the head's 0.70 to 0.85 conditional acceptance is a training ceiling, not a precision one | no; one acceptance pass on `q3` at 4K (where it fits) would bound the question for good |
| MTP layer FP8 | about 0.40 (+0.18) | +0.27 ms per draft (estimate) | would need about +2% tokens per verification; no FP8 linear kernel exists | not before the BF16 bound is measured |
| MTP layer Q5G64 | 0.26 | 0.26 | FMA twins make catch-up rows slower (Q3 catch-up measured 2.0 vs 1.4 ms) | no |
| MTP layer SD4 | 0.21 (-0.01) | | measured -1.0 to -1.1% tokens per step | rejected already |
| Draft head Q3 on `nvfp4` forms | -0.10 | -0.10 | measured no loss on the `q3` base (1.49 vs 1.44) | yes, confirm with one pass |
| Draft head 65,536 rows | -0.13 (Q3) | -0.13 | measured -4.2% / -6.9% accepted drafts; the VocabTrim result (75% trim, negligible loss) does not hold for this head | no |
| Draft head 2-bit | -0.08 | | new format for 78 MiB | no |

The row-exact verifier is untouched by any of these: drafts only propose.

### 4.5 Outlier side plane and the verifier

A column-structured plane (OWQ-style) stores, per tensor, c BF16 columns (N x c) plus c column indices; the main tensor keeps
those columns at zero (or, better, excludes them from the group's scale search, which is where the gain comes from: a super weight
in a group of 64 forces the group scale up and crushes its 63 neighbours, and the calibrated search cannot avoid that because it
maps `amax` to the code range). Decode: the GEMV's row CTA gathers the c activations once, computes the c products in index order
in FP32 and adds them to the row accumulator before writing; the M-row twins do the same per row, so every row is bitwise the
one-row result, exactly as today. For NVFP4 `decode_rows` the side sum is an FP32 FMA term appended after the MMA chunks in warp
order; rows stay independent MMA columns. Prefill adds a small BF16 GEMM. Cost: +4.5% streamed bytes at 1% of columns (+2.2% at
0.5%), one gather of c activations per row-tile, no extra launches. Unstructured SpQR outliers do not fit: the per-code position
dependence is the variable-length decoding that constraint 3 forbids. Published payoff at 3 bits is large (SpQR and OWQ close most
of the gap to 4-bit GPTQ [1, 3]), but both sit on top of GPTQ error feedback, which the engine does not have; the honest expectation
is that the side plane pays only for tensors whose weighted error is concentrated in few columns, which the converter can measure
offline (section 5, probe 0).

### 4.6 Small objects

| Object group | Count | Params | Bytes | Keep as |
|---|---|---|---|---|
| `gdn/a_projection`, `gdn/b_projection` (48 x 5120 each) | 96 | 23.6 M | 45.0 MiB BF16 | BF16: the community's worst-case tensors under Q3_K [34]; the NVFP4 study says 4 bits would be fine [29], but 4 bits would save 34 MiB, 0.3% of the artifact |
| `gdn/convolution` (4 x 10240) | 48 | 2.0 M | 3.8 MiB BF16 | BF16: llama.cpp never quantizes `ssm_conv1d` [13] |
| `gdn/a_log`, `gdn/dt_bias` | 96 | 4.6 K | 18 KiB FP32 | FP32 |
| norms (input, post-attention, gdn, q/k, final, MTP) | about 215 | 0.7 M | 1.3 MiB BF16 | BF16 |
| recurrent state (48 x 128 x 128 per layer) and conv state | per sequence | | 0.15 GiB FP32 | FP32: the SSM state and its inputs are the hard part of SSM quantization [25]; 0.15 GiB per sequence is not where the memory is |
| total | 456 | 26.5 M | 0.05 GiB | |

## 5. Experiment plan

Infrastructure first (one day of converter work, no GPU time): (a) `--override regex=format` in `recipes.storage()`, with the
manifest recording the overrides; (b) `importance.py` mappings for `text/output_head` and `text/draft_head` to `output.weight`,
and for the MTP layer to the `nextn.*` entries, after re-running `llama-imatrix` with `--process-output` (and `--nextn` if the GGUF
carries the MTP layer) on the BF16 GGUF of each lineage in use, the base checkpoint for `q3` and `nvfp4` and Swift-1.5 for
`nvfp4-compressed`, since neither published matrix covers the head; (c) a `--report-error` pass that writes, per object and per candidate format,
the imatrix-weighted relative squared error the search already computes, and the share of that error carried by the top 0.5% and
1% of columns. Cost per probe: a calibrated `q3` conversion is a minute or two (measured); the teacher-forced run is 1281 prefill
tokens plus 1279 one-row steps at about 16 ms, 1 to 2 minutes with loading (estimate); 3 to 5 minutes per probe, with the 11 to
12 GiB artifacts built and deleted in sequence. Every probe changes one thing against the shipped artifact and reports NLL, paired
dNLL with s.e., KL and top-1 from `compare_reference.py`. The decision rule is the exchange rate of section 2.3: a saving of X GiB
may cost at most 0.054 X KL, a spend of X GiB must gain at least 0.054 X KL, and anything under about 0.02 KL or 0.02 nats is not
resolved on the current document and goes to the larger held-out evaluation before it ships.

| # | Probe | Cost | Go if | Payoff |
|---|---|---|---|---|
| 0 | offline: per-object weighted error at Q3/Q4/Q5 and top-1% column share of the error (no artifact) | minutes | columns carrying over 30% of a tensor's error flag it for the side plane; the error ranking orders probes 2 to 6 | free |
| 1 | LM head alone at Q5G64 (then with the `output.weight` imatrix at Q3) | 2 probes | KL falls by more than 0.02: the head is a large share of `q3`'s 0.293 and the imatrix is the first fix; Q4 head if the imatrix recovers less than half of the Q5 gain | 0 bytes, or +0.15 GiB |
| 2 | `gdn/value_z` Q5 to Q4 | 1 probe | ΔKL under +0.019 and dNLL within 1 s.e. | -0.35 GiB per token, -3.3% |
| 3 | `gdn/output` Q3 to Q4; `attention/output` Q3 to Q4 | 2 probes | ΔKL under -0.010 and -0.003 respectively | +0.24 GiB |
| 4 | `mlp/down` all layers Q4 (ceiling), then layers 0 to 7 and 56 to 63 only | 2 probes | ceiling gain over 0.036 KL justifies option B; if the 16-layer set captures over half of the ceiling, ship option A | +0.66 or +0.17 GiB |
| 5 | `gdn/query_key` Q4 to Q3; `attention/query_key` Q4 to Q3 | 2 probes | ΔKL under +0.006 and +0.004 | -0.19 GiB |
| 6 | per-layer scan of the winner of 2 to 4, in 8-layer blocks, then bisect; separately layers 2, 6, ..., 62 (the GDN layers before each attention layer) against the other GDN layers | 8 + 2 probes | any block differing from the mean by over 0.02 KL gets its own rule | layer-dependent bits, as `use_more_bits` does |
| 7 | combined `q3` recipe (2 + 3 + 4A + 1) | 1 probe + larger eval | total ΔKL under -0.03 at under +0.1 GiB; speed within 1% at 4K and 32K; `SpeculativeDecodeCudaIntegrationTest` passes (it must, nothing in the verifier changed) | the new `q3` |
| 8 | draft head Q3 on `nvfp4-compressed` | 1 acceptance pass (deterministic) | tokens per verification within 1% of NVFP4's | -0.10 GiB resident, slightly faster drafts |
| 9 | MTP layer BF16 on `q3` at 4K, acceptance only | 1 pass | if tokens per verification rise under 3%, close the question for every precision above NVFP4 | bounds the drafting-stack question |
| 10 | side plane prototype on the one or two tensors probe 0 flags, as a converter-side simulation first (BF16 columns, main tensor requantized without them, error reported), kernels only if the simulated weighted error falls by over 20% | days | KL gain per GiB above the exchange rate | +2 to 4.5% bytes for a quality step |

The llama.cpp toolchain in the user's hands gives a cheap second track for ranking only: `llama-quantize --tensor-type` variants of
the Swift 1.5 GGUF scored with `llama-perplexity --kl-divergence-base` reproduce probes 2 to 5 in Q3_K / Q4_K / Q5_K terms without
touching the engine; the ranking should transfer, the magnitudes will not.

## 6. Risks and open questions

- **Resolution.** The current document resolves about 0.02 KL. Probes 2 and 5 are savings whose acceptable loss is of that order;
  a 10K-token held-out evaluation with its own BF16 reference is needed before any saving ships.
- **The GDN evidence is one paper on this model.** [29] is a September 2026 preprint with an independent review calling its
  mechanism only partly supported, and its headline metric is benchmark accuracy, which hides KL of the size the engine measures.
  Its quantizer (16-element blocks, calibrated per module) is finer than Q4G64. Probe 2 is the test.
- **Long context.** Two sources argue in opposite directions: delta-rule forgetting says GDN weight error does not accumulate
  [29]; a community uploader claims 4-bit DeltaNet projections hurt most at long context [33], with no numbers. The harness
  measures at 1281 tokens of prefix only. A 32K teacher-forced slice (prefix 30K, 1279 steps) on the winning recipe is the check,
  and the `q3` form is the one that runs 128K, so it carries the risk.
- **Two lineages.** `q3` and `nvfp4` are conversions of the base checkpoint with ISTA-DASLab's matrix; `nvfp4-compressed`, the
  artifact deployed by default, is a conversion of Swift-1.5 with UkisAI's `imatrix-swift15-v1mix.gguf` (309 chunks of 4096
  tokens), each measured against its own BF16 (baseline sections 1 to 3). The `q3` probes of section 5 therefore run on the base
  lineage while the deployed form is Swift; a reallocation found on one checkpoint has to be confirmed on the other before it
  ships, the `output.weight` imatrix has to be generated per lineage, and the MTP layer's provenance in Swift-1.5 is undocumented.
- **Mixed-format dispatch.** A Q4/Q5 linear inside an NVFP4 artifact (or NVFP4 inside `q3`) is untested in the loader and plan.
- **Head and embedding at 2 bits.** Not proposed: Qwen heads are the sensitive kind [24], the saving is 0.15 GiB each, and both
  need new kernels. The 243 padding rows (248,320 minus 248,077) are 0.5 MiB; the count of reserved but unused tokenizer entries is
  not known here, and trimming them would change greedy output in the rare case one of them wins.
- **A low-rank head residual** (ARCHead: INT4 core plus residual at 25.6% of BF16 storage, relative perplexity 1.007 on Qwen3-8B
  [24]) suggests a Q3 head plus a rank-64 BF16 residual (about 30 MiB, one extra small GEMV, row-exact) as a cheaper way to buy head
  quality than Q4; untested here.
- **Outlier payoff without GPTQ.** SpQR and OWQ gains sit on GPTQ error feedback; without it, protected columns fix the group-scale
  problem but not the rounding of everything else. Probe 0 quantifies before any kernel is written. Codebook formats also absorb
  outliers (see the codebook chapter); FP6 for `value_z` is the FP4-formats chapter's question.
- **Fused tensors and scales.** The engine fuses q+k, gate+v, v+z and gate+up at conversion and calibrates each fused object, with
  Q group scales per row, so the per-module-calibration pitfall of [29] does not apply; but a per-part format (v at Q5, z at Q4
  inside one object) would need the layout to carry two formats, which it does not.

## 7. References

1. Dettmers, Svirschevski, Egiazarian et al., "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression", 2023, arXiv 2306.03078; code github.com/Vahe1994/SpQR (cloned; `spqr_engine.py`, `main.py`).
2. Kim et al., "SqueezeLLM: Dense-and-Sparse Quantization", 2023, arXiv 2306.07629.
3. Lee et al., "OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models", AAAI 2024, arXiv 2306.02272; code github.com/xvyaward/owq.
4. Dong et al., "HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks", NeurIPS 2020, arXiv 1911.03852.
5. "KronQ: LLM Quantization via Kronecker-Factored Hessian", COLM 2026, arXiv 2607.07964.
6. "Channel-Wise Mixed-Precision Quantization for Large Language Models", arXiv 2410.13056.
7. Kong, Pandey, Ponzina, Rosing, "A KL Lens on Quantization: Fast, Forward-Only Sensitivity for Mixed-Precision SSM-Transformer Models", April 2026, arXiv 2604.13440.
8. Unsloth, "Unsloth Dynamic 2.0 GGUFs", https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs and https://unsloth.ai/blog/dynamic-v2.
9. Lyra-Vhess, "OpenDynamicGGUF-UngreedySolver", README (fetched), https://github.com/Lyra-Vhess/OpenDynamicGGUF-UngreedySolver.
10. ISTA-DASLab, "Qwen3.8-27B-GSQ-RCO-GGUF", https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF (model card and per-tensor allocation files, seen in search).
11. "GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling", arXiv 2604.18556; code github.com/IST-DASLab/GSQ (README fetched; RCO not described there).
12. IST-DASLab/GSQ issue 9, "Reproduction tooling for Qwen3.8-27B GSQ-RCO mixed-precision GGUF release".
13. ggml-org/llama.cpp (cloned 2026-10-10): `src/llama-quant.cpp`, `tools/quantize/README.md`, `tools/imatrix/imatrix.cpp` (`--process-output`, `--nextn`), `gguf-py/gguf/tensor_mapping.py`, `src/models/qwen35.cpp`.
14. ikawrakow/ik_llama.cpp (cloned): `docs/parameters.md` (`--custom-q` PR 244, `--layer-similarity` PR 328, `--dry-run` PR 1309).
15. ggml-org/llama.cpp PR 12718, "imatrix: add option to display importance score statistics", and discussion 12741, "Experimenting with custom quants using --tensor-type".
16. "MXSens: Sensitivity-Aware Mixed-Precision Quantization for Efficient LLM Inference", arXiv 2607.17733.
17. Sun et al., "Massive Activations in Large Language Models", 2024, arXiv 2402.17762.
18. Yu et al., "The Super Weight in Large Language Models", 2024, arXiv 2411.07191.
19. "Super Weights in LLMs and the Failure of Selective Training", 2026, arXiv 2607.08733.
20. "Massive Activations in Hybrid Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike Plateaus", 2026, arXiv 2608.12149.
21. bartowski, "Qwen3.8-27B-GGUF", https://huggingface.co/bartowski/Qwen3.8-27B-GGUF.
22. bartowski, "ukisai_Swift-Qwen3.8-27b-GGUF" (computed per-tensor layout comparison), https://huggingface.co/bartowski/ukisai_Swift-Qwen3.8-27b-GGUF.
23. AtomicChat, "Qwen3.8-27B-GGUF" and Qwen/Qwen3.8-27B discussion 65, https://huggingface.co/Qwen/Qwen3.8-27B/discussions/65.
24. Kocabay et al., "ARCHead: Activation-Metric Residual Correction for Large Language Model Output Heads", 2026, arXiv 2608.02703.
25. Chiang et al., "Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models", ICML 2025, arXiv 2503.22879, PMLR v267 chiang25a.
26. enyac-group/Quamba (cloned): `quamba/qMamba2.py`, `configs/hybrid/mamba2-8b/hybrid_blocks_config.json`, README.
27. "SSDi8: Accurate and Efficient 8-bit Quantization for State Space Duality", 2026, arXiv 2608.21952 (source of the tabulated Quamba2 Mamba2-8B perplexities).
28. Xu et al., "MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods", ICLR 2025, arXiv 2501.13484.
29. Kozyrev, Maiboroda, "Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM", September 2026, arXiv 2609.04098.
30. Dong et al., "Dissecting Outlier Dynamics in LLM NVFP4 Pretraining", February 2026, arXiv 2602.02047.
31. ukisai, "Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF", https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF.
32. cdiamond, "Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF", https://huggingface.co/cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF, and the author's write-up (DEV Community / NVIDIA forums, seen in search).
33. utautako, "Qwen3.8-27B-NVFP4-MTP-Q8attn-GGUF", https://huggingface.co/utautako/Qwen3.8-27B-NVFP4-MTP-Q8attn-GGUF.
34. outsourc-e, "Qwen3.8-27B-Unleashed-GGUF", https://huggingface.co/outsourc-e/Qwen3.8-27B-Unleashed-GGUF.
35. vmarcelo, "Qwen3.8-27B-MIX_GGUF", https://huggingface.co/vmarcelo/Qwen3.8-27B-MIX_GGUF.
36. "VocabTrim: Vocabulary Pruning for Efficient Speculative Decoding in LLMs", 2025, arXiv 2506.22694.
37. "QSpec: Speculative Decoding with Complementary Quantization Schemes", 2024, arXiv 2410.11305.
38. "ML-SpecQD: Multi-Level Speculative Decoding with Quantized Drafts", 2025, arXiv 2503.13565.
39. ml-explore/mlx-lm discussion 890, "EAGLE-3 speculative decoding prototype: implementation + Apple Silicon analysis" (4-bit EAGLE-3 head).
40. "QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache", 2025, arXiv 2502.10424.
41. Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule", ICLR 2025, arXiv 2412.06464 (the recurrence S_t = alpha_t S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T, which is what "overwriting along the key direction" refers to).
42. Euhedral repository (branch `perf/mtp-gated-depth`): `tools/euhedral_artifacts/{inventory,recipes,importance,grouped}.py`; `docs/MTP_VERIFIER.md`, `docs/MTP_SPECULATIVE.md`, `docs/MTP_DRAFT_CONFIDENCE.md`, `docs/NVFP4_COMPRESSED.md`, `docs/NVFP4_RESIDENCY.md`, `docs/NVFP4_NATIVE.md`, `docs/QUALITY.md`, `tools/README.md`.
