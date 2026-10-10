# FP4 block formats on sm_120: what the family can still give

Scope: block-floating-point weight formats for the Qwen3.8-27B artifacts on the RTX 5070 Ti (sm_120): NVFP4, MXFP4, their
variants (block size, scale format, scale search, adaptive block scaling, non-uniform grids), FP6/FP8 for the sensitive input
projections, 2:4 sparse FP4, SVDQuant's low-rank branch, and a per-tensor scale table for Q3. Everything is measured against
`00-baseline.md`, and its eight constraints are applied to every candidate in section 4. Numbers marked *own computation* come
from two NumPy scripts written for this chapter (synthetic Gaussian, Laplacian and Student-t(4) weights, 50,000 blocks per
cell); they rank formats, they do not predict NLL. Web search ended before the Four Over Six, MR-GPTQ and model-card tables could
be read in full; where a published row carries only what a search summary showed, the gap is named.

## 1. Summary

NVFP4 is the right member of the FP4 family for this card, and the engine's SD4 variant (4.25 bpw, 16-entry E4M3 scale table,
native OMMA prefill) is already the best 4.25-bpw format sm_120 can run natively: on synthetic weights its error is 0.45 to 0.6x
MXFP4's at equal bytes, because UE8M0 scales clip or waste up to a factor 1.33 of a block's range while E4M3 scales round to
within 6%. No published FP4 variant beats NVFP4 plus a scale search by a margin the engine's harness could resolve: Four Over Six
("4/6") cuts block MSE 11 to 18% against plain amax scaling and a full E4M3 sweep (now the default in TensorRT Model Optimizer
and llm-compressor) 25 to 32%; the engine's importance-weighted search (6 codes below, 2 above RTN) already takes 13 to 19%, and
its one gap is upward: letting a block take a scale up to 5 codes above RTN, which contains the 4/6 candidate, removes a further
6 to 10% at zero kernel cost. An MSE-optimal 16-entry grid is only 0 to 11% better than E2M1 with a searched scale on
heavy-tailed weights and forfeits the 815-TFLOPS prefill route, so it is not worth building. The FP4 family cannot improve the Q3
MLP (no dense FP4 variant reaches 3.25 bpw; sparse FP4 at 3.0 bpw costs 0.5 to 1.1 nats one-shot, 0.3 to 0.4 with learned masks
that take hundreds of H100 hours), but it can improve the `q3` form's input projections: Q4G64 `query_key` replaced by SD4 at
identical bytes lowers their error by a quarter (Gaussian) to half (Laplacian) with no new kernel, and FP6 E2M3 with the existing
64-group FP16 scales (6.25 bpw) is the one 6-bit code that expands exactly to E4M3 and so runs on today's MXFP8 prefill route,
with 4 to 8x lower MSE than Q5G64 for 1.0 bpw more on 3.6 B parameters (0.42 GiB). Re-quantizing the Q3 scale plane to a
256-entry per-tensor table saves 0.37 GiB of streamed bytes for a simulated MSE increase under 1%, SD4's result transplanted.
FP8 weights, block-8 NVFP4 and SVDQuant's branch do not pay on this budget. On the activation side, the published W4A4 model
cards (Qwen3-32B NVFP4 at 99.8% OpenLLM-v1 recovery) say single-term FP4 activations are viable in principle; the engine's
one-term result (output RMS error 9.5%, KL doubled) needs rotation and per-block scale choice before a retest (rotations chapter).

## 2. Mechanism

### 2.1 Formats and containers

| Format | Element | Scale | bpw | Native MMA on sm_120 (`docs/nvidia/tensor-cores.md`) |
|---|---|---|---|---|
| NVFP4 | E2M1: sign, 2 exponent, 1 mantissa bits; magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6; two zero codes | E4M3 per 16 along K (7 bits used, 2^-9 to 448), FP32 per tensor = amax/(6 x 448) | 4.5 | `kind::mxf4nvf4 scale_vec::4X ue4m3`: 815 TFLOPS dense, 1621 sparse; FP4 activations only |
| NVFP4-SD4 (engine) | E2M1 | 4-bit index into 16 E4M3 codes per tensor, FP32 per tensor | 4.25 | the same MMA; `_sd4` twins look the scale up in registers (two `prmt`) |
| MXFP4 (OCP MX v1.0) | E2M1 | UE8M0 per 32: a power of two, bias 127, 0xFF is NaN | 4.25 | `kind::mxf4 scale_vec::2X ue8m0`: 815 TFLOPS |
| MXFP6 | E2M3 (max 7.5, subnormals m/8) or E3M2 (max 28) | UE8M0 per 32 | 6.25 | inside `kind::f8f6f4` only: 204 TFLOPS with FP32 accumulate (the full-rate block-scaled bypass needs E4M3 operands, GeForce rule 3) |
| MXFP8 | E4M3 or E5M2 | UE8M0 per 32 | 8.25 | `kind::mxf8f6f4 e4m3`: 408 TFLOPS, the engine's Q3 prefill route |
| Q4G64 / Q5G64 (engine) | INT -8..7 / -16..15 | binary16 per 64 | 4.25 / 5.25 | codes expanded exactly to E4M3 in producer warps, MXFP8 MMA, group scale applied in FP32 |

Container facts from the CUTLASS sources (commit 0b55a2f, 2026-09-23):

- Under `kind::f8f6f4` an E2M1 value sits in bits 2 to 5 of its byte and an FP6 value in the low 6 bits; `ldmatrix.b8x16.b4x16_p64`
  lands 4-bit data in the low bits, so CUTLASS shifts E2M1 fragments left by 2 before the MMA (`include/cute/atom/mma_traits_sm120.hpp`).
  The packed `mxf4` kinds use plain nibbles, which is why the engine's weight layout is consumed in place.
- TMA has packed 6-bit tensor-map types, `CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B` for `float_e2m3_t` and `float_e3m2_t` (by its
  name, 16 six-bit values with 16-byte alignment; `include/cute/arch/copy_sm90_desc.hpp`). Example
  `79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu` multiplies MXFP8 activations by `mx_float6_t<float_e3m2_t>` weights on
  sm_120, and notes GeForce has no TMA multicast and no dynamic datatypes.
- **E2M3 and E3M2 values are exactly representable in E4M3** (at most 3 mantissa bits, exponents inside 2^-6..2^8), so FP6
  weights can take the engine's prefill route unchanged apart from a 64-entry expansion table: 408 TFLOPS at the MMA, 204
  effective with two activation terms. INT6 cannot: 17..31 need 5 significant bits. Q5's -16..15 is the widest integer range
  that is E4M3-exact. Own derivation from the format definitions.
- Sparse FP4. `mma_traits_sm120_sparse.hpp` types the block-scaled sparse 16x8x128 FP4 MMA with `ValTypeA = sparse_elem<4,
  uint8_t>` (one stored byte, two codes, standing for 4 logical values) and `ValTypeE = sparse_elem<16, uint8_t>` (one metadata
  byte per 16 logical values), against `sparse_elem<2, ...>` and `sparse_elem<8, uint8_t>` for the 8-bit kinds. FP4 metadata
  therefore costs **0.5 bit per logical weight**, half of FP8's, which fits the reports that the FP4 sparse pattern is pairwise
  4:8 (a community sm_120 simulator patch; SharQ, arXiv 2606.26587; the PTX ISA page was not reachable). Per logical weight:
  2.0 code bits + 0.5 metadata + 0.5 E4M3 scale per 16 = **3.0 bpw**, 2.75 with an SD4 table. `tensor-cores.md`'s "3 to 3.5" is
  the upper half of this.
- SVDQuant (arXiv 2411.05007): per linear, the weight is smoothed against activation outliers, then split into a BF16 low-rank
  branch of rank r plus a 4-bit residual, run side by side. The branch costs 2 r (N + K) bytes; summed over this model's streamed
  linears (N + K) is 6.64 M per rank, so r = 32 is 0.40 GiB read per token (0.13 bpw equivalent) and r = 64 is 0.79 GiB.

### 2.2 Grids and their error (own computation)

Relative MSE (error energy over weight energy) of block formats on synthetic weights, block of 16 along K unless noted. Scale
searches are unweighted oracles over the stated candidates. "Engine" is the search of `nvfp4.py` (E4M3 codes 6 below to 2 above RTN).

| 4 bits, block 16 | Gaussian | Laplacian | Student-t(4) |
|---|---|---|---|
| INT4 -7..7, scale amax/7 | 0.00726 | 0.01027 | 0.01120 |
| INT4 with scale search (0.70..1.05 amax/7) | 0.00638 | 0.00937 | 0.01030 |
| E2M1, ideal (FP16) scale amax/6 | 0.00886 | 0.00833 | 0.00814 |
| E2M1, 4/6 (best of amax/6 and amax/4 per block) | 0.00741 | 0.00735 | 0.00726 |
| E2M1, continuous scale search (0.55..1.6 amax/6) | 0.00616 | 0.00633 | 0.00638 |
| NVFP4 (E4M3 scale) round to nearest | 0.00907 | 0.00862 | 0.00846 |
| NVFP4, engine search -6..+2 codes | 0.00737 | 0.00730 | 0.00732 |
| NVFP4, search -6..+5 codes | 0.00665 | 0.00684 | 0.00689 |
| MXFP4 at block 32, OCP floor rule | 0.01319 | 0.01587 | 0.01727 |
| MXFP4 at block 32, best of floor/ceil | 0.01243 | 0.01499 | 0.01620 |
| NF4 table (QLoRA quantiles), absmax | 0.00654 | 0.00670 | 0.00671 |
| Lloyd-Max 16-entry grid trained on x/amax | 0.00545 | 0.00628 | 0.00634 |
| the same grid snapped to 4 significant bits (E4M3-exact) | 0.00591 | 0.00635 | 0.00646 |
| the same grid snapped to 8 significant bits (BF16-exact) | 0.00545 | 0.00629 | 0.00633 |

| 5 and 6 bits, block 64 (FP16 scale) | Gaussian | Laplacian | Student-t(4) | bpw |
|---|---|---|---|---|
| Q3 -3..3 amax/3 (reference) | 0.0632 | 0.0998 | 0.1233 | 3.25 |
| Q4 -7..7 amax/7 | 0.01159 | 0.01985 | 0.02553 | 4.25 |
| INT5 -15..15 amax/15 | 0.00252 | 0.00439 | 0.00568 | 5.25 |
| INT5 with scale search | 0.00232 | 0.00412 | 0.00539 | 5.25 |
| FP6 E2M3 amax/7.5 | 0.00065 | 0.00068 | 0.00071 | 6.25 |
| FP6 E2M3 with scale search | 0.00051 | 0.00055 | 0.00060 | 6.25 |
| FP6 E3M2 amax/28 | 0.00250 | 0.00226 | 0.00210 | 6.25 |
| INT6 -31..31 amax/31 | 0.00059 | 0.00103 | 0.00134 | 6.25, not E4M3-exact |
| NVFP4 engine search at block 8 | 0.00578 | 0.00530 | 0.00523 | 5.0, no native MMA |

Block-size trend for searched NVFP4 (Gaussian): 0.00977 at block 64, 0.00869 at 32, 0.00737 at 16, 0.00578 at 8. Each halving
buys 11 to 22% MSE; the 16-to-8 step (22%) costs 0.5 bpw and has no hardware scale granularity (`scale_vec::4X` is per 16).

What the table says:

1. **Why NVFP4 beats MXFP4.** With the OCP rule (`floor(log2 amax) - 2`, which llama.cpp's `quantize_row_mxfp4_ref` implements
   with a doubled value table) a block whose amax has a binary mantissa above 1.5 is clipped by up to 1.33x; the upper power
   instead wastes the 4 and 6 codes. On synthetic weights MXFP4 carries 1.45 to 2.0x the MSE of round-to-nearest NVFP4 and 1.8
   to 2.4x that of searched NVFP4 at nearly equal bytes (4.25 against 4.5). This is the mechanism behind every published MXFP4
   deficit in section 3 and behind the "Bridging the Gap" finding that power-of-two scale quantization severely degrades
   accuracy (arXiv 2509.23202). SD4 keeps E4M3 granularity at MXFP4's byte cost: the right 4.25-bpw design for this hardware.
2. **Scale search is worth more than the grid.** Plain amax scaling leaves 25 to 32% of the MSE on the table; 4/6 and the
   engine's -6..+2 search each recover a little over half of it, -6..+5 (which contains amax/4, 4 to 5 E4M3 codes above RTN)
   about four fifths. The engine's search only lowers scales (clipping where importance is low); mapping a block maximum to 4
   instead of 6, so that the 4..6 gap does not swallow the second-largest values, is the half 4/6 adds.
3. **A non-uniform grid gains little on heavy tails.** The MSE-optimal 16-entry grid for Gaussian blocks of 16 is nearly uniform
   without a zero (magnitudes 0.047, 0.142, 0.242, 0.351, 0.472, 0.612, 0.777, 0.977 of amax); it beats searched E2M1 by 11% on
   Gaussian weights and by under 1% on Laplacian and Student-t weights, which is what LLM weights look like. NF4 is 5 to 6% worse
   than searched E2M1 at block 16. Snapping the optimal grid to E4M3-representable values costs 1 to 8% and keeps `decode_rows`
   exact in BF16 (4-bit grid x 4-bit E4M3 scale = 8 significant bits).
4. **E2M1 codes carry 3.90 bits because of amax scaling at block 16.** With the block maximum pinned to code 6, the sign is a
   full bit and the 8 magnitudes are nearly equiprobable: 3.899 bits for Gaussian blocks (3.82 Laplacian), the 3.90 measured in
   `docs/NVFP4_COMPRESSED.md`. INT4 at block 64 carries 3.50 (Gaussian) to 3.07 (Laplacian) bits, which is why P2E2-style entropy
   coding works for Q3 and not for NVFP4 codes.
5. **At 6 bits, E2M3 is the format.** E2M3 at block 64 has 4 to 8x lower MSE than Q5G64 and 1.5 to 1.9x lower than INT6 on heavy
   tails (INT6 is 10% better only on Gaussian weights); E3M2 is no better than INT5 on Gaussian weights (0.00250 against 0.00252
   at block 64, worse at block 16) and never better than E2M3, in line with the MX paper's direct-cast result (E2M3 26.38 BLEU
   against E3M2 25.97 on Transformer-Base, FP32 26.85; arXiv 2310.10537) and the AMD ROCm note that E2M3 typically wins once a
   shared scale covers the range. MicroMix (arXiv 2508.02343) finds the opposite for some models; that is about activations.
6. **Block-8 NVFP4 (5.0 bpw) does not replace Q5G64 (5.25 bpw):** worse on Gaussian and Laplacian weights, equal on
   Student-t(4), and without a native MMA. Code bits beat block bits once the block is 16.

### 2.3 What carries over from the pretraining recipes

NVIDIA's NVFP4 pretraining recipe (arXiv 2509.25149; 12 B parameters, 10 T tokens, MMLU-Pro 62.58 against FP8's 62.62) has four
parts: random 16 x 16 Hadamard transforms on the inputs of the weight-gradient GEMMs, stochastic rounding of gradients, 2D 16 x 16
block scaling of weights so forward and backward share one representation, and sensitive blocks kept in BF16. For post-training
weight quantization only the last transfers directly (per-tensor mixed precision, candidates B and D); 2D scaling serves the
transposed GEMM, 1D per-16 along K is what the inference MMA consumes; stochastic rounding is a gradient tool; the Hadamard
transform transfers to the activation side (rotations chapter). Quartet (arXiv 2505.14669) and Quartet II (arXiv 2601.22813) are
MXFP4/NVFP4 training methods built on QuEST-style clipping and unbiased gradients; their inference-relevant output is QuTLASS
(IST-DASLab/qutlass), which fuses rotation, quantization and scale computation for group sizes 16/32/64/128 on sm_120. UE5M3 scale
factors (arXiv 2609.02846) address scale underflow in training; for weights, E4M3 under the per-tensor FP32 global spans 17.8
octaves below the largest block, so PTQ needs nothing. The adaptive types IF4 (arXiv 2603.28765) and MixFP4 (arXiv 2605.31035)
choose FP4 or scaled INT4 per block and signal it in the unused sign bit of the E4M3 scale: zero metadata, but the native MMA
decodes E2M1 only, so INT4 blocks would need a software path; the Lloyd-Max row bounds the gain (about 10% MSE on Gaussian
blocks, none on heavy tails).

## 3. Published results

Values as shown in search summaries and repository READMEs; BF16 baselines where the source gave them.

| Model | Format | Metric | Result | Baseline | Source |
|---|---|---|---|---|---|
| Llama 3 1B / 8B / 70B | MXFP4 W4A4 RTN | WikiText-2 ppl | 17.67 / 9.66 / 5.53 | not retrieved | Four Over Six, arXiv 2512.02010 |
| Llama 3 1B / 8B / 70B | NVFP4 W4A4 RTN | WikiText-2 ppl | 14.27 / 8.43 / 4.00 | not retrieved | same |
| Qwen3 1.7B / 8B / 32B | MXFP4 W4A4 RTN | WikiText-2 ppl | 26.95 / 14.08 / 11.92 | not retrieved | same |
| Qwen3 1.7B / 8B / 32B | NVFP4 W4A4 RTN | WikiText-2 ppl | 23.06 / 12.68 / 9.85 | not retrieved | same; 4/6 lower "in most settings", values not retrieved |
| Llama and Qwen3 (averages) | NVFP4 / MXFP4 with MR-GPTQ | recovery | about 96.6% / 92.3% | BF16 | Bridging the Gap, arXiv 2509.23202 |
| Qwen3-32B | NVFP4 W4A4 (llm-compressor) | OpenLLM v1 average | 74.20 (99.83%); v2 97.79%, MMLU-Pro 94.01%, AIME24 90.93% | 74.32 | RedHatAI/Qwen3-32B-NVFP4 card |
| Qwen3-32B | NVFP4A16 (weights only) | OpenLLM v1 | 99.98% recovery; MMLU-Pro 94.73% | BF16 | RedHatAI/Qwen3-32B-NVFP4A16 card |
| Llama-3.3-70B-Instruct | NVFP4 W4A4 | OpenLLM v1 average | 78.57 (98.49%); GSM8K about 90%; MMLU-Pro 93.98% | 79.77 | RedHatAI/Llama-3.3-70B-Instruct-NVFP4 card |
| Llama-3.3-70B-Instruct | INT8 W8A8 | OpenLLM v1 | 99.4% recovery | BF16 | RedHatAI ...quantized.w8a8 card (separate run) |
| 70B to 235B models | NVFP4 | recovery | "consistently about 99%" | BF16 | Red Hat developers article, 2026-02-04 |
| Qwen3.8-27B (Unsloth NVFP4, llm-compressor) | NVFP4 | KLD on text; GSM8K (500, thinking off) | 0.0475; 97.2 | 97.0 | Qwen/Qwen3.8-27B HF discussion 192 |
| Qwen3.8-27B (Unsloth NVFP4) | NVFP4 | GSM8K; MMLU-Pro | 95.4; 78.9 (no pair CI-separated) | 95.5; 80.4 | 2026 arXiv paper, ID unverified |
| Qwen3-8B | SOAR over NVFP4 baseline | MMLU; GSM8K | +0.77; +3.26 points | NVFP4 | arXiv 2605.12245 |
| Llama, Qwen (W, A, KV, Q in NVFP4) | ScaleSweep | retained accuracy | above 93% | BF16 | arXiv 2606.07618 |
| 12 B Mamba-Transformer, 10 T tokens | NVFP4 pretraining | MMLU-Pro | 62.58 | 62.62 (FP8) | arXiv 2509.25149 |
| Transformer-Base | MXFP6 E2M3 / E3M2 direct cast | BLEU | 26.38 / 25.97 | 26.85 (FP32) | MX paper, arXiv 2310.10537 |
| FLUX.1-dev | NVFP4 W4A4 with / without SVDQuant | LPIPS (lower is better) | 0.205 / 0.345 | BF16 | deepcompressor README (arXiv 2411.05007) |
| Qwen3-8B | SVDQuant rank 32 vs H-SVDQuant rank 4, W4A4 | WikiText-2 ppl | 11.21 vs 10.44 | not retrieved | 2026 arXiv paper, ID unverified |
| LLaMA-3-8B | weight spectrum | singular values flat beyond r about 256 | | | TwinQuant, arXiv 2606.01556 |

No card for one checkpoint at NVFP4, FP8-dynamic and INT4-AWQ side by side was retrievable (huggingface.co is blocked here); the
like-for-like pair is Qwen3-32B W4A4 99.83% against W4A16 99.98%. The engine's own numbers are of another kind (teacher-forced
NLL and KL on 1279 tokens): `nvfp4` +0.029 nats, KL 0.147, top-1 88.2% against the base model's BF16; `nvfp4-compressed`
(Swift-1.5, UkisAI matrix) +0.028 +/- 0.009, KL 0.155, top-1 89.4%, perplexity 8.76 against 8.51; SD4 against plain NVFP4, both
base conversions, +0.005 +/- 0.009, KL 0.037. Unsloth's KLD of 0.0475 on text is a third of the engine's KL; corpora differ, but one
structural difference is that llm-compressor recipes keep `lm_head` in BF16 while the engine quantizes the LM head to NVFP4 and
the embedding to Q3 (section 6).

2:4 sparsity, one-shot and with mask training:

| Model | Method | WikiText-2 ppl (dense) | Source |
|---|---|---|---|
| LLaMA-2 7B / 13B / 70B | SparseGPT 2:4 | 10.17 / 8.32 / 5.40 (5.12 / 4.57 / 3.12) | Wanda README (locuslab/wanda) |
| LLaMA-2 7B / 13B / 70B | Wanda 2:4 | 11.02 / 8.27 / 5.16 | same |
| LLaMA-2 7B | MaskLLM, learned masks, 2 B tokens of C4 | 6.78 (SparseGPT 10.42; dense 5.12) | MaskLLM README (NVlabs/MaskLLM), arXiv 2409.17481 |
| LLaMA-3 8B / 3.1 8B | MaskLLM | 8.49 / 8.58 (SparseGPT 17.64 / 18.65; dense 5.75 / 5.89) | same |
| Llama-3-8B | MaskLLM reproduced by TEAL | 8.532 (dense 5.870), about 800 H100 hours | TEAL, arXiv 2408.14690 |
| LLaMA-3-70B | SparseGPT 2:4 with DenoiseRotator | gap to dense cut from 8.1 to 3.4 ppl points | DenoiseRotator (ID not retrieved) |
| Qwen2.5-7B | 2:4, zero-shot average | RoI 64.54, SparseGPT 62.82, Wanda 61.58 (dense 70.08) | Reservoir of Importance, arXiv 2608.23048 |
| Qwen3-32B (MLP only, 50% unstructured) | SparseGPT / Wanda | average 60.27 / 18.96 (dense 71.68) | SpenseGPT, arXiv 2606.10445 |

In nats, the one-shot 2:4 penalties are ln(5.40/3.12) = 0.55 (70B SparseGPT), 0.50 (70B Wanda) and 1.12 (Llama-3-8B
SparseGPT); the best learned masks still cost 0.28 (LLaMA-2 7B) to 0.39 (Llama-3 8B). The `q3` artifact's entire quantization
penalty is 0.170 nats at 3.25 bpw.

Two general results frame the above. Dettmers and Zettlemoyer (arXiv 2212.09720) found float and quantile 4-bit types beat
integer types and a block of 64 to 128 with 16-bit scales optimal in their study; the block-16 E4M3 design spends 0.25 bpw more
to go finer. "Scaling Laws for Precision" (arXiv 2411.04330) found post-training quantization damage grows with tokens per
parameter: a 27 B model trained on a very large corpus is on the hard end, which is why 4 bits costs +0.03 nats and 3.25 bits +0.17.

## 4. Fit with Euhedral

Candidates: **A** extended NVFP4/SD4 scale search; **B** Q4G64 `query_key` projections replaced by SD4 at equal bytes; **C** Q3
scale plane re-quantized to a per-tensor table ("Q3-ST"); **D** FP6 E2M3 G64 for `gdn/value_z` and `attention/gate_value` (FP8
G64 as fallback); **E** decode-only non-uniform 16-entry grid; **F** 2:4 sparse FP4; **G** SVDQuant branch; **H** single-term
FP4 activations in prefill.

| Constraint (00-baseline section 6) | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| 1 Row-exact 1-8 rows | unchanged | unchanged (SD4 `decode_rows` twins exist) | new GEMV and twins: a byte index and a 256-float shared table replace the binary16 load, FMA sequence otherwise identical | Q6-style FMA GEMV (nibble plane + 2-bit plane, as Q5's nibble + bit plane) with per-row twins | BF16 MMA twins stay exact only with an E4M3-snapped grid | sparse `HMMA.SP` (204 TFLOPS) with metadata per fragment: new kernels | one extra deterministic rank-r GEMV pair per linear, same sequence at 1-8 rows | not involved: verification never takes the native route |
| 2 Prefill at 9+ rows off the BF16 path | unchanged | unchanged (native OMMA) | MX route, group scale from the table | MX route unchanged: E2M3 expands exactly to E4M3 (64-entry producer table) | **lost**: no native MMA for a non-E2M1 grid; the MX route needs block 32 (a k32 MMA spans two block-16 scales) at 204 effective TFLOPS against 600-700 achieved | sparse OMMA at 1621 TFLOPS, new tile | branch GEMMs small at prefill | native route, one MMA per K step instead of two |
| 3 Decode instruction budget | none | none | a byte load and one `ld.shared` per group instead of a 16-bit load | about +1 instruction per code pair over Q5; the 223-register cliff at 4 rows is the risk | pair-table lookup as today | metadata gather per fragment | +2 r FMAs per output row and a dependent reduction | none |
| 4 Determinism | yes | yes | yes | yes | yes | yes | fixed order | yes |
| 5 Offline conversion, new ordinals | one constant (`SEARCH_ABOVE` 2 to 5) | recipe only (per-tensor formats already mix: q3 artifacts carry NVFP4 MTP tensors) | new layout ordinal; table by DP or log-domain k-means, codes re-rounded | new format ordinal, two planes | new format plus table | SparseGPT pass plus sparse packing | SVD plus smoothing | unchanged |
| 6 Artifact decides | yes | yes | yes | yes | yes | yes | yes | route choice already in `ArtifactProfile` |
| 7 Memory per token | 0 | 0 | -0.37 GiB (q3) | +0.42 GiB (E2M3) or +1.26 GiB (FP8) | 0 | -1.8 to -2.5 GiB against q3 | +0.40 GiB (r = 32) | 0 |
| 8 Drafting separate | applies to the MTP layer too | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

### Bytes per token and resident memory

| Form | Streamed GiB per token | File GiB | Note |
|---|---|---|---|
| `q3` today | 10.7 | 11.72 | runs 128K with nothing host-backed |
| `q3` + C (8-bit scale index on every grouped tensor) | 10.33 | about 11.35 | 25.6 B x 0.125 bpw = 0.37 GiB; a 10-bit index saves 0.28 GiB |
| `q3` + B (`query_key` Q4 to SD4) | 10.7 | 11.72 | identical bytes |
| `q3` + D (E2M3 G64 on `value_z`, `gate_value`) | 11.1 | about 12.14 | +1.0 bpw on 3.61 B params; at 128K (KV 2.25 GiB) host-backing may start, 2 to 2.5% of decode per 128 MiB |
| `q3` + D with FP8 G64 | 11.96 | about 12.98 | +3.0 bpw on 3.61 B params |
| `nvfp4-compressed` today | 12.7 | 13.77 | |
| sparse NVFP4 (F) | 8.9 (3.0 bpw) / 8.2 (2.75 with SD4 table) | about 10 | quality in section 3 |
| `nvfp4` + G (r = 32) | 13.8 | 14.9 | branch read every token |

### Expected quality

- **A.** Synthetic MSE 6 to 10% below today's search. The SD4 "2 codes below" experiment moved NLL by -0.003 +/- 0.009 for a
  comparable MSE change, so A is not resolvable on the 1279-token document (s.e. 0.008 to 0.009 for NVFP4 forms); it is a free
  change to carry into any larger evaluation. Scales only move up, so clipping cannot grow, but the importance-weighted error
  must judge it: a higher scale zeroes small values in important columns.
- **B.** Same bytes; synthetic MSE 0.0099 to about 0.0075 (Gaussian, -25%) and 0.0174 to about 0.0073 (Laplacian, -58%), and the
  native prefill route for these tensors (Q4 runs the MX route at 117 to 125 TFLOPS today). Expected NLL gain on `q3` is small,
  because the Q4 tensors are 1.6 B of 25.6 B parameters and were given 4 bits for sensitivity; the step-1 diagnostic settles it.
- **C.** Own simulation on a 2048 x 5120 Student-t(5) tensor with a 4-octave row-norm spread (9 octaves of group scales, 6238
  distinct binary16 values, 12.2 bits of scale entropy, more than the 9.1 bits the engine's 43% compression implies for real
  tensors): with codes re-rounded under the table scale and each group taking its least-error entry near the nearest, relative
  MSE rises +0.36% at 256 entries (8-bit index), +0.18% at 1024 (10-bit), +1.65% at 64 (6-bit); nearest-entry assignment
  without re-rounding costs about twice that. Under the linearity theorem (arXiv 2411.17525: layer-wise L2 error adds linearly
  to loss in this regime) +0.4% of a +0.170-nat penalty is about +0.001 nats, the size of SD4's measured +0.005 +/- 0.009. The
  table must contain each tensor's largest scale (the SD4 lesson: a frequency table clipped outlier blocks 8x and cost 2 nats).
  P2E2 carries the same scales byte for byte, so `q3-compressed` takes the same table and index plane.
- **D.** MSE on those tensors falls 4 to 8x. Whether NLL moves depends on their share of the `q3` penalty, which is unknown: the
  MLP at Q3 has 10 to 20x their relative MSE on 67% of the bytes, so unless the input projections' sensitivity (input Hessian
  trace) is far higher, the gain is under +0.02 nats and unmeasurable on one document. In the NVFP4 forms the same tensors sit at
  4.5 bpw; FP6 there costs +1.75 bpw x 3.61 B = 0.74 GiB, too much at 64K unless only `attention/gate_value` (0.59 B, +0.12 GiB)
  proves sensitive. FP8 G64 (+1.26 GiB on `q3`) is justified only if FP6 fails the diagnostic, which the MSE table makes unlikely.
- **E.** 0 to 11% MSE on top of searched E2M1 for a 3 to 5x slower prefill on the affected tensors (TTFT at 128K is already 95 to
  133 s). Not worth it.
- **F.** 0.5 to 1.1 nats one-shot at 8 to 70 B; 0.3 to 0.4 nats with learned masks that took about 2 B tokens and hundreds of
  H100 hours for 7 to 8 B models. Dense Q3 at 3.25 bpw (+0.170) beats sparse FP4 at 3.0 bpw by a wide margin, and no 2025-2026
  one-shot method (FISTAPruner, OPTIMA, RoI, ProxSparse, SparseForge) reports 2:4 within 0.1 nats on a 27 to 70 B model. The
  1621-TFLOPS MMA helps prefill, which is not the bottleneck. Verdict as in `tensor-cores.md`: research, not a win, and not
  reachable on one 16 GB GPU.
- **G.** Built for W4A4 diffusion transformers, where smoothing moves activation outliers into the weights and a rank-32 branch
  absorbs them (FLUX: LPIPS halved). For weight-only LLM decode the branch only absorbs weight outliers, and the LLM weight
  spectrum is flat beyond rank 256 (TwinQuant), which matches the one LLM data point found (Qwen3-8B, rank-32 SVDQuant 11.21 ppl
  at W4A4, beaten by a Hessian-aware rank-4 variant). Cost here: +0.40 GiB per token and an extra dependent GEMV pair per linear.
  Its place is the activation side (H), as an alternative to rotation for outlier channels, not the weight path.
- **H.** The model cards show per-token-dynamic E4M3 block scales with a calibrated per-tensor activation scale (llm-compressor:
  20 calibration samples; ModelOpt's `NVFP4ActHeadroomCalibrator` anchors it to a low percentile with headroom) reach 98.5 to
  99.8% recovery at W4A4 on 32 to 70 B models. The engine's one-term attempt (output RMS error 9.5% against 0.88%, model KL 6e-3
  against a 3.1e-3 floor) was judged by the drift window; 6e-3 is 4% of the weight-quantization KL of 0.147. A second attempt
  needs a Hadamard rotation of size 16 per block (rotations chapter; "Bridging the Gap" finds the rotation size should match the
  group for NVFP4), per-block scale choice at quantization time (4/6, or the engine's six-candidate search, which gave 8.8%
  against 9.5%), SmoothQuant-style per-channel scaling folded into the weights, and a gate on teacher-forced KL rather than the
  window bound. The prize is about a third of the gate_up tile time (1.573 against 2.405 ms at 1024 rows).

### Interaction with MTP, DFlash2 and the row-exact verifier

None of A to D touches verification: it runs the 1-8 row decode kernels, which A and B leave unchanged and C and D replace by
another deterministic FMA sequence whose twins repeat it per row. `ArtifactProfile`'s depth (2 on Q3 forms, 4 on NVFP4 forms)
follows acceptance, which only drafting precision moves; A applies to the MTP layer as well. C changes the P2E2 fused decode and
its prefill expansion (shared scale plane), so `q3-compressed` gets the same ordinal change. H changes prefill numerics and the
drift test only.

## 5. Experiment plan

Ordered by information per hour. Harness: `00-baseline.md` section 9 (teacher-forced NLL, paired dNLL, KL, top-1;
`compare_teacher_forced.py` for pairs; `llama-perplexity --kl-divergence` for the llama.cpp arm). Differences under 0.02 nats
(NVFP4 forms) or 0.05 (Q3 forms) need the larger held-out set.

1. **Per-family sensitivity sweep (diagnostic for B and D, no engine code).** With the user's llama.cpp tree, quantize the base
   GGUF with `llama-quantize --tensor-type` overrides: everything at the `q3`-equivalent mix, then one family at a time raised to
   F16 (`gdn/value_z`, `attention/gate_value`, `gdn/query_key`, `attention/query_key`, `mlp/down`, LM head). Measure KL against
   BF16 on the chat document and a second corpus. Output: each family's share of the 0.293-nat KL. Go for D if the Q5 families
   own more than 0.02 nats, for B if the Q4 families own more than 0.01. One afternoon.
2. **A: `SEARCH_ABOVE` 2 to 5** in `nvfp4.py` (the SD4 cost window already reaches 40 codes). Reconvert `nvfp4-compressed`,
   run the teacher-forced pair against today's artifact, record acceptance. Keep if dNLL <= 0 within s.e.; a measurable gain is
   not expected. Converter minutes plus one test run.
3. **B: `q3` recipe with SD4 `query_key` tensors.** Recipe change only; verify the mixed-format layer graph loads and the
   exactness tests pass; measure dNLL against `q3` and prefill time. Go: dNLL below zero by more than 1 s.e. on two documents, or
   equal quality with faster prefill. One day.
4. **C: Q3 scale table.** First converter-only: for every Q3/Q4/Q5 tensor build the 256-entry table, re-round, report the weighted
   MSE increase and the largest clipped weight (expected under 1% and none). Then the layout ordinal, GEMV and twins, MX producer,
   P2E2 variant and scalar reference, with SD4's bit-for-bit expansion test. Go: MSE study under 1% on every tensor, decode bytes
   -0.37 GiB measured as tok/s at 4K and 128K, dNLL within noise, exactness tests green. One to two weeks; the saving is 3.5% of
   decode time or 0.37 GiB of context margin (about 21K tokens of KV).
5. **D: E2M3 G64 for `gdn/value_z` and `attention/gate_value`**, only if step 1 says the Q5 families matter. Converter: E2M3
   encode with the existing group search (candidate scales amax/(7.5 + s), least-squares refit, binary16), nibble plane plus 2-bit
   plane. Kernels: Q6 GEMV and twins from the Q5 code, the 64-entry E4M3 expansion in the MX producer, scalar reference. Go: dNLL
   recovers most of what F16 recovered in step 1, decode slowdown under 4%, 128K resident or host-backing under 256 MiB. Two weeks.
6. **H: one activation term, second attempt**, once the rotations chapter's kernels exist: rotated blocks, per-block scale choice,
   headroom-calibrated global scale, judged by teacher-forced KL at 512 + 1024 against the two-term route. Go: KL within 1.5x of
   the two-term floor (about 5e-3) and no top-1 loss beyond noise.
7. **Not planned:** E (grid), F (sparsity), G (SVDQuant on the weight path), FP8 weights unless D fails, block-8 NVFP4.

## 6. Risks and open questions

- **Synthetic weights are not these weights.** Section 2.2 ranks formats for Gaussian, Laplacian and Student-t(4) blocks; real
  tensors have column structure (which the imatrix weights) and rows with 4 or more octaves of norm spread. The converter can
  reproduce every row of the table on the actual tensors in minutes; that study should precede any kernel work.
- **MSE is a proxy.** SD4 lowered squared error 15 to 25% and moved NLL by +0.005 +/- 0.009; "2 codes below" lowered it further
  and moved NLL by -0.003. Nothing in A to C is expected to be resolvable on one document; their quality claims are bounds, and
  the larger held-out set is a prerequisite for closing them.
- **The LM head and the embedding.** Published NVFP4 recipes keep `lm_head` in BF16 and report KLDs a third of the engine's; the
  engine's NVFP4 LM head (0.67 GiB) and Q3 embedding belong in the step-1 sweep. A BF16 head costs 2.37 GiB and is impossible
  here; an FP8 head +0.51 GiB and a new GEMV; an E2M3 head +0.26 GiB with the step-5 kernels.
- **Importance matrix mismatch.** Swift-1.5 is converted with UkisAI's matrix, the base forms with ISTA-DASLab's. A scale above
  RTN zeroes small values in the block, which only the weights can judge: compare A with and without `--imatrix`.
- **FP6 decode registers.** The Q5 twins already sit at the 223-register cliff at 4 rows; a 2-bit plane adds live state. If the
  4-row twin slows beyond 4%, D is a decode regression at 64K, where verification is 73% of the step.
- **Sparse pattern.** The pairwise 4:8 reading of the FP4 sparse MMA is inferred from CUTLASS types and two secondary sources;
  confirm against the PTX ISA before citing 3.0 bpw as a hard number. It does not change the verdict.
- **Not read in full:** Four Over Six's own tables (4/6 against NVFP4 against BF16), the llm-compressor result that an expanded
  MSE search beats 4/6 (PR 2950, referenced in `observers/mse.py`), and arXiv 2609.04098 ("Why Gated DeltaNet Survives 4-Bit
  Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM"), the first paper to check before step 6 since it covers
  the GDN activations the engine would quantize. A does not depend on the first two, since it subsumes the 4/6 candidate.
- **KV cache and attention in FP4** are in the KV chapter; the per-16 E4M3 scaling and the H256 rotation there are the same
  mechanisms as section 2.2 and the rotations chapter.

## 7. References

1. Cook, Guo, Xiao, Lin, Han. Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling. arXiv 2512.02010
   (2025). Code: https://github.com/mit-han-lab/fouroversix (`ScaleRule.static_6`, `static_4`, `mse`).
2. Cook et al. Adaptive Block-Scaled Data Types. arXiv 2603.28765 (2026). MixFP4, arXiv 2605.31035 (2026).
3. NVIDIA. Pretraining Large Language Models with NVFP4. arXiv 2509.25149 (2025).
4. Castro, Panferov et al. Quartet: Native FP4 Training Can Be Optimal for Large Language Models. arXiv 2505.14669 (NeurIPS
   2025), https://github.com/IST-DASLab/Quartet. Quartet II, arXiv 2601.22813 (2026).
5. Egiazarian et al. Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization. arXiv 2509.23202 (2025).
   https://github.com/IST-DASLab/FP-Quant, https://github.com/IST-DASLab/qutlass.
6. ScaleSweep, arXiv 2606.07618 (2026). Bao et al., SOAR, arXiv 2605.12245 (2026), https://github.com/steven-bao1/SOAR.
   OCGQuant, arXiv 2609.00066; ARCQuant, arXiv 2601.07475; DuQuant++, arXiv 2604.17789; RaZeR, arXiv 2501.04052; Benford prior
   for PTQ, arXiv 2602.00165; A Comprehensive Evaluation on Quantization Techniques for LLMs, arXiv 2507.17417 (titles and IDs
   from search results; not read). UE5M3 FP4 Block Scaling, arXiv 2609.02846 (2026).
7. Malinovskii et al. Pushing the Limits of LLM Quantization via the Linearity Theorem (HIGGS). arXiv 2411.17525 (NAACL 2025).
   Grid Games, arXiv 2605.12327 (2026).
8. Dettmers, Zettlemoyer. The case for 4-bit precision: k-bit Inference Scaling Laws. arXiv 2212.09720 (ICML 2023).
9. Kumar et al. Scaling Laws for Precision. arXiv 2411.04330 (ICLR 2025).
10. Darvish Rouhani et al. Microscaling Data Formats for Deep Learning. arXiv 2310.10537 (2023). OCP Microscaling Formats (MX)
    Specification v1.0 (September 2023), as described at https://fprox.substack.com/p/ocp-mx-scaling-formats.
11. MicroMix, arXiv 2508.02343; MX+, arXiv 2510.14557; AMD ROCm blog, High-Accuracy MXFP4, MXFP6, and Mixed-Precision Models on
    AMD GPUs. ZeroQuant(4+2), arXiv 2312.08583 (2023); FP6-LLM kernels, https://github.com/usyd-fsalab/fp6_llm.
12. Li et al. SVDQuant. arXiv 2411.05007 (ICLR 2025); https://github.com/mit-han-lab/deepcompressor. TwinQuant, arXiv 2606.01556.
13. Frantar, Alistarh. SparseGPT. arXiv 2301.00774 (2023), https://github.com/IST-DASLab/sparsegpt. Sun et al. Wanda. arXiv
    2306.11695 (ICLR 2024), https://github.com/locuslab/wanda. Fang et al. MaskLLM. arXiv 2409.17481 (NeurIPS 2024),
    https://github.com/NVlabs/MaskLLM. TEAL, arXiv 2408.14690. Reservoir of Importance, arXiv 2608.23048. FISTAPruner, EMNLP 2025
    (aclanthology 2025.emnlp-main.1498). OPTIMA, arXiv 2512.13886. ProxSparse, arXiv 2502.00258. SparseForge, arXiv 2605.06402.
    SpenseGPT, arXiv 2606.10445. SharQ, arXiv 2606.26587.
14. Model cards (via search summaries): https://huggingface.co/RedHatAI/Qwen3-32B-NVFP4, .../Qwen3-32B-NVFP4A16,
    .../Llama-3.3-70B-Instruct-NVFP4, .../Llama-3.3-70B-Instruct-quantized.w8a8. Red Hat Developers, Accelerating large language
    models with NVFP4 quantization (2026-02-04). Qwen/Qwen3.8-27B HF discussion 192 ("NVFP4 Shootout"). Unsloth,
    https://unsloth.ai/docs/models/qwen3.8. arXiv 2609.04098; arXiv 2601.09527.
15. NVIDIA TensorRT Model Optimizer (commit 54d4416, 2026-10-10): `qtensor/nvfp4_tensor.py` (global scale amax/(6 x 448), E4M3
    scale clamp to [2^-9, 448]); `calib/mse.py` (`NVFP4MSECalibrator`: sweep of the 126 valid E4M3 scales per block, plain or
    local-Hessian weighted, with `four_over_six` folded into the amax); `calib/nvfp4_act_headroom.py`; `config.py`
    (`fp8_scale_sweep`, "recommended for NVFP4").
16. vllm-project/llm-compressor (commit 2297585, 2026-10-09): `observers/mse.py` (`nvfp4_expanded_mse`: 0.8x to 1.8x of the
    group range in 112 steps, replacing the `fouroversix` observer, PR 2950); `observers/imatrix.py`; `entrypoints/higgs/` (ILP
    per-tensor scheme selection); `examples/quantization_non_uniform/quantization_nvfp4_fp8.py`.
17. NVIDIA/cutlass (commit 0b55a2f): `include/cute/atom/mma_traits_sm120.hpp`, `mma_traits_sm120_sparse.hpp`,
    `include/cute/arch/mma_sm120_sparse.hpp`, `include/cute/arch/copy_sm90_desc.hpp`, `include/cute/pointer_sparse.hpp`,
    `examples/79_blackwell_geforce_gemm/`.
18. ggml-org/llama.cpp (commit 23b0202, 2026-10-10): `ggml/src/ggml-common.h` (`block_mxfp4`: 1 E8M0 byte + 16 code bytes per
    32; `block_nvfp4`: 4 UE4M3 bytes + 32 code bytes per 64, no per-tensor global); `ggml/src/ggml-quants.c`.
19. Euhedral (branch `perf/mtp-gated-depth`, commit 8e73f0a): `docs/NVFP4_NATIVE.md`, `docs/NVFP4_COMPRESSED.md`,
    `docs/PREFILL_MX.md`, `docs/COMPRESSED_Q3.md`, `docs/nvidia/tensor-cores.md`, `tools/euhedral_artifacts/nvfp4.py`,
    `tools/euhedral_artifacts/grouped.py`, `native/src/nvfp4/nvfp4.cuh`, `native/src/nvfp4/scale_table.cuh`.
20. Own computations for this chapter: `grid_mse.py` and `q3_scale_table.py` (scratchpad; NumPy, seeds 12345 and 7).
