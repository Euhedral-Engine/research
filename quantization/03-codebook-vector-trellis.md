# Codebook, vector and trellis quantization at 2.0 to 3.5 bits per weight

Scope: formats whose stored symbol is an index into a codebook (scalar lookup tables, lattice and vector codebooks, additive
codebooks) or a state of a trellis whose values are computed from the index, as a possible fifth form for Euhedral at 2.5 to 3.5
bpw. Rotations are the rotations chapter's subject and fine-tuning procedures the fine-tuning chapter's; this chapter reports
quality with and without them where the sources separate the two. Everything is measured against `00-baseline.md` (sections 1
to 9). Numbers marked "estimate" are this chapter's own arithmetic, not measurements.

## 1. Summary

The family splits into three decode schemes: small lookup tables (NF3/NF4, SqueezeLLM, Any-Precision, llama.cpp's IQ4_NL and its
IQ2/IQ3 lattice grids, QuIP#'s E8P, QTIP's HYB), large per-layer tables (AQLM, VPTQ, GPTVQ) and computed trellis codes (QTIP's
1MAD/3INST, EXL3's mul1/mcg, ik_llama.cpp's integer KT). The best-supported rate-distortion evidence for this exact model is a
same-harness KL table for Qwen3.8-27B: EXL3 at 2.5 / 3.0 / 3.5 / 4.0 bpw reaches KL 0.075 / 0.045 / 0.023 / 0.016 with top-1
agreement 90.0 / 92.7 / 94.7 / 96.2% against BF16, where the community NVFP4 checkpoints of the same model sit at KL 0.037 to
0.065 and top-1 89 to 93% [22]. On Llama-3.1-8B and 70B, EXL3 without any fine-tuning matches GGUF IQ quants that spend 0.5 to
0.8 bpw more, and matches or beats fine-tuned AQLM and PV-tuned AQLM at 2.0 to 2.3 bpw [19]. Most of that gain is the rounding
(Hadamard incoherence plus LDLQ with a full input Hessian), not the codebook: ik_llama.cpp's KT trellis with imatrix rounding and
no rotation buys only 0.1 to 0.2 bpw over its own IQ lattice grids [20]. A 2.5 bpw form of EXL3 quality would beat today's
calibrated `q3` (+0.170 nats, KL 0.293, top-1 76.2%, 3.25 bpw) by a wide margin even after allowing for the harder document the
repository measures on, and a 3.0 to 3.5 bpw form would land at or inside NVFP4's +0.029 nats at 4.5 bpw (estimate, section
4.6). Verdict: worth building, as an integer-valued trellis (EXL3's structure with an 8-significant-bit generator) decoded on BF16
tensor cores the way `nvfp4::decode_rows` already works, because at 2.5 bpw the FP32 FMA GEMV must decode 1.3x faster than Q3's
and has no instruction headroom left (section 4.1, constraint 3). The lattice-LUT alternative (IQ3_S-style D4 grid or E8P) is the
fallback if the trellis's prefill expansion costs too much: its entries are exactly E4M3 so the MXFP8 prefill route takes them
as they are. Cost: a Hessian-collecting converter (about 2 hours per conversion on a fast GPU, longer on the 5070 Ti), a one-row
plus multi-row MMA decode family, one expansion kernel, one EDRL format ordinal; roughly 4 to 8 engineer-weeks (estimate). The
decisive experiments need no kernels: same-harness KL of IQ quants and EXL3 conversions on the repository's document, then a
Python reference of the format written back into a BF16 GGUF and scored with `llama-perplexity --kl-divergence`.

## 2. Mechanism

### 2.1 Scalar lookup tables

- **NF3/NF4 (QLoRA) and NFL (FLUTE)**: 8 or 16 entries at the quantiles of a normal distribution, one absmax scale per group of
  64; FLUTE's NFL learns the table and scales by straight-through estimation on calibration data and runs them on a lookup-table
  GEMM kernel (2, 3, 4 bits, groups 32 to 256) [8]. Decode: one table byte or register select per weight, times the group scale.
- **SqueezeLLM**: one non-uniform 3-bit or 4-bit table per output channel from k-means weighted by the diagonal Fisher
  information, plus 0.05 to 0.45% of weights kept in a sparse FP16 matrix (CSR) that adds a sparse matvec; 3-bit plus 0.45%
  sparsity is 3.24 bpw [7]. **Any-Precision LLM** reuses the clustering, quantizes a 3-bit seed and splits every cluster in two per
  extra bit, so one 8-bit tensor holds nested 3 to 8-bit models [9].
- **IQ4_NL / IQ4_XS (llama.cpp)**: 16 fixed values `-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113` with a
  binary16 scale per 32 (4.5 bpw) or a binary16 per 256 with 6-bit sub-scales per 32 (4.25 bpw) [21].
- **QTIP HYB**: a 512-entry 2-D table (`tlut_bits` Q = 9, V = 2 weights per entry) from k-means on N(0, I2), indexed by the trellis
  state through `idx * (idx + 1) >> (16 - Q - 1)` with the sign of the first component taken from bit 15 [2].

### 2.2 Lattice and vector codebooks

- **QuIP# E8P** [1]: 8-dimensional codebook of 2^16 entries built from the E8 lattice: 227 points of D8 + 1/2 inside radius
  sqrt(10) plus 29 norm-12 points give 256 absolute patterns (one 32-bit word of nibbles each), times 2^7 sign flips (the eighth
  sign is the parity) and a +1/4 or -1/4 shift; entries are odd multiples of 1/4 up to 15/4. Two bits per weight, a per-matrix
  scale, no group scales; 3 and 4 bits add residual E8P codebooks (RVQ). Weights are first made incoherent with a randomized
  Hadamard transform and rounded with LDLQ (block error feedback against the input Hessian), then fine-tuned. Decode: one lookup
  of the 1 KiB absolute grid per 8 weights, a sign mask, a shift.
- **llama.cpp IQ2_XXS / IQ2_XS / IQ2_S** [21]: 8-dimensional grids of 256 / 512 / 1024 points whose coordinates take only the
  three magnitudes 8, 25, 43 (stored as bytes, applied with a 1/4 factor), 7 sign bits plus parity (IQ2_XXS, IQ2_XS) or 8 explicit
  sign bits (IQ2_S), 4-bit scales per 32 or 16 and a binary16 per 256: 2.0625 / 2.3125 / 2.5625 bpw. **IQ3_XXS / IQ3_S**: 4-D grids
  of 256 / 512 points ("D4 lattice" in the source), coordinates 4, 12, ..., 52, 62 and the odd integers 1 to 15 respectively,
  7 or 8 sign bits per 8 weights, 4-bit scales: 3.0625 / 3.4375 bpw. Rounding is imatrix-weighted nearest-grid search with a scale
  search, no cross-column error feedback.
- **AQLM** [3]: each group of 8 weights is the sum of M codebook vectors, codebooks of 2^16 (1x16) or 2^8 (2x8) 8-dimensional
  FP16 entries learned per layer (1 MiB per codebook), beam search encoding, then fine-tuning; **PV-Tuning** [4] fine-tunes the
  discrete codes as well as the continuous parameters.
- **VPTQ** [5]: vector length 8 (or 12, 16), a 65536-entry main codebook plus a residual codebook of 256 to 65536 entries per
  layer; second-order (GPTQ-style) Hessian error feedback drives a k-means codebook; naming `v8-k65536-256` means 16 + 8 bits per
  8 weights = 3 bpw. **GPTVQ** [6] is 1-, 2- and 4-dimensional VQ with GPTQ error feedback and small per-block codebooks (2.125 to
  2.25 bpw for 2-D). **GLVQ** (NeurIPS 2025) learns one lattice generator matrix per group of weights; decoding a group is a small
  matrix-vector product [12]. **Tetra** (2026) uses the 24-dimensional Leech lattice: 48 bits per 24 weights through a 64-state
  Golay trellis and one 16 KiB table [15].

### 2.3 Trellis codes

- **QTIP** [2]: a "bitshift trellis" with L = 16 state bits and K bits per weight; the state of weight j is the 16-bit window at bit
  offset jK of the packed stream, so consecutive weights share 16 - K bits of state and the code is a 2^16-state trellis
  with 2^K branches. Encoding is Viterbi over all 2^16 states (tail-biting; the paper's "overlap" trick halves the work),
  applied inside LDLQ on 16 x 16 tiles after a randomized Hadamard transform. Values come from the state by computation, not
  storage: **1MAD** `x = x * 34038481 + 76625530 mod 2^32; v = (sum of the 4 bytes - 510) / 147.8`; **3INST**
  `x = x * 89226354 + 64248484; v = fp16(x & 0x8fff8fff ^ 0x3b603b60).low + .high`; **HYB** as in 2.1. No group scales: one scale
  per matrix, sign vectors on inputs and outputs. The paper's kernels (HYB, L = 16, Q = 9, V = 2, 2/3/4 bits) keep the 2 KiB table
  in shared memory replicated 32 times so that lanes hit distinct banks, and feed `mma.m16n8k16` FP16 x FP16 with FP32 accumulate.
- **EXL3** [19]: QTIP's trellis with 16 x 16 tiles (256 weights, 16K uint16 words per tile), a 128-point block Hadamard on both
  sides (`had_k = had_n = 128`) with FP16 sign vectors `suh`/`svh`, no per-tensor scale ("scale is no longer used"; the codebook's
  RMS is 1.2437), and three computed codebooks: `3inst` (QTIP's), `mcg` (`x *= 0xCBAC1FED`, same mask trick), and the default
  `mul1`: `x *= 0x83DCD12D; n = dp4a(x, 0x01010101, 0x6400)` which lands the byte sum in an FP16 mantissa, then
  `v = fp16(n) * (1/147.7) - 10.39`, i.e. `(sum - 510)/147.7`. mul1 also permits half-integer bitrates (2.5, 3.5) by alternating K
  with period 16, so a 2.5 bpw EXL3 tensor is exactly 2.5 bpw. The converter computes Hessians on the fly from 250 rows x 2048
  calibration tokens, regularizes the diagonal by 2.5% of its mean, runs LDLQ in 16-row steps with a fused Viterbi kernel, and
  can take YAQA Kronecker Hessians for two-sided LDLQ (`ldlq_2hess`). The LM head defaults to 6 bits, MTP layers to 4 bits
  (uncalibrated), embeddings to 16 bits. The GEMV kernel decodes two weights with one funnel shift or BFE per 16-bit window, one
  IMUL, one IDP4A, a pack and one HFMA2, issues one `mma.m16n8k16` pair per 16 x 16 tile **with FP16 accumulation folded to FP32 on
  a fixed cadence**, covers 1 to 8 token rows (`EXL3_GEMV_MAX_M 8`), and runs the input Hadamard under a cooperative grid barrier.
- **ik_llama.cpp IQn_KT** [20]: not a sliding-window trellis. One 16-bit seed per group of 8 weights (IQ2_KT, IQ3_KT) or one 15-bit
  seed per group of 4 (IQ4_KT) generates the group by iterating `x = 0xCBAC1FED * x; v = dp4a(x & 0x3f3f3f3f, 0x01010101, -126)`,
  an integer in [-126, 126] (the first implementation used QTIP's FP16 3INST; the integer generator replaced it in PR 505/511/529
  for speed and for exact int8 dot products). Block scales: 4-bit index into a 16-entry table per 32 weights (IQ2_KT, IQ3_KT; IQ3_KT
  stores |v| and one explicit sign bit per weight), 8-bit per 32 for IQ4_KT, one FP32 per row: 2.125 / 3.125 / 4.0 bpw (IQ1_KT:
  1.75). Encoding: imatrix-weighted search over k-means clusters of the 2^16 generated 8-vectors plus neighbours, no Hadamard, no
  LDLQ. CUDA decode is `IMUL, LOP3, IDP4A` per weight and the GEMV dots the int8 values against Q8_1 activations with `dp4a`.
- **2026 follow-ups** (abstracts only): **Proteus** (ICML 2026) is a lookup-free trellis with an integer-mixing generator and a
  per-layer affine Gaussianizer, reporting 2-bit Llama-2 7B to 70B PTQ above QTIP with lower decode bandwidth [16]; **XOR-Trellis**
  (arXiv 2610.00432, 2026-09-30) replaces the state-to-value table by shallow XORs and a small FP4 lookup, i.e. an E2M1-valued
  trellis, and a curvature-aware Viterbi without Hadamard (Llama-3.1-8B WikiText-2 8.00 against 8.69 for an unrotated trellis)
  [14]; **CurveTQ** (arXiv 2610.09212) is a rotation-free trellis with curvature-weighted search [17]; **GuidedQuant** (ICML 2025)
  weights any layer-wise PTQ objective by end-loss gradients [10]; **YAQA** (2025) rounds against a Kronecker-factored full-model
  Hessian and reports about 30% lower KL than LDLQ across quantizers including QTIP [11]; **Q-Palette** (NeurIPS 2025) is a library
  of fractional-bit NUQ/VQ/TCQ quantizers for optimal bit allocation [13].

### 2.4 Comparison

Decode operations are per weight pair, from the sources' kernels where published and otherwise estimates from the decode
formulas (marked e). "Entry bits" is the number of significant bits a decoded value needs before any scale (section 4.2).

| Scheme | Example | bpw (codes + scales) | Codebook storage | Encoding | Decode ops per pair | Entry bits |
|---|---|---|---|---|---|---|
| Scalar LUT, 8 to 16 entries | NF3/NF4, SqueezeLLM, IQ4_NL | 3.0 to 4.5 | 16 to 32 B (registers) | nearest level, k-means | 2 to 3 (e): index extract, PRMT/LDS, scale | arbitrary (NF), 8 (IQ4_NL) |
| 4-D lattice grid | IQ3_XXS / IQ3_S | 3.06 / 3.44 | 1 / 2 KiB | nearest grid point, imatrix | 2 to 3 (e): 1 LDS per 4, sign XOR | 4 (odd ints <= 15) |
| 8-D lattice grid | IQ2_XXS/XS/S, E8P | 2.06 to 2.56; 2.0 | 1 to 8 KiB | nearest grid point; E8 rounding in LDLQ | about 3 (e): 1 LDS per 8, sign mask | 6 (8, 25, 43); 4 (E8P x 4) |
| QTIP HYB | L=16, Q=9, V=2 | 2, 3, 4 (no scales) | 2 KiB, replicated 32x = 64 KiB | Viterbi 2^16 states in LDLQ | about 6: shift, IMAD, mask, LDS, XOR [2] | arbitrary FP16 |
| Computed trellis, FP16 | QTIP 3INST, EXL3 mcg | 2 to 8, half steps (mul1) | none | Viterbi in LDLQ | about 9 (e): 2 BFE, 2 IMUL, 2 LOP3, 2 PRMT, 1 HADD2 | 11+ (sum of two FP16) |
| Computed trellis, integer | EXL3 mul1 | 2 to 8, half steps | none | Viterbi in LDLQ | about 8 (e): 2 BFE, 2 IMUL, 2 IDP4A, 1 PRMT, 1 HFMA2 | 10 (n - 510) |
| Seeded integer group | ik IQ2_KT / IQ3_KT / IQ4_KT | 2.125 / 3.125 / 4.0 | none | cluster search, imatrix | 6 (e): 2 IMUL, 2 LOP3, 2 IDP4A (serial chain of 8) | 8 (-126..126) |
| Additive VQ | AQLM 1x16, 2x8 | 2.0 + scales | 1 MiB per codebook per layer | beam search + fine-tuning | M global/L2 loads of 16 B per 8 weights, M-1 adds | arbitrary FP16 |
| Vector + residual VQ | VPTQ v8-k65536-k | 2 to 4 | 1 MiB + residual per layer | Hessian-weighted k-means | 2 L2 loads of 16 B per 8 weights, add | arbitrary FP16 |
| Leech lattice + Golay trellis | Tetra | 2.0 + scales (2.7 whole model) | 16 KiB | lattice decoding | 64-state trellis walk per 24 weights | arbitrary |

## 3. Published results

### 3.1 Llama-2, WikiText-2 perplexity at 2 bits (context 4096 unless noted)

| Model | Method | bpw | PPL | Fine-tuned | Source |
|---|---|---|---|---|---|
| 2-70B | FP16 | 16 | 3.31 (3.12 at 2048 ctx) | | [6] |
| 2-70B | RTN / GPTQ / OmniQuant (group 128) | 2.125 | 27.27 / 6.74 / 6.55 | no | [6] |
| 2-70B | GPTVQ 1D / 2D | 2.125 | 5.00 / 4.64 | no | [6] |
| 2-70B | QuIP# (paper table, no FT and no E8 / with FT) | 2 | 4.87 / 4.16 | no / yes | [1] |
| 2-70B | QuIP# as reported by later papers | 2 | 3.91 | yes | [5], [12] |
| 2-70B | AQLM 1x16 / 2x8 | 2.0 / 2.0 | 3.83 / 4.21 | yes | [3] |
| 2-70B | AQLM 1x16 + PV-Tuning | 2.0 | 3.78 | yes | [3], [4] |
| 2-70B | VPTQ v8-k65536 | 2.07 / 2.11 | 3.93 / 3.92 (C4 5.72 / 5.71) | no (PTQ) | [5] |
| 2-70B | QTIP (computed codes, as cited) | 2 | 3.78 (C4 5.48; QuIP# 5.71, AQLM 5.62) | see note | [2] via [12], [31] |
| 2-70B | GLVQ-8D / GLVQ-32D | 2 | 3.62 / 3.36 (C4 5.25 / 5.04) | no (PTQ) | [12] |
| 2-13B | AQLM 1x16 / 2x8 / PV | 2 | 5.22 / 5.63 / 5.05 | yes | [3] |
| 2-7B | QTIP 1MAD / 3INST, no fine-tuning | 2 | 7.05 / 6.82 (FP16 5.12) | no | [2] via [20], [32] |
| 2-7B | QuIP# no fine-tuning | 2 | 8.22 | no | [1] |
| 2-7B | VPTQ / QuIP# / AQLM | 2.02 / 2 / 2.02, 2.29 | 6.13 / 6.19 / 6.64, 6.29 | no / yes / yes | [5] |
| 2-7B | ik IQ2_KT (first, FP16 trellis), llama.cpp PPL | 2.125 | 6.36 (f16 4.94), error 28.7% against QTIP 2-bit 33.2% | no | [20] PR 113 |

Notes. QTIP's paper reports its computed codes (1MAD, 3INST) without fine-tuning and HYB with and without; the 70B figure above
comes from later papers quoting QTIP and its fine-tuning status is not stated there. The QTIP paper's own 3- and 4-bit 70B
cells could not be read from the search snippets (web budget exhausted); the paper states that at 3 and 4 bits QTIP roughly
halves the perplexity gap of QuIP# and AQLM, which are "close to lossless" there. The QuIP# 70B 2-bit value differs between the
paper's table (4.16) and later citations (3.91); treat the method's spread as 3.9 to 4.2.

### 3.2 Llama-3.1 Instruct, EXL3 against GGUF, EXL2, AQLM and VPTQ (turboderp's charts, decoder-only bpw, no fine-tuning for EXL3 and GGUF) [19]

Llama-3.1-70B-Instruct perplexity:

| bpw | EXL3 | GGUF imatrix | Others |
|---|---|---|---|
| 1.6 to 1.8 | 11.36 (1.6, H3), 8.19 (1.8) | IQ1_S 11.69, IQ1_M 9.70 | |
| 2.0 to 2.15 | 6.61 (2.0) | IQ2_XXS 7.52 (2.1) | AQLM 2-bit 1x16-g1 6.55 (2.08, fine-tuned); VPTQ v8-k65536-0 7.12 (2.07), v16-k65536-32768 7.21 |
| 2.25 to 2.5 | 6.05 (2.25), 5.53 (2.5) | IQ2_S 6.49 (2.45) | EXL2 2.4 bpw 10.21 |
| 3.0 to 3.1 | 4.50 (3.0) | IQ3_XXS 5.42 (3.07) | EXL2 2.8 bpw 8.82 |
| 3.4 to 3.6 | 4.16 (3.5) | IQ3_S 4.51 (3.45), IQ3_M 4.47 (3.57) | EXL2 3.5 bpw 6.38 |
| 4.0 to 4.3 | 3.85 (4.0) | IQ4_XS 3.92 (4.27) | AWQ 4-bit 4.19, EXL2 4.0 bpw 4.06 |
| 4.5 to 4.8 | 3.76 (4.5) | Q4_K_M 3.81 (4.8) | EXL2 4.5 bpw 3.93 |
| 5.0 to 6.0 | 3.70 (5.0), 3.66 (6.0) | Q5_K_M 3.68 (5.65) | EXL2 5.0 / 6.0 bpw 3.74 / 3.68 |

Llama-3.1-8B-Instruct KL divergence against the unquantized model:

| bpw | EXL3 | GGUF imatrix | Others |
|---|---|---|---|
| 1.8 to 2.0 | 0.764 (1.8, H3), 0.543 (1.9), 0.312 (2.0) | IQ2_XXS 0.649 (2.13) | AQLM 2x8-g8-pv 0.308 (2.0) |
| 2.25 to 2.5 | 0.215 (2.25), 0.151 (2.5) | IQ2_XS 0.418 (2.38), IQ2_S 0.318 (2.5) | AQLM 1x16-g8 0.330, 1x16-g8-pv 0.243 (2.3) |
| 2.7 to 3.0 | 0.053 (3.0) | IQ2_M 0.245 (2.7) | VPTQ v12-k65536-4096 0.357 (2.75); EXL2 3.0 bpw 0.245 |
| 3.25 to 3.6 | 0.030 (3.5) | IQ3_XS 0.092 (3.27), IQ3_M 0.070 (3.5), IQ3_S 0.074 (3.6) | VPTQ v8-k65536-256 0.112 (3.3); EXL2 3.5 bpw 0.110 |
| 3.8 to 4.3 | 0.013 (4.0) | IQ4_XS 0.023 (4.25) | VPTQ v8-k65536-4096 0.052 (3.8); EXL2 4.0 bpw 0.054 |
| 4.5 to 5.0 | 0.003 (5.0) | Q4_K_M 0.018 (4.8) | VPTQ v8-k65536-65536 0.024 (4.5); EXL2 4.5 / 5.0 bpw 0.023 / 0.015 |

Reading: at equal KL, EXL3 at 2.5 bpw sits between GGUF IQ3_XS (3.27) and IQ2_M (2.7), and EXL3 at 3.0 bpw below IQ3_M (3.5) and
IQ4_XS (4.25) on the 8B; on the 70B, EXL3 3.0 bpw equals IQ3_S at 3.45 and EXL3 3.5 bpw lies between IQ3_M and IQ4_XS. A PTQ-only
trellis at 2.0 to 2.3 bpw equals or beats fine-tuned and PV-tuned AQLM on both models. Caveat from the ik_llama.cpp thread: the
GGUF points include the output tensor at whatever the mixture chose, and EXL3 keeps the embedding at 16 bits.

### 3.3 Qwen3.8-27B (the engine's model) and other Qwen

Same harness, frozen 2048-token WikiText-2 window, mean KL(BF16 || candidate), 2047 positions, vLLM fork [22]:

| Checkpoint | Format | Weights on disk | Mean KL | Top-1 |
|---|---|---|---|---|
| turboderp 6.00 / 5.00 bpw | EXL3 (head 6) | 21.4 / 18.5 GiB | 0.0072 / 0.0095 | 97.6 / 97.8% |
| turboderp 4.00 bpw | EXL3 | 15.7 GiB | 0.0159 | 96.2% |
| turboderp 3.50 bpw | EXL3 | 14.3 GiB | 0.0230 | 94.7% |
| turboderp 3.00 bpw | EXL3 | 12.9 GiB | 0.0454 | 92.7% |
| turboderp SC_3.00bpw_H4 (head 4 bits) | EXL3 | 12.5 GiB | 0.0695 | 90.6% |
| turboderp 2.50 bpw | EXL3 | 11.5 GiB | 0.0747 | 90.0% |
| turboderp 2.00 bpw / SC_2.00_H3 | EXL3 | 10.0 / 9.5 GiB | 0.164 / 0.221 | 85.1 / 83.0% |
| unsloth / nvidia / RadixArk NVFP4 | NVFP4 (W4, parts in BF16) | 20.4 to 22.1 GiB | 0.037 to 0.065 | 88.9 to 93.3% |
| cyankiwi AWQ-INT4, dbirks W4A16 AutoRound | INT4 g128 | 19.6 / 18.1 GiB | 0.044 / 0.044 | 93.3 / 93.4% |
| Qwen FP8, AutoRound INT8 | 8-bit | 28.3 / 28.8 GiB | 0.010 / 0.007 | 96.3 / 97.8% |

The EXL3 weights sizes include the 16-bit embedding (2.4 GiB) and the 6-bit head; the 2.5 bpw body alone is about 7.5 GiB
(estimate from section 1 of the baseline). Head precision matters on this 248K-vocabulary model: the head at 4 instead of 6 bits
costs +0.024 KL and 2.1 top-1 points at 3.0 bpw.

Other Qwen3.8-27B and Qwen3 measurements (different harnesses, not comparable with each other):

| Source | Quant | bpw / size | KL | Top-1 or top-p agreement |
|---|---|---|---|---|
| bartowski model card [23] | IQ2_XXS | 8.88 GB | 0.283 | 78.0% |
| | Q3_K_M | 13.40 GB | 0.056 | 89.8% |
| | IQ3_XXS | 12.32 GB | 0.074 | 88.5% |
| | IQ3_M | 14.86 GB | 0.041 | 91.1% |
| | IQ4_XS | 15.48 GB | 0.019 | 94.1% |
| quark75 model card [24] | EXL3 3.0 bpw | | 0.0487 | 91% |
| ubergarm gist, Qwen3-14B [25] | EXL3 3.47 / 3.0 bpw | | 0.035 / 0.059 | |
| turboderp qbench, Qwen3.8-Flash-Next, self-generated trace [19] | EXL3 2.05 / 3.05 / 4.05 / 5.05 bpw | | 0.068 / 0.0177 / 0.0067 / 0.0040 | |
| | GGUF UD-IQ3_XXS (3.3) / UD-IQ4_XS (4.05) / UD-Q4_K_XL (5.2) | | 0.0349 / 0.0165 / 0.0084 | |
| | NVFP4 W4A16 / W4A4 (4.75 incl. scales) | | 0.0100 / 0.0241 | |

The last block is the only published EXL3-against-NVFP4 curve from one harness on a Qwen3.8 model (a different, MoE member of
the family): EXL3 at 3.05 bpw is 1.8x the KL of NVFP4 W4A16, EXL3 at 4.05 bpw is 0.67x of it.

### 3.4 ik_llama.cpp trellis against its own lattice grids (PTQ with imatrix, no rotation) [20]

| Model | Quant | bpw | PPL (WikiText-2, ctx 512 unless noted) | Comparison |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | f32 | 16 | 7.321 | |
| | IQ2_KT / IQ3_KT / IQ4_KT (0xCBAC1FED generator) | 2.125 / 3.125 / 4.0 | 10.95 / 8.08 / 7.559 | errors 49% / 10% / 3.3% |
| | IQ3_KT mix 3.55 bpw against IQ3_XXS mix 3.49 bpw | | 8.14 against 8.46 | same attention/output mix |
| | IQ1_KT at 1.92 bpw | | "nearly the same quality as IQ2_XXS at 2.13 bpw" | PR 616 plot |
| Llama-3.3-70B mix (ffn, attn_q, attn_o in IQ2_KT) | | about 2.2 | 6.07 (6.14 with QTIP's constants) | |
| DeepSeek-R1-0528 | IQ3_KT mix / IQ2_KT mix | 3.48 / 2.49 | 3.306 / 3.638 (ctx 512) | beats UD-Q3_K_XL 3.334 at the same size |

The author's summary across the series: "trellis quantization can bring a 0.1-0.2 bpw reduction in quantized size for the same
model quality" against i-, k- and iqk-quants, and the new integer generator costs a little quality at 2 bits against the FP16
one but removes NaNs and runs much faster.

### 3.5 Decode speed

| Source | GPU | Model, format | tok/s | Comparison |
|---|---|---|---|---|
| QTIP README [2] | RTX 6000 Ada, bs 1, fused q/k/v and up/gate | Llama-2-7B 2-bit: QTIP / QuIP# / AQLM / FP16 | 188 / 186 / 81.5 / 55.9 | 2-70B: 23.5 / 22.2 / 8.78 / OOM |
| SPHQuant [28] | RTX 4090 | Qwen3-VL-8B FP16 / QTIP 2-bit | 106.9 / 209.8 | 1.96x for 8x fewer bytes |
| ik_llama.cpp PR 511, 113 [20] | RTX 4080 | Llama-3.1-8B IQ2_KT (2.41 GiB) tg128 | 205.5 | IQ2_XXS 208 to 216 on the older build (the FP16 KT was 194) |
| ik_llama.cpp PR 505 [20] | RTX 4080 | Llama-3.1-8B IQ4_KT integer / FP16 generator | 134 / 128 | pp512 8000 / 5800 |
| atomic.chat [26] | RTX 5090 | EXL3 3.00 bpw / EXL3 4.00 bpw / llama.cpp Q5_K_M | 119.2 / 113.1 / 130.1 | llama.cpp 15 to 21% faster at matched size; EXL3 prompt processing up to 2.5x faster |
| Tetra [15] | L40S | Qwen3-8B Tetra 2.7 bpp / AWQ 4-bit / FP16 | 95.0 / 123.2 / 46.3 | kernel 0.82 to 1.03x of a planar baseline on sm_120 |
| FLUTE [8] | A6000 | LLaMA-3 70B W3/W4 LUT GEMM | about 30 to 35 | 2 to 4x other LUT GEMMs at batch < 32, 1.5 to 2x end to end |
| VPTQ [5] | | | | claims 2 to 9x faster decoding than QuIP# (QuIP# rotates activations at decode) |

Against the bandwidth bound (bytes per token over DRAM rate): QTIP's 7B kernel reaches about a third of it (RTX 6000 Ada,
HF generation loop), ik_llama.cpp's IQ2_KT about 74% and IQ4_KT about 86% (717 GB/s RTX 4080; estimate), EXL3 within 15 to 21%
of llama.cpp. None of the published trellis kernels is exact in FP32 (EXL3 accumulates in FP16; ik_llama.cpp dots int8 against
int8 activations), so their speeds are upper bounds for an exact kernel.

### 3.6 Converter cost

| Method | Model | Time | Hardware | Source |
|---|---|---|---|---|
| EXL3 (Hessians on the fly, LDLQ, fused Viterbi) | 27B dense, full recipe pipeline | about 2 h, "most of it the conversion" | one fast GPU | [19] optimize.md |
| EXL3 | small models / 70B+ | minutes / "a few hours" | one high-end consumer GPU, layer by layer, multi-GPU optional | [19] README |
| ik_llama.cpp KT | Llama-3.1-8B | 2.5 to 4.5 min (first version) | 32-core Ryzen 5975WX, AVX2 | [20] PR 113 |
| ik_llama.cpp KT | DeepSeek-R1-0528 IQ3_KT / IQ2_KT mixes | 8 h 48 min / 4.4 h | one socket of a Xeon 6980P | [20] PR 529 |
| AQLM | 7B / 70B | about 1 day / 10 to 14 days | one A100 | [3] |
| QuIP# fine-tuning | 70B | about 50 GPU-hours (paper), about 100 (slides) | A100-class | [1], [34] |
| VPTQ | Llama-3.1-405B | about 17 h | not stated | [5] |

## 4. Fit with Euhedral

### 4.1 The eight constraints

**1. Row-exact 1 to 8 row decode.** Two routes exist. The FP32 FMA twins are open to any format whose one-row decode fits the
instruction budget, but they are register-bound from 4 rows (600 GB/s, the 223-register cliff) and that does not change with the
format. The BF16 tensor-core route (`nvfp4::decode_rows<M>`: weight rows as the MMA's 16 M rows, token rows as its 8 N columns,
FP32 accumulate, 8 rows at most 9% dearer than 1) is row-exact by construction but needs every weight exact in BF16, so
decoded value times scale must have at most 8 significant bits (11 for FP16 MMA, which would also force the activations into FP16
range). Section 4.2 tabulates which codebooks pass. Three ways to pass with a group scale: pick integer-valued entries of at most
8 significant bits and apply the group scale to the group's partial sums in FP32 (zeroed accumulator per 32-K group, two
`m16n8k16` per group and one FFMA per accumulator, exact and row-independent); or make the scale a power of two (UE8M0) so the
product stays exact; or define the format's dequantized weight as `bf16_rn(entry x scale)` in the converter, so that `__hmul2`
in the kernel and the scalar reference compute the same bits (the extra rounding is at most 2^-9 relative, a Gaussian MSE
penalty below 0.02%, section 4.3). The Hadamard rotation and sign vectors of QuIP#/QTIP/EXL3 formats are per-row operations and
do not touch row exactness, but every path (one row, twins, prefill) must apply them with the same instruction sequence.

**2. Prefill at 9+ rows without the 102-TFLOPS fallback.** The MXFP8 route expands codes to E4M3 in producer warps and applies the
binary16 group scale in FP32 after each group. Codebooks whose entries are E4M3-exact (4 significant bits) expand by one table
lookup and keep today's 204-effective / 126-achieved TFLOPS: E8P (odd multiples of 1/4, i.e. odd integers up to 15 after scaling
by 4), IQ3_S's D4 grid (odd integers 1 to 15), IQ3_XXS except its value 62, the 3-bit Lloyd-Max grid in E4M3 form (ratios 1 : 3 :
5.5 : 9), IQ4_NL after rounding its 16 values to E4M3. Integer trellis values (ik's -126..126, EXL3 mul1's -510..510) are not
E4M3-exact: they need two E4M3 terms (`hi = e4m3_rn(v)`, `lo = v - hi`, exact because the residual has at most 4 significant bits),
which with the two activation terms is four MMAs per step, 102 TFLOPS effective (three MMAs and a dropped lo x lo term, 136,
within prefill's relaxed-numerics contract), or an INT8 IMMA route with int8 weights and two int8 activation terms (the baseline's
INT8 two-term screen reached 99 TFLOPS against MXFP8's 126, without weight expansion to pay for). A trellis-coded tensor must
also be expanded once per launch, as P2E2 is, because trellis decoding inside the producer warps would repeat P2E2's 0.5 to 0.6x
outcome (section 7 of the baseline); expansion to 8-bit planes writes 8 bits per weight instead of P2E2's 3.25, so about 0.4 to
0.5 ms per layer per 512-row chunk against 0.25 today (estimate), about 7% more TTFT at 128K.

**3. Decode instruction budget.** The budget in the baseline (1.3x Q3's 7 instructions per pair) assumes the same byte count. A
smaller artifact needs a higher code rate: at 2.5 bpw the DRAM-fed rate is 3.25/2.5 = 1.3x today's, which consumes the measured
1.4x margin of the Q3 kernel; the FMA path then tolerates about 1.08x Q3's instructions per code, at 3.0 bpw about 1.29x
(estimate). Per pair, with the accumulate counted on the FMA path and not on the MMA path: Q3 7; a 4-D or 8-D lattice LUT 3 to 4
on either path; HYB about 6; seeded integer group (KT) 6 plus 2 FMA and 2 conversions on the FMA path (10), 6 on the MMA path;
bitshift integer trellis (EXL3 mul1) about 8 plus accumulate (12) on FMA, 8 on MMA. So LUT formats fit both paths at 2.5 bpw,
trellis formats fit only the MMA path and sit at 0.9 to 1.15x Q3's count there (estimates from the kernels in section 2). Large
per-layer tables (AQLM, VPTQ) are latency-bound on L2 lookups and are out. The MMA path has its own unknown: `decode_rows` reaches
the DRAM floor at 4.5 bpw and would have to decode 1.8x as many codes per second at 2.5 bpw; the L2-resident rate test the
repository already uses for Q3 answers this in a day (section 5).

**4. Determinism.** Every scheme is a pure function of the stored bytes; the MMA path is deterministic in the order the kernel
fixes; the Hadamard pre- and post-passes are per row. Nothing changes under CUDA graphs.

**5. Offline conversion only.** LDLQ needs the full input Hessian per linear (5120^2 x 4 B = 105 MB; 17408^2 for `down`, 1.2 GB),
which the llama.cpp imatrix does not hold (diagonal only). EXL3's recipe (250 x 2048 calibration tokens, model on the host, one
layer on the GPU at a time, Hessians on the fly) is the template; a 27B conversion is about 2 hours on a fast GPU and should fit 16
GB (calibration activations 250 x 2048 x 5120 x 2 B = 5.2 GB, the largest Hessian 1.2 GB, the layer under conversion 0.36 GB;
estimate). The Viterbi encoder is GPU work (EXL3's fused kernel; QTIP's pure PyTorch Viterbi over 2^16 states is far slower).
AQLM-style learned codebooks (days per model) and any fine-tuning stage are out of the converter's budget; PTQ-only methods are
the only candidates, which the results in 3.2 and 3.3 support.

**6. The artifact decides everything.** One new EDRL format ordinal per code family and one layout ordinal; `ArtifactProfile`
sees the format on the MLP tensors and keeps the drafting stack as it is.

**7. Memory is the product.** Section 4.5 quantifies bytes and residency. The short version: a 2.5 bpw body frees about 3.3 GiB
against `q3`, enough for 128K of KV and a device-resident DFlash2 drafter with nothing host-backed; a 3.0 bpw body frees 1.8 GiB.

**8. Drafting precision is a separate knob.** The MTP layer and draft head stay NVFP4 (Q3 drafts already accepted fewer tokens
than NVFP4 ones). EXL3 quantizes MTP layers at 4 bits without calibration and the head at 6; the head's sensitivity in 3.3 says
the LM head should stay Q4/Q5-class or NVFP4, not join the 2.5 bpw body.

### 4.2 Row exactness: which codebooks are short-mantissa

| Codebook | Decoded entry before scale | Significant bits | BF16 MMA exact | FP16 MMA exact | E4M3 expansion exact |
|---|---|---|---|---|---|
| E8P (QuIP#) | odd multiples of 1/4, up to 15/4 | 4 | yes | yes | yes (x4) |
| IQ3_S D4 grid | odd integers 1..15 | 4 | yes | yes | yes |
| IQ3_XXS grid | 4, 12, ..., 52, 62 | 4 (62: 5) | yes | yes | all but 62 |
| IQ2_XXS/XS/S grids | 8, 25, 43 | 6 | yes | yes | no (8, 24 or 26, 44 is) |
| IQ4_NL / NF4-like | 16 fixed ints or quantiles | 8 / arbitrary | yes / after rounding | yes | after rounding to 4 bits |
| 3-bit and 2-bit Lloyd-Max | arbitrary | arbitrary | after rounding | after rounding | after rounding |
| QTIP HYB, k-means 2-D | arbitrary FP16 | 11 | after rounding | yes | after rounding |
| ik IQn_KT integer trellis | -126..126 | 8 | yes | yes | no (two terms or int8) |
| EXL3 mul1 | (n - 510)/147.7, n in 0..1020 | 10 | no | yes | no |
| QTIP 3INST / EXL3 mcg | sum of two FP16 | 11 or more | no | no (sum rounds) | no |
| AQLM, VPTQ, GPTVQ | learned FP16 sums | 11 or more | no | after rounding | no |

Exact MMA products also require the activation side to stay BF16 (as `decode_rows` does); FP16 MMA would need activations
converted to FP16, which overflows for the large pre-norm activations this model family produces, so the BF16 column is the one
that matters, and a 2.5 to 3.5 bpw form should use an 8-significant-bit generator: ik_llama.cpp's `dp4a(x & 0x3f3f3f3f) - 126`,
or an EXL3-style bitshift trellis with that generator instead of mul1 (a converter-side change; the trellis structure is
independent of the generator, and EXL3's own `mcg` and `mul1` are swapped by a flag).

### 4.3 What short-mantissa entries cost (this chapter's computation, Gaussian source, optimal scale per codebook)

| Codebook | MSE | SNR | Same entries rounded to E4M3 (4 bits) | Rounded to 3 bits | Rounded to BF16 (8 bits) |
|---|---|---|---|---|---|
| Distortion-rate bound at 2 / 3 / 4 bits | 0.0625 / 0.0156 / 0.0039 | 12.0 / 18.1 / 24.1 dB | | | |
| Lloyd-Max 2-bit (levels 0.453, 1.510) | 0.1175 | 9.30 dB | +0.07% (ratios 1, 3.25) | +0.21% | +0.001% |
| Lloyd-Max 3-bit (0.245, 0.756, 1.344, 2.152) | 0.03454 | 14.62 dB | +0.44% (1, 3, 5.5, 9) | +1.03% | +0.004% |
| Lloyd-Max 4-bit | 0.00950 | 20.22 dB | +2.3% (1, 3, 5, 7.5, 10, 13, 16, 22) | +7.8% | +0.017% |
| IQ4_NL 16 values | 0.00990 | 20.04 dB | +3.9% (-128, -104, -80, -64, -48, -36, -22, -10, 1, 13, 24, 40, 52, 72, 88, 112) | | exact already |
| IQ2 coordinate set 8, 25, 43 as six scalar levels | 0.0594 | | 8, 24, 44: 0.0582 (better); 8, 26, 44: 0.0599 | | exact already |
| IQ3 coordinate set 1, 3, ..., 15 | 0.01154 | | exact already | | |

Rounding entries to BF16 is free; rounding to E4M3 costs under 0.5% of MSE at 2 and 3 bits and 2 to 4% at 4 bits (roughly 0.1 dB,
about 0.02 bpw). For an E4M3-exact codebook the converter should re-run Lloyd-Max or k-means on the E4M3 grid rather than round
afterwards; the numbers above are an upper bound on the penalty. The trellis's advantage over these scalar grids is the
space-filling and shaping gain: the 3-bit scalar optimum is 3.4 dB from the bound, an 8-dimensional lattice recovers about 0.65 dB
of that and a 2^16-state trellis most of the rest (QTIP's argument), which is the 0.1 to 0.2 bpw ik_llama.cpp measures between
KT and its lattice grids at equal rounding.

### 4.4 Proposed format and layout sketch

Target: the Q3-class tensors (`mlp/gate_up`, `mlp/down`, `gdn/output`, `attention/output`: 19.1 B of the 25.6 B streamed
parameters, 72% of `q3`'s streamed bytes) at 2.5 bpw, optionally the Q4/Q5 inputs at 3.5 to 4.0 bpw; the LM head at Q4G64 or
NVFP4; embedding, draft head and MTP layer unchanged.

- **Codes plane ("T-plane")**: per weight row, K-contiguous 16 x 16 tiles in EXL3's order (a tile = 16 rows x 16 K = one
  `m16n8k16` A fragment), 16K uint16 words per tile (40 words = 80 B at K = 2.5 with the period-16 half-bit mask), tiles of a
  128-row x 128-K block contiguous, rows padded to a multiple of 128 on K as today, 256-byte plane alignment.
- **Generator**: bitshift trellis, L = 16, K = 2.5 or 3, value `v = dp4a(mul(state) & 0x3f3f3f3f, 0x01010101, -126)`, an integer in
  [-126, 126] with RMS about 37 (the converter's unit), BF16-exact. Alternative with the same layout: `mul1` if an FP16 MMA path
  were ever acceptable (it is not, 4.2).
- **Scales**: none per group (the rotation makes the rows near-Gaussian; EXL3 uses none). Per row: one BF16 output scale
  (`svh`, 0.0006 bpw) and the 128-point block Hadamard's sign vector on the inputs (`suh`, shared across rows); the 128-point
  Hadamard itself is a fixed matrix. Per tensor: one FP32 scale in the epilogue. If the rotations chapter rejects a Hadamard on
  the activation path, fall back to KT's structure: 4-bit E4M3-coded block scale per 32 (0.125 bpw) applied to the group's
  partial sums, 2.625 bpw total.
- **Decode kernels**: `trellis::decode_rows<M>` for M = 1..8, the `nvfp4::decode_rows` structure with the pair table replaced by
  the generator (2 BFE, 2 IMUL, 2 LOP3, 2 IDP4A, 1 PRMT per pair, the dp4a accumulator pre-biased to land the integer in a BF16
  bit pattern; the -126 offset moves into a per-row correction `126 x scale x sum(x)` computed once per token row), BF16
  activations, FP32 accumulate, one summation order shared by M = 1 and M > 1, output scale and per-tensor scale in the epilogue.
  A 128-point Hadamard pre-pass on the activation row (the KV path already has an H256 Sylvester kernel) and a 128-point post-pass
  on the outputs, both deterministic per row.
- **Prefill**: `trellis::expand` writes an int8 plane (or E4M3 hi + lo planes) once per launch into the P2E2-style scratch; then
  either the INT8 IMMA route (two int8 activation terms, exact int32 accumulation, FP32 per-row scales after each 128-K tile) or
  the MXFP8 route with two weight terms (three MMAs). Which one wins is a measurement (section 5, step 4); both are at or above
  the 102-TFLOPS BF16 floor and both avoid P2E2's decode-in-producer trap.
- **Converter** (`tools/euhedral_artifacts/trellis.py`): Hessian collection from 250 x 2048 calibration tokens with the model on
  the host and one layer on the GPU (Swift-1.5's own traces for the deployed artifact, as EXL3's optimize.md recommends for
  reasoning models), Hadamard and sign vectors, LDLQ in 16-row steps, Viterbi per tile on the GPU (EXL3's kernel is the reference;
  it is Python plus a CUDA extension, GPL-free MIT-licensed code, so the algorithm can be ported), packing to the T-plane.

### 4.5 Bytes per token, resident GiB, context headroom (estimates from the baseline's inventory and budget)

| Form | Streamed GiB per token | One-row floor at 720 GB/s (q3's rate) | Payload GiB (with 0.48 embedding, 0.25 draft head, 0.22 MTP, 0.05 small) | Freed against q3 (11.72) |
|---|---|---|---|---|
| `q3` today | 10.7 | 16.0 ms | 11.72 | |
| MLP only at 2.5 bpw, rest as q3 | 9.2 | 13.7 ms | 10.2 | 1.5 |
| Q3-class tensors at 2.5 bpw (19.1 B), rest as q3 | 8.4 | 12.5 ms | 9.4 | 2.3 |
| Everything streamed at 2.5 bpw except LM head at 4.25 | 7.6 | 11.3 ms | 8.6 | 3.1 |
| Everything streamed at 3.0 bpw except LM head at 4.25 | 9.0 | 13.5 ms | 10.0 | 1.7 |
| Everything streamed at 3.5 bpw except LM head at 4.25 | 10.5 | 15.6 ms | 11.5 | 0.2 |

Budget: 15.56 GiB device, 0.75 to 0.95 desktop and context, 1280 MiB planner reserve, 157 MiB ReplaySSM, KV 2.25 GiB at 128K,
MTP layer and draft head 0.57 GiB or the DFlash2 drafter 1.06 GiB (NVFP4) to 3.43 GiB (BF16). `q3` at 11.72 GiB runs 128K with
nothing host-backed and no DFlash2; the DFlash2 artifact needs 0.45 to 0.55 GiB more than MTP and already host-backs 468 MiB at
4K and 1.5 GiB at 64K on the `nvfp4-fc` artifact, where verification is bound by those copies. An 8.6 GiB payload (2.5 bpw body)
leaves about 3.1 GiB more than `q3`: 128K of KV plus the NVFP4 DFlash2 drafter fully resident with about 1.5 GiB to spare, or
the BF16 drafter with 128K at the edge; a 10.0 GiB payload (3.0 bpw) fits the NVFP4 drafter and 128K with about 0.5 GiB to
spare. Decode: the one-row floor falls from 16.0 to 11.3 ms (1.42x) at 2.5 bpw and to 13.5 ms (1.19x) at 3.0 bpw if the decode
kernels stay at the DRAM rate; with MTP, the verification rows ride the MMA path at 1.09x of one row instead of today's 600 GB/s
twins, which is a second, independent gain of the same order for the 4 to 8 row quanta (estimate).

### 4.6 Expected quality in the repository's metrics (estimates)

Cross-harness translation is the weak link: the repository's `nvfp4` reads KL 0.147 and top-1 88.2% on its chat document where
the community NVFP4 checkpoints of the same base model read KL 0.037 to 0.065 and top-1 89 to 93% on a WikiText-2 window, so the
repository's document is harder by roughly 3x in KL and 3 to 4 top-1 points. Scaling the same-model EXL3 figures by that ratio:

| Body bpw (EXL3-quality rounding) | KL on jedi95's harness | Expected KL on the repository's document | Expected dNLL | Expected top-1 | Against today |
|---|---|---|---|---|---|
| 2.5 | 0.075 | 0.18 to 0.25 | +0.07 to +0.10 nats | 86 to 87% | better than `q3` (+0.170, 0.293, 76.2%) at 23% fewer bytes |
| 3.0 | 0.045 | 0.11 to 0.15 | +0.03 to +0.05 | 88 to 90% | about `nvfp4` (+0.029, 0.147, 88.2%) at 33% fewer bytes |
| 3.5 | 0.023 | 0.06 to 0.08 | +0.02 to +0.03 | 91 to 92% | better than `nvfp4` at 22% fewer bytes |

Two corrections push these numbers around. Down: the estimates assume EXL3's full recipe (Hadamard, LDLQ with a full Hessian,
the trellis, a 6-bit head); the KT-style recipe (imatrix clustering, no rotation, no LDLQ) would give up roughly 0.3 to 0.5 bpw of
that, by the EXL3-against-IQ gap in 3.2 minus ik's 0.1 to 0.2 bpw trellis gain. Up: the repository's quality harness resolves
0.02 to 0.05 nats, and Swift-1.5 self-generated calibration (EXL3's optimize.md: sensitivities measured on raw text "transfer
poorly" for aligned reasoning models) can move the deployed artifact more than the codebook does. The honest statement: a 2.5
bpw form reaching `q3` quality or better is very likely; a 3.0 bpw form reaching `nvfp4` quality is likely but inside the harness's
resolution; a 3.5 bpw form beating `nvfp4` is likely.

### 4.7 Interaction with MTP and DFlash2

Drafts and verification run through the same linears as decode, so the format's multi-row kernels serve them; on the MMA path
the 5 to 7 verification rows cost about 9% over one row instead of today's register cliff, which is where `q3`'s MTP speedup is
capped (109 tok/s at 4K against a 16.0 ms one-row step, i.e. 1.7x for 3.08 accepted tokens per verification). The drafter proposes
the base distribution; a lower-precision target shifts the verified distribution and so the acceptance rate, which is a one-pass
measurement on the chat corpus (section 9 of the baseline). EXL3 quantizes its MTP layer at 4 bits uncalibrated and reports no
acceptance numbers; keep NVFP4 drafting. Memory is the main interaction (4.5): the 2.5 bpw body is what makes the DFlash2 drafter
resident at 64K and 128K, where today's DFlash2 verification is PCIe-bound.

## 5. Experiment plan

Ordered by cost; each step has a stop condition. Steps 0 to 2 need no kernel.

0. **Place lattice codebooks on the repository's KL scale (hours, no code).** From the BF16 GGUF already in the toolchain,
   `llama-quantize` with the ISTA imatrix to IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS, IQ3_S and, as scalar controls, Q3_K_S and Q3_K_M; run
   `llama-perplexity --kl-divergence-base bf16.kld` on `chat-corpus-v1-document.md` with the repository's prefix and steps. This
   yields the first same-harness comparison of an 8-D and 4-D lattice codebook with imatrix rounding against `q3`'s KL 0.293 and
   `nvfp4`'s 0.147. Go: IQ3_XXS (3.06 bpw) at KL at most 0.20 or IQ3_S (3.44 bpw) at most 0.15; otherwise the codebook alone is
   not the lever and step 2's scalar control decides.
1. **Measure the trellis ceiling on this checkpoint (one to two days).** Convert Swift-1.5 with `exllamav3/convert.py` at 2.5,
   3.0 and 3.5 bpw (head 6, default calibration; then `-cd` with a self-generated Swift-1.5 trace), and score KL and top-1 on the
   repository's document with `eval/model_diff.py` or qbench; the reference implementations differ from llama.cpp's BF16 by about
   the kernel floor (KL 3e-3), so the comparison with `q3` and `nvfp4` is direct. Go: 2.5 bpw at most half of `q3`'s KL and 3.0 bpw
   at most `nvfp4`'s. This also tells whether Swift-1.5-trace calibration beats the ISTA/UkisAI imatrix for the deployed artifact.
2. **Python reference of the proposed format, scored without kernels (one to two weeks).** In `tools/euhedral_artifacts/`,
   implement the integer bitshift trellis with the 8-bit generator, LDLQ with a Hessian collected from 250 x 2048 tokens (model
   on the host, one layer on the 5070 Ti at a time), the 128-point block Hadamard and sign vectors; quantize the MLP of all 64
   layers at 2.5 and 3.0 bpw; write the dequantized weights into a BF16 GGUF and score with `llama-perplexity --kl-divergence`
   (exact repository metric, CPU). Ablate 2 x 2: with and without the Hadamard, with and without LDLQ (imatrix nearest search
   instead), and add the scalar control Q3G64 with LDLQ, which is the cheapest possible improvement to `q3` (no kernel change) and
   belongs to the GPTQ/rotations chapters but must be on this table. Measure per-layer relative output error against BF16 as well.
   Go for the fifth form: trellis 2.5 bpw at KL at most 0.18 and 3.0 bpw at most 0.147, and at least 0.05 KL better than Q3G64 +
   LDLQ at 3.25 bpw; otherwise ship Q3G64 + LDLQ and stop.
3. **Decode-rate microbenchmark (one week).** A one-row `trellis::decode_rows<1>` on L2-resident `gate_up` and `down`, then the
   M = 4 and M = 8 twins; the harness in COMPRESSED_Q3.md (cold weights rotated past L2, CUDA events). Go: one-row `gate_up` at
   most 1.15x its bytes over 855 GB/s (about 66 us at 2.5 bpw) and M = 8 at most 1.15x M = 1; exactness by `SpeculativeVerify`-style
   bitwise comparison between M = 1 and M = 8 rows on the kernel before any engine integration.
4. **Prefill route (one week).** `trellis::expand` plus the INT8 IMMA two-term route and the MXFP8 three-term route on
   512-row chunks; measure ms per layer and the drift test. Go: at most 10% more TTFT at 128K than `q3`, drift within the floor.
5. **Artifact and engine (two to three weeks).** EDRL ordinals, loader, `ArtifactProfile`, converter pipeline, the exactness
   tests, `TeacherForcedQualityCudaIntegrationTest`, benchmarks at 4K / 32K / 64K / 128K with MTP and DFlash2, acceptance per
   verification. Go: at 2.5 bpw, dNLL at most +0.10 with KL at most 0.18, decode at least 1.3x `q3` at one row and at least 1.15x
   with MTP at 4K, 128K resident with the DFlash2 drafter; at 3.0 bpw, dNLL at most +0.04 and decode at least 1.15x `q3`.

## 6. Risks and open questions

- **ALU-bound decode at 2.5 bpw.** The byte reduction raises the required code rate by 1.3x; whether `decode_rows`'s structure has
  1.8x code-rate headroom over its 4.5 bpw operating point is unmeasured (step 3). If not, the LUT lattice form (IQ3_S or E8P
  style, 3 to 4 instructions per pair, E4M3-exact) is the fallback at a cost of about 0.1 to 0.2 bpw in quality.
- **FP32-exact kernels are slower than the published ones.** EXL3 accumulates in FP16 and ik_llama.cpp dots int8 against int8;
  their speed numbers are upper bounds. The FP32-accumulate `m16n8k16` runs at 102 TFLOPS peak, which is irrelevant at one row but
  caps the 8-row twin at about 12x less than the FP16-accumulate instruction's 190 (still far above what 8 rows need).
- **Hessian collection on 16 GB.** 250 x 2048 tokens through a 54.7 GB BF16 model with the model on the host is hours of PCIe
  and CPU time per conversion; the exact figure on the i9-14900K is unknown. EXL3's "faux-LDLQ" (no decomposition) and a diagonal
  Hessian (the imatrix) are the degraded fallbacks, with unknown loss on this model.
- **Calibration distribution.** The deployed checkpoint is Swift-1.5; EXL3's optimize.md reports that raw-text sensitivities
  transfer poorly for aligned reasoning models. Step 1 measures the difference; it may be larger than the codebook's effect.
- **LM head and inputs.** EXL3's head at 4 bits costs +0.024 KL on this model; the head must stay at 4.25 bpw or better. Whether
  the Q4/Q5 input projections (22% of `q3`'s bytes) tolerate 3.5 bpw trellis coding is untested here; EXL3 treats them like the MLP.
- **Hadamard on the activation path.** Every linear gains a 128-point block transform on its input row and on its output; cheap,
  but new code on the row-exact path, and a decision for the rotations chapter. The KT-style no-rotation variant removes it at a
  quality cost.
- **Prefill expansion memory.** An 8-bit expanded plane of `gate_up` is 178 MB against P2E2's 72 MB; the P2E2 scratch policy
  (128 MiB of weights plus an activation region) needs a larger region or output-row chunking.
- **Acceptance with a 2.5 bpw target** is a measurement, not a derivation; a drop of a few percent is plausible.
- **Open: fractional bitrates per tensor.** EXL3's half-integer K (mul1 only) and Q-Palette show that 2.5 bpw is not special; a
  per-tensor recipe (EXL3's `sc_optimize.py` style) could spend 2.25 on `gate_up` and 3.0 on `down`, which this chapter did not model.
- **Open: E2M1-valued trellises.** XOR-Trellis (2026) decodes to FP4 values; if its quality holds up, an E2M1 x per-group-scale
  format would ride the native 815-TFLOPS NVFP4 prefill route unchanged. Only the abstract was available.
- **Open: YAQA and GuidedQuant rounding** on top of the trellis report about 30% lower KL than LDLQ at the same bits; EXL3 already
  accepts YAQA Hessians. This is the second-order lever after the format exists.
- **Numbers not found.** QTIP's 70B 3- and 4-bit table cells, SqueezeLLM's 13B and 65B rows, Any-Precision's perplexities, Proteus
  and CurveTQ results beyond their abstracts, and an EXL3 decode measurement on a 4090/5090 at a fixed model (the atomic.chat
  numbers do not name the model). The web search budget for this run was exhausted before these could be chased.

## 7. References

1. Tseng, Chee, Sun, Kuleshov, De Sa. "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks." ICML 2024, arXiv:2402.04396. Repository: https://github.com/Cornell-RelaxML/quip-sharp (`lib/codebook/latticee8_padded12.py`, README).
2. Tseng, Sun, Hou, De Sa. "QTIP: Quantization with Trellises and Incoherence Processing." NeurIPS 2024, arXiv:2406.11235. Repository: https://github.com/Cornell-RelaxML/qtip (`lib/codebook/bitshift.py`, `qtip-kernels/src/inference.cu`, README speed table).
3. Egiazarian et al. "Extreme Compression of Large Language Models via Additive Quantization." ICML 2024, arXiv:2401.06118. Repository and model table: https://github.com/Vahe1994/AQLM.
4. Malinovskii et al. "PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression." NeurIPS 2024, arXiv:2405.14852.
5. Liu et al. "VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models." EMNLP 2024, arXiv:2409.17066. Repository: https://github.com/microsoft/VPTQ.
6. van Baalen et al. "GPTVQ: The Blessing of Dimensionality for LLM Quantization." arXiv:2402.15319.
7. Kim et al. "SqueezeLLM: Dense-and-Sparse Quantization." ICML 2024, arXiv:2306.07629. Repository: https://github.com/SqueezeAILab/SqueezeLLM.
8. Guo et al. "Fast Matrix Multiplications for Lookup Table-Quantized LLMs" (FLUTE). EMNLP 2024 Findings, arXiv:2407.10960. Repository: https://github.com/HanGuo97/flute (README results and kernel support table).
9. Park et al. "Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs." arXiv:2402.10517.
10. Kim et al. "GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance." ICML 2025, arXiv:2505.07004.
11. Tseng et al. "Model-Preserving Adaptive Rounding" (YAQA). arXiv:2505.22988. Repository: https://github.com/Cornell-RelaxML/yaqa.
12. Zhang, Wu, Wang, Lin. "Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression." NeurIPS 2025, arXiv:2510.20984. Repository: https://github.com/xzhang9308/GLVQ.
13. Lee, Song. "Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment." NeurIPS 2025, arXiv:2509.20214.
14. Que, Elkayam, Pyakurel, Pan, Gope. "XOR-Trellis: Ultra-Low-Complexity Dequantization and Curvature-Aware Hadamard-Free LLM Quantization." arXiv:2610.00432 (2026-09-30).
15. Malandrino. "Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per Parameter." arXiv:2609.35465 (not peer reviewed); related arXiv:2609.02652 and arXiv:2603.11021.
16. Yang, Li et al. "Proteus: Lookup-Free Trellis-Coded Quantization by Lattice-Breaking Compute Codes for 2-Bit LLMs." ICML 2026 poster, https://icml.cc/virtual/2026/poster/63948.
17. "CurveTQ: Rotation-Free Trellis Quantization of LLM Weights via Curvature-Weighted Search." arXiv:2610.09212. Also BCJR-QAT arXiv:2605.10655, CCQ arXiv:2507.07145 (abstracts only).
18. GuidedQuant and YAQA secondary figures: https://jinukkim.me/blog/guidedquant/ and https://www.together.ai/blog/yaqa (unverified against the papers).
19. turboderp-org/exllamav3: https://github.com/turboderp-org/exllamav3. README (EXL3 section), `doc/convert.md`, `doc/optimize.md`, charts `doc/llama31_70b_instruct_bpw.png`, `doc/llama31_8b_instruct_kld_bpw.png`, `doc/qb_kld.png`, `doc/procedural_codebook.png`; sources `exllamav3/exllamav3_ext/quant/codebook.cuh`, `exl3_dq.cuh`, `exl3_gemv_kernel.cuh`, `exllamav3/modules/quant/exl3.py`, `exl3_lib/quantize.py`. Commit cloned 2026-10-10.
20. ikawrakow/ik_llama.cpp: https://github.com/ikawrakow/ik_llama.cpp. README section "Trellis quants"; PR 113 (first trellis, 2024-11), PR 441 (CPU), PR 453, PR 484, PR 505 (integer IQ4_KT), PR 511 (integer IQ2_KT, RTX 4080 table), PR 529 (V2, DeepSeek results), PR 616 (IQ1_KT); `ggml/src/ggml-cuda/convert.cu` (`trellis_next_int`), `ggml/src/ggml-cuda/template-instances/mmvq-instance-iq4_kt.cu`, `ggml/src/iqk/iqk_quantize.cpp` (`QuantizerIQKT`), `ggml/src/ggml-common.h` (`block_iq2_kt` etc.).
21. ggml-org/llama.cpp: https://github.com/ggml-org/llama.cpp. `ggml/src/ggml-common.h` (block structures, `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `kvalues_iq4nl`), `ggml/src/ggml-quants.c` (`iq2xs_init_impl`, "3-bit using D4 lattice"), `tools/quantize/README.md`. Grid coordinate sets computed from the tables in this chapter.
22. jedi95. "Qwen 3.8 27B KLD Testing." https://github.com/jedi95/qwen3.8-27b-kld (README table, 2048-token WikiText-2 window, BF16 reference).
23. bartowski/Qwen3.8-27B-GGUF model card (KLD table), https://huggingface.co/bartowski/Qwen3.8-27B-GGUF, as quoted in search results; also unsloth/Qwen3.8-27B-GGUF discussion 49 (bartowski IQ2_XXS re-measured at KLD 0.301, top-1 76.5%).
24. quark75/Qwen3.8-27B-EXL3-3.0bpw model card, https://huggingface.co/quark75/Qwen3.8-27B-EXL3-3.0bpw (KLD 0.0487, top-1 91%).
25. ubergarm. "Kicking the tires on early version of exllamav3," https://gist.github.com/ubergarm/9d560bab80241b90dac802e91b656743 (Qwen3-14B KLD).
26. "EXL3 vs GGUF: Quality, Speed and VRAM," https://atomic.chat/blog/guides/exl3-vs-gguf (RTX 5090 decode and KLD figures; model not named in the snippet).
27. Together AI. "Even Better, Even Faster Quantized LLMs with QTIP," https://www.together.ai/blog/even-better-even-faster-quantized-llms-with-qtip.
28. "SPHQuant: Efficient extreme low bit weight quantization for Vision-Language Models," arXiv:2609.24875 (RTX 4090 QTIP 2-bit against FP16 decode figures).
29. mobiusml/hqq, https://github.com/mobiusml/hqq (scalar HQQ/HQQ+, not a codebook method; 158 tok/s Llama-3-8B 4-bit on a 4090 via gemlite).
30. Euhedral repository, branch `perf/mtp-gated-depth`: `docs/COMPRESSED_Q3.md`, `docs/NVFP4_NATIVE.md`, `docs/PREFILL_MX.md`, `docs/DFLASH2.md`, `docs/NVFP4_RESIDENCY.md`, `docs/QUALITY.md`, `native/src/q3/contiguous.cuh`, `native/src/nvfp4/nvfp4.cuh`; and `00-baseline.md` in this folder.
31. alphaXiv summary of QTIP (Llama-2-70B 2-bit C4 5.48 against QuIP# 5.71 and AQLM 5.62), https://www.alphaxiv.org/abs/2406.11235 (secondary).
32. Liner review of QTIP (Llama-2-7B 2-bit 1MAD 7.05, 3INST 6.82), https://liner.com/review/qtip-quantization-with-trellises-and-incoherence-processing (secondary).
33. "ITQ3_S: High-Fidelity 3-bit LLM Inference via Interleaved Ternary Quantization," arXiv:2603.27914 (Llama-3-8B IQ4_XS 6.41, IQ3_S 7.03, FP16 6.14; not used in the tables).
34. QuIP# ICML 2024 slides, https://icml.cc/media/icml-2024/Slides/34816.pdf (fine-tuning about 100 GPU-hours for 70B).
