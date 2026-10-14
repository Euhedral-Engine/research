# Quantization roadmap for the four Euhedral forms

Synthesis of the nine chapters in this folder for the maintainer of the Euhedral inference engine: what each quantization
technique can buy for the `q3`, `q3-compressed`, `nvfp4` and `nvfp4-compressed` artifacts of Qwen3.8-27B and its Swift-1.5
fine-tune on one RTX 5070 Ti (16 GB) in quality (NLL, KL, top-1 against BF16), bytes per token and resident memory, what it costs,
and in what order to try it. Every number comes from `00-baseline.md` ("baseline") or a chapter ("chapter 05"); estimates are
labelled as the chapters label them. The chapters carry the references.

## Abstract

All four forms sit about one bit of effective precision below the public frontier for the same checkpoint, and the gap is in
rounding and bit allocation, not in kernels. `nvfp4` measures KL 0.147 to BF16 at 4.64 whole-file bpw where public 4.4 to 5.3 bpw
GGUFs measure 0.010 to 0.019; `q3` measures 0.293 at 3.74 bpw where public 3.0 to 4.1 bpw files measure 0.05 to 0.08 (chapter 09).
The kernel drift floor is KL 3.4e-3 to 4.7e-3, 30 to 85 times below the quantization error (baseline), and the deployed Swift-1.5
`nvfp4-compressed` lands where the base `nvfp4` does (KL 0.155, top-1 89.4%), so neither the checkpoint nor the matrix explains the
gap (chapter 09). The single document, its genre and the NVFP4 KV cache account for a factor of roughly 2 to 3 of the 5 to 10x
(chapters 03 and 09, estimates); the rest is the converter: a per-group scale search with no error feedback and no channel scaling,
a grid that reaches code -4 only when a group's largest value is negative (chapter 01), and a `q3` ladder that spends Q5 on
`gdn/value_z` while holding `mlp/down`, `gdn/output`, `attention/output` and the LM head at Q3 (chapters 05 and 09). Three levers act
at zero bytes and zero kernel work: signed scales, GPTQ around the existing group search and AWQ folds (chapter 01: `q3` from +0.170
to about +0.11 to +0.13 nats, estimate); a rebalanced allocation (chapter 05: 0.03 to 0.06 KL at net -0.06 GiB streamed, estimate);
and block-wise reconstruction training on the 16 GB card (chapter 08: about +0.08 to +0.11 nats in 15 to 20 hours, estimate). The
structural follow-on is an integer-valued trellis at 2.5 to 3.0 bpw decoded on BF16 tensor cores, which same-model evidence puts at
or beyond `q3` quality with 28% fewer streamed bytes (chapter 03). `q3-compressed`, a lossless transcode of round-to-nearest codes,
has no calibrated future: a compressed calibrated Q3 lands near 3.0 bpw and P2E2's fast path would overflow (chapter 07); its
successor is a -3..3 Q3 whose codes are trained rather than searched (chapter 08), or the trellis.

What to do first costs no engine code: the held-out evaluation of chapter 09 (four texts, 32K scored tokens each, Swift-1.5 BF16
as the reference), because the current document resolves 0.02 nats on NVFP4 forms and 0.05 on Q3 forms (baseline) and most gains
proposed here are of that size; then the four measurements every later decision depends on: the weight-only KL of the existing
artifacts, dequantized into a GGUF and scored in llama.cpp (chapters 01 and 09); a PyTorch fake-quant module reproducing the
converter byte for byte (chapter 08); the KV codec swap inside today's 144-byte rows, which measures what the cache costs at all
(chapter 06); and the code histogram of the calibrated `q3` artifact, which decides whether any -4-capable P2E2 layout is worth
building (chapter 07). Then the converter one-liners, the allocation probes at 3 to 5 minutes each, and the training tiers.

## How to read this folder

| File | What it covers | Verdict |
|---|---|---|
| [00-baseline.md](00-baseline.md) | The four forms at PR 119 (commit 269a890): inventory, formats, memory budget, execution routes, the eight constraints a new format must meet, what was already rejected, the harness | Quantization error is 30 to 85 times the kernel noise floor; memory is the product, and the quality headroom is in the converter. |
| [01-rounding-and-calibration.md](01-rounding-and-calibration.md) | GPTQ, AWQ, signed scales, learned rounding, FP4 scale search, imatrix practice, calibration data, Hessian collection on 16 GB | Signed scales, then GPTQ with the existing group search as quantizer, then AWQ folds cut `q3` to about +0.11 to +0.13 nats (estimate) for zero bytes and no kernels; NVFP4 gains are below the harness except a wider E4M3 scale search. |
| [02-rotations-and-outlier-transforms.md](02-rotations-and-outlier-transforms.md) | QuIP/QuIP#, QuaRot, SpinQuant, FlatQuant, HIGGS, Hadamard FP4 recipes, mapped onto the hybrid architecture | Weight-only rotation gives none to small on `q3` and is likely negative on NVFP4; it cannot unlock single-term FP4 activations; a single MXFP8 activation term on the Q3 prefill route is the real experiment. |
| [03-codebook-vector-trellis.md](03-codebook-vector-trellis.md) | Scalar LUTs, lattice grids, AQLM/VPTQ, QTIP/EXL3/KT trellises, as a fifth form at 2.5 to 3.5 bpw | Build an integer-valued bitshift trellis decoded on BF16 tensor cores; EXL3 at 2.5 bpw beats `q3` on the same model, and a 3.0 bpw form lands near `nvfp4` at 32% fewer bytes; 4 to 8 engineer-weeks, decided by two no-kernel experiments first. |
| [04-fp4-block-formats-and-sm120.md](04-fp4-block-formats-and-sm120.md) | NVFP4, MXFP4, SD4, FP6, 2:4 sparse FP4, SVDQuant, a Q3 scale table, on sm_120 | NVFP4-SD4 is already the best native 4.25 bpw format; the free wins are a wider scale search and SD4 for the Q4 `query_key` tensors; FP6 only if the Q5 families prove sensitive; grids, sparsity and SVDQuant do not pay. |
| [05-mixed-precision-and-the-hybrid-architecture.md](05-mixed-precision-and-the-hybrid-architecture.md) | Sensitivity measurement, Transformer and GDN evidence, outlier protection, heads, embedding, drafting stack | The `q3` ladder is inverted; probe `gdn/value_z` at Q4 and spend the saving on `mlp/down`, `gdn/output` and `attention/output`, give the LM head an imatrix for free, move the draft head to Q3 on NVFP4 forms. |
| [06-kv-cache-and-recurrent-state.md](06-kv-cache-and-recurrent-state.md) | 3-bit and 2-bit KV grids under the existing H256 rotation, page layouts, long-context evidence, the GDN state | Build INT3 keys and values at 3.5 bits per element (14 KiB per token), then 2.5-bit values; `nvfp4-compressed` at 128K from 52 to roughly 70 to 80 tok/s (estimate); the GDN state stays FP32. |
| [07-lossless-and-entropy-coding.md](07-lossless-and-entropy-coding.md) | P2E2, -4-capable layouts, scale-plane coding, rANS/tANS/Huffman, rate-distortion against finer quantizers | A compressed calibrated Q3 lands at 92 to 94% of row-split bytes, not 81.5%, and P2E2's fast path would take the slow path on 99.8% of slices; measure the calibrated histogram first; no entropy coder beyond P2E2 fits the GEMV. |
| [08-finetuning-qat-and-low-rank-correction.md](08-finetuning-qat-and-low-rank-correction.md) | Block-wise reconstruction, QAT and QAD, scale-only tuning, low-rank branches, drafter training | Block-wise training on the 16 GB card roughly halves `q3`'s gap (estimate) at zero bytes; restricted to -3..3 it lifts the P2E2 form to `q3` quality; a drafter fine-tune is free speed; a uniform 2-bit MLP is not viable. |
| [09-public-baselines-and-evaluation.md](09-public-baselines-and-evaluation.md) | Every public quantization of Qwen3.8-27B found, the recipes behind the best, Swift 1.5, evaluation methodology | The forms sit about one bit below the public frontier; evaluate every form on Swift-1.5 over four texts at 32K scored tokens with weight-only attribution runs, and gate on thinking-token length. |

## Where the forms stand

The engine's numbers (baseline) are teacher-forced over 1279 tokens of one document after a 1281-token prefix, each form against
the BF16 of the checkpoint it was converted from, in llama.cpp on the CPU; standard errors of the paired NLL difference in parentheses.

| Form | Source, matrix | bpw (main tensors / whole file) | File | NLL vs BF16 (s.e.) | KL | Top-1 | Decode tok/s 4K / 32K / 64K / 128K | Host-backed at 32K / 64K / 128K |
|---|---|---|---|---|---|---|---|---|
| `q3` | base; ISTA-DASLab imatrix, calibrated Q3G64 with Q4/Q5 inputs | 3.25 / 3.74 | 11.72 GiB | +0.170 (0.021) | 0.293 | 76.2% | 109 / 105 / 93 / 67 | 0 / 0 / 0 |
| `q3-compressed` | base; none (round-to-nearest), P2E2 transcode | about 2.65 / 3.24 | 10.16 GiB | +0.275 (0.025) | 0.400 | 70.5% | 95 / 92 / 82 / 64 | 0 / 0 / 0 |
| `nvfp4` | base; ISTA-DASLab imatrix, calibrated NVFP4 | 4.5 / 4.64 | 14.52 GiB | +0.029 (0.008) | 0.147 | 88.2% | 116 / 101 / 72 / 30 | 1.0 / 1.6 / 2.8 GiB |
| `nvfp4-compressed` (deployed) | Swift-1.5; UkisAI `imatrix-swift15-v1mix.gguf`, calibrated SD4 | 4.25 / 4.40 | 13.77 GiB | +0.028 (0.009) | 0.155 | 89.4% | 146 / 102 / 88 / 52 | 0.27 / 0.84 / 1.97 GiB |

The 25.6 B streamed parameters cost 2.98 GiB per bit per weight; `q3` moves its 10.7 GiB in 16.0 ms at about 720 GB/s against a
measured 855 GB/s; the KV cache is 18 KiB per token (2.25 GiB at 128K); each 128 MiB of host-backed weights costs about 2 to 2.5% of
decode speed (baseline). SD4 against plain NVFP4 on the same base conversion is +0.005 (0.009) nats and KL 0.037, so the two NVFP4
forms are one point on the curve (baseline, chapter 09).

The external yardstick is a same-harness KL table on a 2048-token WikiText-2 window for the base model (chapter 03) and the public
GGUF cards (chapter 09):

| Public quantization of Qwen3.8-27B (base) | bpw | KL to BF16 | Top-1 | Chapter |
|---|---|---|---|---|
| EXL3 2.5 / 3.0 / 3.5 / 4.0 bpw | 2.5 / 3.0 / 3.5 / 4.0 | 0.075 / 0.045 / 0.023 / 0.016 | 90.0 / 92.7 / 94.7 / 96.2% | 03 |
| Community NVFP4 checkpoints (parts in BF16) | about 4.5 to 5 | 0.037 to 0.065 | 88.9 to 93.3% | 03 |
| AWQ-INT4 and AutoRound W4A16, g128 | 4 | 0.044 / 0.044 | 93.3 / 93.4% | 03 |
| bartowski Q4_K_M / Q3_K_M (wiki.test, 100 chunks) | 5.28 / about 3.9 | 0.0129 to 0.0139 / 0.0466 to 0.0564 | | 09 |
| unsloth UD-Q4_K_XL / UD-IQ2_XXS | 5.22 / 2.68 | 0.0096 / 0.146 | 96.0 / 83.0% | 09 |
| ISTA GSQ-RCO IQ3_S (10.99 GiB) | 3.51 | benchmarks only: AIME25 and LiveCodeBench v6 equal to BF16 | | 09 |

Three things matter in this comparison. The engine's numbers are end to end: every artifact runs its KV cache in NVFP4 and its
kernels relaxed, and the NVFP4 forms quantize prefill activations to two FP4 terms, so the weight rounding's own share is not
isolated anywhere in the repository (baseline); a community llama.cpp measurement puts an NVFP4 KV cache on this model near 0.015
KL (chapter 09). The text differs: chapter 09 estimates the genre effect at about 2x and the one-document sampling error at 0.01
to 0.025 KL, chapter 03 scales same-model EXL3 numbers by about 3x for the repository's document; both are estimates, both fall
far short of 5 to 10x, and only a same-harness run settles it. The gap is not kernels and not the checkpoint or matrix (chapter
09). The public recipes keep the LM head at Q6_K or BF16 and the GDN output projection at 5 to 8 bits, use error feedback or
learned assignments instead of a per-group scale search, and search the per-tensor allocation (chapters 01, 04, 05, 09); the `q3`
file is larger than ISTA's IQ3_S (chapter 09). That is the headroom this folder is about.

## What the evidence says, by lever

### Rounding and calibration (chapter 01)

The converter already does one of the three things this family does: a 39-candidate clipping and stretch search per group of 64,
weighted by the imatrix diagonal, which took `q3` from +0.275 to +0.170. It does neither cross-column error feedback (GPTQ) nor
input-channel scaling (AWQ), and it clamps every binary16 scale positive, so code -4 is reached only in groups whose largest
magnitude is negative; llama.cpp flips the scale's sign instead, and the Q3 decode kernel, the reference kernel and the P2E2
transcoder appear to accept a negative scale as read (argued from the source, not from running it). The estimates below rest on a
synthetic re-implementation of the search, whose RTN-to-calibrated step reproduces the measured -38%, and on published W3 g128
recoveries of 13 to 29% for GPTQ and 16 to 64% for learned rounding.

| Form | Today | Signed scales | + GPTQ (own search as quantizer) | + AWQ folds | + learned rounding (50 to 200 iterations) |
|---|---|---|---|---|---|
| `q3` | +0.170, KL 0.293, 76.2% | +0.15 to +0.16, KL about 0.27 | +0.13 to +0.15, KL 0.23 to 0.26 | +0.11 to +0.13, KL 0.20 to 0.24 | +0.10 to +0.12, KL 0.18 to 0.22 |
| `q3-compressed` | +0.275 (RTN) | not possible (P2E2 has no -4) | calibrated -3..3 plus GPTQ: about +0.17 to +0.19 | | |
| `nvfp4`, `nvfp4-compressed` | +0.029 / +0.028 | n/a; wider E4M3 search: -5 to -10% of weight error | -10 to -20% of weight error | small | -10 to -20% of weight error |

All estimates (chapter 01, section 4.5). Costs: signed scales, one day; the wider NVFP4 search (`SEARCH_ABOVE` 2 to 6, global
scale `amax / (6 x 256)`), one day; a calibration pass that streams one layer at a time and accumulates full Hessians, 16 x 16
block diagonals and a Swift-1.5 imatrix with the LM head and MTP, one to two weeks to build and 30 to 60 minutes per run at 262K
tokens in under 7 GB of device memory; GPTQ around `search_group_codes_torch`, one to two weeks and 20 to 60 minutes per
conversion; AWQ folds, one week (free for every norm-fed linear and for `mlp/down` and `attention/output`; not for `gdn/output`,
the heads or the embedding); learned rounding, two to three weeks, after AutoRound has set the ceiling at 1 to 3 GPU hours. Both
public matrices lack the LM head and the MTP `fc` (`llama-imatrix` needs `--process-output` and `--nextn`), and calibrated codes
are less peaked around zero, so the 81.5% P2E2 ratio measured on RTN codes must be remeasured (chapters 01, 07).

### Rotations (chapter 02)

A residual-stream rotation folds for free into the embedding, norms, LM head and every norm-fed or residual-writing linear; online
transforms before `mlp/down`, `attention/output` and `gdn/output` cost 0.25 to 0.4 ms per token unfused (1.5 to 2.5% of a 16 ms
step), about nothing fused, and are row-exact by construction. The weight-only verdict is narrow: a signed Hadamard makes the
imatrix diagonal exactly flat, so the weighted scale search degenerates to the unweighted one. On a toy with heavy-tailed weights
and realistic importance Q3 gains (true error 4.4 to 4.6% with the imatrix against 3.5 to 3.8% rotated) but NVFP4 loses about 60%
(0.44 to 0.47% against 0.73 to 0.78%), and the measured P2E2 ratio implies 2.40 bits per code and about 20% BIG codes, what
Gaussian groups give (20.5%), so the groups are already near-Gaussian. Expected gain on weights: none to small for `q3`, likely
negative for the NVFP4 forms; rotation pays only paired with error feedback (chapter 01) or Gaussian-optimal trellises (chapter
03). The sigmoid output gate blocks SpinQuant's offline per-head rotation, and nothing on the GDN value path can be rotated
offline.

Chapters 02 and 04 disagree about single-term FP4 activations in prefill, about the bar rather than the physics. Chapter 02
simulates one NVFP4 term at 9.51% RMS on Gaussian rows, 9.28% with massive-activation channels and 9.55% after rotation, against
the engine's measured 9.5%: the Gaussian floor of E2M1 at block 16, which rotation cannot lower. Chapter 04 keeps a retest
plausible with a per-block Hadamard, a 4/6-style scale choice (the engine's six-candidate search gave 8.8%), SmoothQuant scaling
and a gate on teacher-forced KL instead of the drift window, because the one-term run's KL of 6e-3 is 4% of the weight KL and the
prize is a third of the gate_up tile time (1.573 against 2.405 ms at 1024 rows). The error stays near 9% either way, so the
question is whether double the drift-floor KL in prefill is acceptable, which the engine rejected (baseline, section 7). Both
chapters endorse a single MXFP8 activation term on the Q3 prefill route instead: 2.65% RMS, no rotation, half the MMA work for 1.3
to 1.7x on gate/up and down, about +0.01 KL (estimates; chapter 02, E3). Random signs on the KV H256 and an offline V rotation are
free (chapter 02, E4).

### Codebook and trellis formats (chapter 03)

The same-model evidence is the strongest in the folder: on a frozen WikiText-2 window EXL3 at 2.5 bpw reaches KL 0.075 and top-1
90.0% where community NVFP4 checkpoints sit at 0.037 to 0.065 and 89 to 93%, and EXL3 at 3.0 bpw reaches 0.045 and 92.7%. Most of
the gain is the rounding (Hadamard incoherence plus LDLQ against a full input Hessian), not the codebook: ik_llama.cpp's KT
trellis with imatrix rounding and no rotation gains only 0.1 to 0.2 bpw over its own lattice grids. The verdict is an
integer-valued bitshift trellis (EXL3's structure with an 8-significant-bit generator so that `code x scale` is exact in BF16)
decoded on BF16 tensor cores the way `nvfp4::decode_rows` works, because at 2.5 bpw the FP32 FMA GEMV must decode 1.3x faster than
Q3's and has about 1.08x of instruction headroom per code (baseline, chapter 03). The lattice-LUT alternative (IQ3_S's D4 grid or
E8P, E4M3-exact, riding the MXFP8 prefill route unchanged) is the fallback; trellis prefill expands once per launch to an int8
plane for an INT8 IMMA or two-term MXFP8 route.

| Body at EXL3-quality rounding | Streamed GiB per token | One-row floor at 720 GB/s | Payload | Freed against `q3` | Expected KL on the repository document | Expected dNLL | Expected top-1 |
|---|---|---|---|---|---|---|---|
| `q3` today | 10.7 | 16.0 ms | 11.72 GiB | | 0.293 | +0.170 | 76.2% |
| 2.5 bpw, LM head at 4.25 | 7.7 | 11.5 ms | 8.7 GiB | 3.0 GiB | 0.18 to 0.25 | +0.07 to +0.10 | 86 to 87% |
| 3.0 bpw, LM head at 4.25 | 9.1 | 13.6 ms | 10.1 GiB | 1.6 GiB | 0.11 to 0.15 | +0.03 to +0.05 | 88 to 90% |
| 3.5 bpw, LM head at 4.25 | 10.5 | 15.7 ms | 11.5 GiB | 0.2 GiB | 0.06 to 0.08 | +0.02 to +0.03 | 91 to 92% |

Estimates (chapter 03, sections 4.5 and 4.6): the quality columns scale the WikiText numbers by about 3x for the harder document,
and a KT-style recipe without rotation or LDLQ gives up roughly 0.3 to 0.5 bpw of them. Cost: a Hessian-collecting converter
(about 2 hours per conversion on a fast GPU; the PCIe and CPU time on the i9-14900K is unknown), a one-row and multi-row MMA
decode family, an expansion kernel, one EDRL format and one layout ordinal, roughly 4 to 8 engineer-weeks. Two risks decide it
early: whether `decode_rows` has the 1.8x code-rate headroom a 2.5 bpw stream needs (unmeasured), and the LM head, which must stay
at 4.25 bpw or better (a 4-bit head cost +0.024 KL in the EXL3 table). The decisive experiments need no kernels: same-harness KL
of IQ quants and EXL3 conversions on the repository's document, then a Python reference of the format written back into a BF16
GGUF and scored with `llama-perplexity --kl-divergence` against a Q3G64 plus LDLQ control (chapter 03, steps 0 to 2).

### FP4 block formats (chapter 04)

NVFP4 is the right FP4 member for this card and SD4 is already the best 4.25 bpw format sm_120 runs natively: on synthetic weights
MXFP4 carries 1.8 to 2.4x the MSE of searched NVFP4 because UE8M0 scales clip or waste up to 1.33x of a block's range. No
published FP4 variant beats NVFP4 plus a scale search by a margin the harness could resolve; the one gap is upward, a scale up to
5 codes above round-to-nearest (which contains the Four Over Six candidate) for 6 to 10% less MSE at zero kernel cost, but the SD4
"2 codes below" experiment moved NLL by -0.003 (0.009) for a comparable MSE change, so this is a free change for a larger
evaluation, not a measurable one. For `q3` the chapter finds three things: SD4 in place of Q4G64 on the `query_key` tensors at
identical bytes lowers their error by a quarter (Gaussian) to half (Laplacian) and moves them onto the native prefill route; FP6
E2M3 with the existing 64-group FP16 scales (6.25 bpw) is the one 6-bit code that expands exactly to E4M3, with 4 to 8x lower MSE
than Q5G64 for +0.42 GiB on `gdn/value_z` and `attention/gate_value`; and a 256-entry per-tensor table for the Q3 scale plane
saves 0.37 GiB of streamed bytes (3.5% of decode time or about 21K tokens of KV) for +0.36% MSE, at the price of a new layout
ordinal and a table in every Q3 kernel family (chapter 07's lossless index saves 0.12 to 0.16 GiB).

Chapters 04 and 05 pull `gdn/value_z` in opposite directions: chapter 04 offers to raise it to FP6 if the Q5 families carry more
than 0.02 nats of `q3`'s penalty, chapter 05 proposes to lower it to Q4 because the GDN evidence does not support the Transformer
"V is sensitive" rule. Both route through the same diagnostic, a per-family sweep with `llama-quantize --tensor-type` that raises
one family at a time to F16 and reads the KL on the chat document plus a second corpus, one afternoon with no engine code (chapter
04, step 1; chapter 05's second track).

### Bit allocation across the hybrid (chapters 05 and 09)

Chapters 05 and 09 agree, from different evidence, that the `q3` ladder is inverted. Chapter 05 reads every llama.cpp mix
(attention V, `ffn_down`, the output head and the outer eighths of the layers get more bits) and the super-weight literature (the
most destructive scalars live in early `mlp.down_proj`), and finds the GDN half pointing the other way: a 2026 preprint quantized
all 496 linears of Qwen3.8-27B to NVFP4 W4A4 with accuracy close to BF16 and argued that the delta rule forgets injected error
within hundreds of tokens. Chapter 09 reads the public recipes: `linear_attn.out_proj` (the engine's `gdn/output`, Q3 today) is
the most KLD-sensitive tensor in a per-tensor sweep on Qwen3.5-9B and every community build pins it at Q5_K to Q8_0, low-bit SSM
tensors produce "endless reasoning traces", `in_proj_z` tolerates MXFP4 badly, and FFN gate/up tolerate about 3 bits. The exchange
rate the two base forms define is 0.054 KL and 0.052 nats per GiB; a reallocation must beat it (chapter 05).

| Family (q3) | Today | Probe | Streamed GiB | Go if (chapter 05) |
|---|---|---|---|---|
| `gdn/value_z` | Q5G64 | Q4G64 | -0.35 (-3.3%) | ΔKL under +0.019, dNLL within 1 s.e. |
| `gdn/query_key` | Q4G64 | Q3G64 | -0.12 | ΔKL under +0.006 |
| `gdn/output` | Q3G64 | Q4G64 | +0.18 | ΔKL under -0.010 |
| `attention/output` | Q3G64 | Q4G64 | +0.06 | ΔKL under -0.003 |
| `mlp/down` | Q3G64 | Q4G64 in layers 0 to 7 and 56 to 63 (A), all layers as the ceiling (B) | +0.17 (A) / +0.66 (B) | A gains 0.009, B 0.036 KL |
| `text/output_head` | Q3G64, uniform weights | Q3 with the `output.weight` imatrix; Q4 if the imatrix recovers less than half of a Q5 probe's gain | 0, then +0.15 | KL falls by more than 0.02 |
| Combined (2 + 3 + 4A + head imatrix) | | | net -0.06 | total ΔKL under -0.03 at under +0.1 GiB; speed within 1% |

Chapter 05 expects the rebalancing to be worth 0.03 to 0.06 KL on the engine's scale (estimate, from the 20 to 33% KLD reductions
computed per-tensor layouts achieve on this model near 3.5 bpw), which the single document can just resolve; each probe is a
minute or two of conversion and 1 to 2 minutes of teacher forcing after one day of converter work (`--override regex=format`,
`importance.py` mappings for the head, draft head and MTP, a `--report-error` pass). The one probe where the chapters genuinely
disagree is `gdn/value_z`: chapter 05's case rests on one preprint whose mechanism an independent review calls only partly
supported, chapter 09's warnings about 4-bit DeltaNet projections at long context carry no numbers, and `z`, which chapter 09 says
tolerates MXFP4 badly, is fused with `v` in one object whose layout cannot carry two formats; the go threshold plus a 32K
teacher-forced slice is the resolution. For the NVFP4 forms chapter 05 keeps every base linear at SD4, gives the LM head its
imatrix for zero bytes, and moves the draft head from NVFP4 to Q3 (-0.10 GiB resident; the Q3 head accepted 1.49 drafts per
verification against 1.44 for NVFP4 on the `q3` base).

### KV cache and recurrent state (chapter 06)

The engine's KV format (E2M1 codes, E4M3 scale per 16, after a per-head H256 rotation) is already the family the literature
settled on for 4-bit caches, so the only open question is how many code bits the rotated coordinates need. Published long-context
evidence puts 4 bits near lossless, 3 bits at a small measurable loss at 7B scale, and 2 bits per element at a real loss on
retrieval and reasoning (TurboQuant's 3-bit mode drops AIME25 on Qwen3-30B-A3B from 0.683 to 0.443). The chapter's simulation on
rotated rows puts INT3 midrise times E4M3 per 16 at 1.8x today's error, INT2 at 4.2x, and INT4 midrise at the same 4.5 bits per
element at 0.85x, a free-quality candidate if the tables are rebuilt anyway. Integer codes up to 15 times an E4M3 scale stay exact
in FP16, so the verifier, the prefix cache and the lookup kernels survive with new tables.

| K / V format | KiB per token | KV at 128K | `nvfp4-compressed` host-backed at 128K | `nvfp4-compressed` 128K decode | `q3` 128K decode |
|---|---|---|---|---|---|
| NVFP4 / NVFP4 (today) | 18 | 2.25 GiB | 1.97 GiB | 52 tok/s (measured) | 67 tok/s (measured) |
| INT3 / INT3 | 14 | 1.75 GiB | 1.47 GiB | about 70 (estimate) | 70 to 75 (estimate) |
| INT3 / INT2 | 12 | 1.50 GiB | 1.22 GiB | about 80 (estimate) | 70 to 75 (estimate) |

The payoff is memory, not arithmetic: `q3` host-backs nothing and gains only the attention bytes (+4 to +12% at 128K, under 3% at
32K). The cost is a new codec, new tables in four attention kernels and their exact twins, page sizing in the planner and prefix
cache, a KV-format field in the artifact profile, and a long-context harness the repository does not have. The first step is about
50 lines: swap only the codec inside today's 144-byte rows in a test build, so every kernel computes the candidate numerics with
exact twins; this also measures, for the first time, what the cache costs at all. The GDN state stays FP32: BF16 storage saves at
most 1.2% of a decode token and 72 MiB, every reference keeps it FP32, and slow heads amplify per-step rounding to 1.4 to 4% of
the state (estimate).

### Lossless coding and the future of `q3-compressed` (chapters 03, 07, 08)

Chapter 07 closes the question the baseline left open. The calibrated search does not only produce -4s that P2E2 cannot store; it
flattens the code distribution. A Gaussian model that reproduces the two measured RTN statistics (2.3 bits of entropy, 80% of
codes in -1, 0, +1) predicts the search raises entropy to 2.7 bits and the BIG fraction from 20% to 37%, puts a -4 in 71% of
groups (often several), and would push even a hypothetical -4-free P2E2 to 92% of row-split bytes. The best lossless layout that
stores -4 (a two-level payload with a one-bit escape stream) lands near 94% for about 18 to 19 instructions per code pair against
P2E2's 11 and Q3's 7, so one-row decode would be roughly 25% below `q3` instead of 13% (estimate). At P(BIG) = 0.37, 99.8% of
slices would take P2E2's slow path; at 0.315 (the -3..3 search), 68%. In rate-distortion terms entropy-coded calibrated Q3 is a
3.0 bpw point about 4 dB above the Shannon bound, so a finer native quantizer wins at equal bytes.

| `q3-compressed` candidate | bpw | Expected NLL vs BF16 | Build | Chapter |
|---|---|---|---|---|
| Today: RTN Q3, P2E2 | 2.66 | +0.275 (measured) | none | baseline |
| Calibrated Q3 restricted to -3..3, P2E2 unchanged | 2.88 (ratio 0.886, model) | about +0.20 to +0.22 (estimate); 73% of the calibrated gain | converter argument; raise P2E2's 16-BIG fast-path limit | 01, 07 |
| Same plus GPTQ | 2.88 or more | about +0.17 to +0.19 (estimate) | converter | 01 |
| Calibrated Q3 with -4, two-level layout B | 3.06 (0.942) | +0.170 (same values as `q3`) | new layout and kernels, 25% slower decode | 07 |
| Block-wise trained Q3 restricted to -3..3, P2E2 | 2.65 (if the histogram holds) | +0.10 to +0.15 (estimate) | training (15 to 20 h on the card) plus a `--codes-from` converter entry | 08 |
| Integer trellis body at 2.5 bpw | 2.5 | +0.07 to +0.10 (estimate) | new format, kernels, converter; 4 to 8 weeks | 03 |

The three chapters that propose a successor agree that the full-range P2E2 layout of `docs/P2E2_FULL_RANGE.md` is not the route;
they differ in how much of `q3`'s quality a 2.5 to 2.9 bpw artifact keeps. Chapters 01 and 07 put a searched -3..3 artifact at
about three quarters of the calibrated gain (chapter 01: a quarter forfeited; chapter 07: 73% kept), worse than `q3` by 0.03 to
0.05 nats at 0.37 bpw fewer bytes; chapter 08 expects a trained -3..3 artifact to reach or beat `q3`'s +0.170 at 2.65 bpw, saving
1.56 GiB (about 90K tokens of KV), because training replaces the scale search rather than restricting it; chapter 03 expects a 2.5
bpw trellis to beat `q3` outright but needs new kernels. The order follows from cost: histogram (hours), restricted-search
conversion (hours), trained -3..3 artifact (days), trellis (weeks). Byte identity between `q3` and `q3-compressed`, which the
tests assume, ends with any change of rounding (chapter 07), and the P2E2 ratio must then be remeasured (chapter 08).

### Fine-tuning after rounding (chapter 08)

This is the only family that improves quality at zero bytes and zero kernel changes: its output is Q3/Q4/Q5 codes with binary16
scales, or NVFP4 codes with E4M3 scales, in the layouts the kernels already serve. At 3 bits the model is in ParetoQ's
compensation regime, where few tokens suffice; a uniform 2-bit MLP is in the reconstruction regime and is not viable.

| Tier | What is trained | Hardware, time, money | `q3` expected | `q3-compressed` (-3..3) expected | NVFP4 forms |
|---|---|---|---|---|---|
| Today | | | +0.170, KL 0.293 | +0.275 | +0.029 / +0.028 |
| 0a: scales and norms, block-wise | 400 M binary16 scales and 0.66 M norm weights | 5070 Ti, 5 h (1024 x 2048 tokens) to 15 to 20 h, $0 | +0.13 to +0.15 | +0.19 to +0.22 | within noise |
| 1: all parameters per block (Block-AP) | one decoder block at a time against cached BF16 activations; 6 GB of optimizer state, 11 to 16 min per block | 5070 Ti 15 to 20 h with 172 GB of NVMe caches (43 GB at 1024 samples), $0; or one H100 3 to 6 h, $10 to $25 | +0.08 to +0.11, KL about 0.15 to 0.18 | +0.10 to +0.15 | +0.015 to +0.02 (low confidence) |
| 2: end-to-end QAD, 0.1 to 1 B tokens | all parameters or LoRA, KL to the teacher's top-k | 8x H100, $50 to $650 | +0.07 to +0.09 | +0.09 to +0.13 | +0.01 to +0.02 |
| 3: MTP block, unrolled to depth 2 to 4, NVFP4 fake-quant | the 0.42 B drafter on the quantized target's own greedy generations | 5070 Ti, 0.5 to 1.6 h per 10 M tokens, $0 | speed only; output never changes | | |

All quality figures are estimates (chapter 08, section 4.2); the Q3 rows are medium confidence, the NVFP4 rows sit below the
harness resolution. The drafter row rests on Swift 1.5 shipping the base model's MTP head (the card is reported to say so; whether
the LoRA touched that block is not stated; baseline, chapters 08 and 09), so a drafter tuned to the quantized target's hidden
states changes speed and never output; HASS added 8.47% throughput to DeepSeek-R1's official MTP with two epochs of
off-distribution data. Two risks govern the family: quantization inflates reasoning length and ModelOpt's QAD doubled one
benchmark's output length at unchanged score, so every gate needs a thinking-token guard; and the converter has no entry for
pre-rounded codes, so a `--codes-from` recipe with a byte-equality test against the fake-quant model is new work.

Chapters 01 and 08 give two routes to roughly halving `q3`'s +0.170, and they are compatible. Chapter 01's is post-training:
signed scales, GPTQ and AWQ folds reach +0.11 to +0.13 for about an hour of converter time plus a 30 to 90 minute calibration
pass, and learned rounding reaches +0.10 to +0.12 for 1 to 3 GPU hours. Chapter 08's is training: block-wise reconstruction
reaches +0.08 to +0.11 for 15 to 20 hours. The ranges overlap, the methods compose, and the cheaper one goes first and sets the
baseline the dearer one must beat. The reader's choice is which infrastructure to build first: chapter 01's Hessian-collecting
calibration pass, which the trellis converter of chapter 03 also needs, or chapter 08's block-wise training loop with its
fake-quant module, which the drafter fine-tune and the -3..3 successor also need; both are one to two weeks, and the fake-quant
module is the smaller step.

## Ranked roadmap

Ordered by expected payoff per unit of effort, zero-engine-code measurements first. "Build" is none, converter, kernel or format
(a new EDRL ordinal). Quality numbers are the chapter's estimates unless marked measured; every go/no-go runs on the repository's
harness (`TeacherForcedQualityCudaIntegrationTest`, `compare_reference.py`, the exactness tests, the benchmark configs, the
acceptance pass) plus the held-out set of row 1.

| # | Experiment | Forms | Build | Expected effect | Cost | Go / no-go | Chapter | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | Held-out evaluation: T1 (today's document), T2 wikitext-2, T3 code, T4 Swift-1.5 generations with thinking, 32K scored tokens each, Swift-1.5 BF16 reference, `compare_reference.py` extended to percentile KL and delta-p | all | none (llama.cpp toolchain) | resolution from 0.02 to 0.05 nats to about 0.005; the prerequisite for every row below | 1 day, CPU | s.e. at or below 0.005 nats on Q3-class forms; a change counts only above 2 s.e. on T2 and T4 with the same sign on all four | 08, 09 | DONE (Euhedral PR #123) |
| 2 | Weight-only KL: dequantize each existing artifact to a BF16 GGUF, score with `llama-perplexity --kl-divergence` on T2 with an F16 KV cache; the same run on the public IQ3_XXS, IQ3_S, Q3_K_M quants and the per-family F16 sweep (`llama-quantize --tensor-type`) | all | none | separates weight rounding from the KV cache and FP4 prefill; ranks each family's share of `q3`'s 0.293; the first number comparable to the public column | half a day to one afternoon, CPU | the attribution is the result; go for FP6 (row 22) if the Q5 families own more than 0.02 nats, for SD4 `query_key` (row 11) if the Q4 families own more than 0.01; IQ3_S at or under 0.15 KL says the codebook is a lever | 01, 03, 04, 05, 09 | DONE (Euhedral PRs #123, #124): Q4/Q5 families own about 0.004 KL, LM head 0.061 of q3's 0.193; IQ3_S 0.072 at q3's size, so the codebook is a lever |
| 3 | Fake-quant parity: a PyTorch module reproducing `grouped.py` and `nvfp4.py` byte for byte on the `q3` and `nvfp4-compressed` recipes, exported and compared teacher-forced | all | none (tools) | the substrate for rows 17 to 19 and for any PyTorch-side weight-only KL | 1 to 2 days | KL between fake-quant model and artifact under 1e-2 | 08 | DONE (Euhedral PR #124): 984 objects byte for byte on q3, nvfp4, nvfp4-compressed; `--codes-from` reproduces SHA-256; engine vs fake-quant KL 0.0076 (q3), 0.0081 (nvfp4-compressed) |
| 4 | KV codec swap inside today's 144-byte rows: INT4-midrise, INT3-midrise, INT2-midrise against E2M1, exact twins included | `q3`, `nvfp4-compressed` | none (test-build switch, about 50 lines) | measures the cache's own KL cost for the first time; decides INT4-midrise (0.85x error at equal bytes) against E2M1 | 1 day | INT3/INT3 and INT3/INT2 within KL 0.01 and 1 top-1 point of E2M1; no-go for a grid that doubles the E2M1 cache's KL | 06 | DONE (Euhedral PR #131): cache + FP4 prefill + relaxed kernels add 0.004-0.005 KL on nvfp4-compressed (weights alone 0.024 at 32K), no growth from 2.5K to 32K; INT4 midrise -5 to -9% KL at equal bytes on nvfp4-compressed (flat on q3); INT3 +0.007 KL, -0.6 to -0.8 top-1 at 32K (inside the bar); INT2 rejected |
| 5 | Code histogram of the calibrated `q3.edrl`: entropy, P(BIG), P(-4), -4 per group, BIG per 32-code lane, payload words per slice, distinct FP16 scales per tensor | `q3`, `q3-compressed` | none (read-only script) | gives the bytes of every P2E2 variant and the scale-index width directly | hours, CPU | -4 layout only if P(BIG) at or under 0.25 and P(-4) at or under 0.01; no-go above P(BIG) 0.30; 10-bit scale index if every tensor has at most 1024 distinct scales | 07 | DONE |
| 6 | Trellis ceiling: convert Swift-1.5 with exllamav3 at 2.5 / 3.0 / 3.5 bpw (head 6), default calibration and a self-generated Swift trace, score KL and top-1 on T1 | fifth form | none | the ceiling for row 20 and the first test of Swift-trace calibration against the two imatrices | 1 to 2 days | 2.5 bpw at most half of `q3`'s KL; 3.0 bpw at most `nvfp4`'s | 03 | TODO |
| 7 | Signed scales for Q3/Q4/Q5: remove the clamp in `_binary16_scales`, flip the sign when the group's largest value is positive | `q3` (not `q3-compressed`) | converter | -0.015 to -0.02 nats, 9 to 13% of weighted MSE; zero bytes | 1 day | dNLL improves by at least 0.02 on the holdout; exactness and oracle tests pass on the converted artifact | 01 | DONE (Euhedral PR for signed scales): q3 KL -8 to -11% on the holdout, weight-only 0.140 -> 0.127 at zero bytes; acceptance about -4%, decode within noise at 32K |
| 8 | NVFP4 scale search above round-to-nearest: `SEARCH_ABOVE` 2 to 5 or 6, global scale `amax / (6 x 256)`, SD4 table DP rerun | `nvfp4`, `nvfp4-compressed`, MTP layer | converter | -6 to -10% weight MSE; invisible on one document | 1 day | weight-only KL (row 2) down 5%; dNLL at or under 0 within s.e.; acceptance unchanged | 01, 04 | NO-GO (Euhedral PR for the native search options): on top of native GPTQ with act-order, `nvfp4-compressed` got worse at above 5 / scale max 256 (KL 0.028 / 0.030 / 0.058 / 0.156 on T1-T4) and at above 6 (0.028 / 0.031 / 0.058 / 0.153); the SD4 table already absorbs the reach. `nvfp4` at 5 / 256: 0.026 / 0.031 / 0.057 / 0.157, -1 to -9% on its own baseline but not better than the shipped artifact on T3/T4 |
| 9 | Importance for the LM head, draft head and MTP: rerun `llama-imatrix --process-output` (and `--nextn`) per lineage, map in `importance.py` | all | converter | a free gain of unknown size; the head is the first fix if a Q5 head probe moves KL by more than 0.02 | 1 day plus imatrix runs | head probe: KL down more than 0.02; Q4 head (+0.15 GiB) only if the imatrix recovers less than half | 01, 05 | TODO |
| 10 | Restricted -3..3 calibrated conversion transcoded with the unchanged `q3_p2e2.py`, teacher-forced and benchmarked at 4K and 32K | `q3-compressed` | converter (one argument) | about +0.20 nats (model), ratio about 0.886 (2.88 bpw); 68% of slices on P2E2's slow path | hours | NLL within 0.05 nats of `q3`; decode within 15% of `q3`; also records the real ratio | 01, 07 | DONE |
| 11 | SD4 in place of Q4G64 on `attention/query_key` and `gdn/query_key` in the `q3` recipe | `q3` | converter (recipe) | identical bytes; MSE -25% (Gaussian) to -58% (Laplacian); native prefill for those tensors; first test of mixed-format dispatch | 1 day | dNLL below zero by more than 1 s.e. on two documents, or equal quality with faster prefill; exactness tests pass | 04, 05 | NO-GO (row 2: the Q4 families own under 0.004 KL) |
| 12 | Calibration pass tool (layer streaming, full Hessians, 16 x 16 block diagonals, a Swift-1.5 imatrix with head and MTP; 256 x 2048 tokens, half Swift's own thinking-mode generations) and GPTQ around the existing group search in `grouped.py` and `nvfp4.py` | all | converter | `q3` to +0.13 to +0.15 nats, KL 0.23 to 0.26 (with row 7); NVFP4 -10 to -20% of weight error; zero bytes | 2 to 4 weeks; 30 to 60 min per calibration run, 20 to 60 min per conversion, under 7 GB device | block-Hessian weights alone: weight-only KL down 3%; GPTQ: `q3` dNLL -0.03 and KL -10% with top-1 up; NVFP4 weight-only KL -10% with acceptance down at most 2% relative; remeasure the P2E2 payload | 01 | DONE (Euhedral PRs #128, #130: holdout KL -18 to -43% on all four artifacts; q3 acceptance up slightly, NVFP4 within prompt noise). Native calibrator on the engine's kernels (PR #133, 13 min per artifact): act-order plus GPTQ for the MTP layer adopted for `q3` only (KL 0.086 / 0.080 / 0.183 / 0.263 on T1-T4, T4 -10%); on the other three forms it was worse on T4; a 384-sequence calibration set was worse on T2 |
| 13 | AWQ scales folded into norms, `up` rows and `v` rows, searched on the quantized output error, before GPTQ | `q3`, then NVFP4 forms | converter | a further -0.02 nats / -7% KL on `q3` | 1 week | the stated gain on the holdout; skip for NVFP4 if row 12 is within 10% of the weight-only floor | 01 | TODO |
| 14 | `q3` allocation probes: `value_z` Q5 to Q4, `gdn/output` and `attention/output` Q3 to Q4, `mlp/down` Q4 in layers 0 to 7 and 56 to 63 (and all layers as the ceiling), `query_key` Q4 to Q3, then the combined recipe and a 32K teacher-forced slice | `q3` | converter (`--override regex=format`, `--report-error`) | combined: 0.03 to 0.06 KL at net -0.06 GiB streamed | 1 day of converter work, then 3 to 5 min per probe | per-probe thresholds in the allocation table above; combined ΔKL under -0.03 at under +0.1 GiB, speed within 1% at 4K and 32K, speculative tests pass | 05, 09 | DONE (Euhedral PR #127): value_z Q4 (-0.38 GB, KL flat) + MLP down Q4 in layers 0-5/58-63; outputs Q4 rejected (T4 dNLL), query_key Q3 rejected; q3 KL -7 to -11% on T1/T2 at -0.25 GB |
| 15 | Draft head Q3 on `nvfp4-compressed` | NVFP4 forms | converter (recipe) | -0.10 GiB resident; slightly faster drafts | one deterministic acceptance pass | tokens per verification within 1% of the NVFP4 head | 05 | REJECTED: Q3 draft head on nvfp4-compressed lowered acceptance (4K 3.63 -> 3.43, 64K 3.17 -> 3.02); decode -3% at 4K and 64K, +4% at 32K |
| 16 | Combined converter-only `q3` and `nvfp4-compressed` recipes (rows 7 to 15) as the new shipped artifacts | all | converter | the sum of the rows above; byte-identical `nvfp4-compressed`, `q3` within +0.1 GiB | one conversion each plus the holdout | holdout dNLL and KL better on T2 and T4; thinking length within 10% of BF16 Swift; acceptance within 1% | 01, 05, 09 | TODO |
| 17 | Tier 0a then Tier 1 block-wise training on Swift-1.5 (two blocks first, then all 65), exported through a new `--codes-from` entry | `q3` | converter entry plus training scripts | `q3` to +0.08 to +0.11 nats, KL about 0.15 to 0.18; zero bytes | 5 h to 20 h on the 5070 Ti with NVMe caches, $0; or 3 to 6 h on one H100, $10 to $25 | two blocks: block-output MSE down 30%; artifact: dNLL better by at least 0.03 on the holdout, thinking length within 10%, no external-text regression; below 0.02 the method does not pay here | 08 | MEASURED, NOT ADOPTED: native trainer in the quantization module (Java + CUDA kernels checked against float64 central differences). Tier 0a on q3 (scales and norms, 200 x 2048 calibration tokens, 2 epochs): two-block test held-out block-output MSE -36% (layers 10, 11); all 64 layers in turn: KL 0.088 / 0.080 / 0.202 / 0.225 on T1-T4 against GPTQ's 0.086 / 0.080 / 0.183 / 0.263 (T4 -15%, T3 +10%), acceptance unchanged. The gain concentrates in the first layers (layer 0 -33%, later layers about -1% each on their input's carried error); a loss on the output distribution rather than hidden-state MSE, or code-heavier calibration, are the open levers |
| 18 | Tier 1 with codes clamped to -3..3, transcoded to P2E2, after raising P2E2's fast-path limit (about 30 BIG per lane, 48-word window) | `q3-compressed` | converter entry, training, a kernel constant | the P2E2 form at or below `q3`'s +0.170 at about 2.65 bpw: 1.56 GiB freed, about 90K tokens of KV | same as row 17 plus days for the fast path | dNLL at or under +0.170 with the P2E2 ratio at or under 0.85; fast path verified against the scalar reference on RTN and trained artifacts | 07, 08 | TODO |
| 19 | MTP drafter fine-tune: dump post-final-norm hiddens and greedy tokens on 5 to 20 M tokens of the deployed artifact's own generations, train the block unrolled to depth 4 in NVFP4 fake-quant, re-screen depth | all (speed) | converter (replaces `mtp/*`) | speed only; HASS precedent +8.47% throughput | 1 to 2 days; 0.5 to 1.6 h per 10 M tokens | +0.2 tokens per verification at the shipped depth or a deeper depth winning at 32K; output hashes equal greedy on every prompt | 08 | TODO |
| 20 | Integer bitshift trellis form: Python reference and ablation first (with and without Hadamard, with and without LDLQ, Q3G64 plus LDLQ as control), then `trellis::decode_rows<M>` microbenchmark, expansion plus INT8 IMMA or two-term MXFP8 prefill, then artifact and engine | fifth form; `q3` inherits Q3G64 plus LDLQ | format (one format and one layout ordinal), kernels, converter | 2.5 bpw: 7.7 GiB streamed (one-row floor 11.5 ms, 1.39x), payload 8.7 GiB (3.0 GiB freed, 128K plus a resident DFlash2 drafter), KL 0.18 to 0.25, dNLL +0.07 to +0.10; 3.0 bpw: 9.1 GiB, 1.6 GiB freed, KL 0.11 to 0.15 | 1 to 2 weeks for the reference, 1 week each for decode and prefill, 2 to 3 weeks for the engine; 4 to 8 engineer-weeks; about 2 h per conversion | reference: 2.5 bpw KL at most 0.18, 3.0 at most 0.147, and at least 0.05 better than Q3G64 plus LDLQ (else ship Q3G64 plus LDLQ and stop); decode: one-row `gate_up` at most 1.15x its bytes over 855 GB/s and M = 8 at most 1.15x M = 1; engine: decode at least 1.3x `q3` at one row, TTFT at 128K at most 10% over `q3` | 03 | TODO |
| 21 | Q3 scale-plane table: 256-entry per-tensor table with codes re-rounded, or the 12 to 13 bit exponent-table index | `q3`, `q3-compressed` | format (layout ordinal), GEMV and twins, MX producer, P2E2 variant, reference | -0.37 GiB streamed (3.5% decode or about 21K tokens of KV) for +0.36% MSE; lossless variant -0.12 to -0.16 GiB | 1 to 2 weeks | converter study: MSE under 1% on every tensor and no clipped maximum; bitwise expansion test; tok/s gain at 4K and 128K; dNLL within noise | 04, 07 | TODO |
| 22 | FP6 E2M3 G64 for `gdn/value_z` and `attention/gate_value` (nibble plane plus 2-bit plane, 64-entry E4M3 expansion table) | `q3` | format and kernels (Q6-style GEMV and twins) | MSE on those tensors down 4 to 8x for +0.42 GiB; only if row 2 shows the Q5 families own more than 0.02 nats | 2 weeks | dNLL recovers most of what F16 recovered in row 2; decode slowdown under 4%; 128K resident or under 256 MiB host-backed | 04 | NO-GO (row 2: the Q5 families own under 0.004 KL) |
| 23 | INT3 keys and values (112-byte rows, 14 KiB per token), then INT3 keys with INT2 values (12 KiB), with a long-context harness (needle at 9 depths at 32K, 64K, 128K; a RULER subset at 64K) | `nvfp4-compressed` first, then all | format (KV-format field in the profile), codec, four attention kernels and twins, planner and prefix cache | 128K: KV 2.25 to 1.75 or 1.50 GiB; `nvfp4-compressed` host-backed 1.97 to 1.47 or 1.22 GiB, decode 52 to about 70 or 80 tok/s; `q3` +4 to +12% at 128K | weeks | row 4 quality intact; 128K retrieval within 1 point of E2M1; `nvfp4-compressed` at 128K at least 15% faster; a chain-of-thought task before any 3-bit cache ships | 06 | TODO |
| 24 | Single-term MXFP8 activations on the Q3 prefill route (drop the `lo` term), no rotation | `q3`, `q3-compressed` (prefill only) | kernel variant | gate/up and down 1.3 to 1.7x; about +0.01 KL; TTFT at 128K is 95 to 133 s today | 1 to 2 weeks | gate/up at least 1.3x on 512-row chunks and KL up by at most 0.01 through the new route; report TTFT at 32K and 128K | 02 | TODO |
| 25 | R1-only rotated artifacts (signed H_20 x H_256 folded into embedding, norms, norm-fed inputs, residual outputs, heads; uniform importance on rotated inputs), with per-tensor BIG-code and kurtosis dumps before and after | `q3`, `nvfp4` | converter | none to small on `q3`, likely negative on `nvfp4`; closes the weight-only rotation question and tests the near-Gaussian inference | 1 day, two conversions | KL down 10% (`q3` to at most 0.264, `nvfp4` to at most 0.132) or dNLL beyond 2 s.e.; otherwise stop the track and leave rotation to rows 12 and 20 | 02 | TODO |

Rented end-to-end QAD (0.1 to 1 B tokens, $50 to $650) follows row 17 only if it saturates, for perhaps another 0.01 to 0.03 nats
(chapter 08); the outlier side plane (chapter 05, probe 10) enters only if row 14's `--report-error` pass finds tensors whose top
1% of columns carry over 30% of the weighted error.

## What not to do

The engine's own rejected list stands (baseline, section 7) and the chapters extend it. Each row carries the number that closes it.

| Dead end | Evidence | Source |
|---|---|---|
| SD4 table of the 16 most frequent scales | clipped outlier blocks by up to 8x; NLL 4.32 against 2.23 | baseline |
| SD4 scales 2 codes below round-to-nearest | NLL -0.003 (noise); largest weights clipped by up to 14% | baseline |
| One activation term for native NVFP4 prefill | output RMS error 9.5% against 0.88%; KL doubled; the 9.5% is the Gaussian floor of E2M1 at block 16 and rotation does not lower it | baseline, chapter 02 |
| Relaxed native FP4 verification | 4% faster; changed every output | baseline |
| GDeflate or nvCOMP ANS over Q3 codes; P2E2 decode inside the prefill producer | 0.885 of row-split, 12x slower than P2E2 expansion; producer decode at 0.5 to 0.6x | baseline |
| Deeper host-staging rings; per-tensor pinned allocations; a Q3 MTP layer; lossless coding of NVFP4 codes | 4 KiB pages fall to 25 to 30 GB/s; Q3 drafts accepted fewer tokens; codes carry 3.90 of 4 bits | baseline |
| Weight-only rotation of the NVFP4 forms | block-16 grids gain nothing; the calibrated search loses its weights (toy: 0.44 to 0.47% error with the imatrix against 0.73 to 0.78% rotated) | chapter 02 |
| Learned rotations (SpinQuant, OSTQuant) or FlatQuant locally | need the quantized 27B model in the loop, which 16 GB cannot hold in BF16; FlatQuant adds an online transform before every linear | chapter 02 |
| The full-range P2E2 layout as the route to a compressed calibrated `q3` | 92 to 94% of row-split bytes at 1.7x the decode instructions; about 25% below `q3` decode | chapter 07 |
| rANS, tANS, Huffman or Tunstall decoders inside the GEMV; escape coding of Q4 and Q5 codes | 10 to 22 instructions per pair or 0.25 bpw of per-lane offsets; Q4/Q5 escapes lose under calibration; the Q5 fifth bit is the sign bit at 1.0 bit of entropy | chapter 07 |
| A uniform 2-bit MLP | 0.3 to 0.8 nats on heavily trained models even after QAT; EXL3 at 2.0 bpw reads KL 0.164 on this model against 0.075 at 2.5 | chapters 03, 08 |
| 2:4 sparse FP4 at 3.0 bpw | 0.5 to 1.1 nats one-shot; 0.3 to 0.4 with learned masks that took about 2 B tokens and hundreds of H100 hours on 7 to 8 B models; dense Q3 costs 0.170 | chapter 04 |
| A non-uniform 16-entry decode grid; SVDQuant's branch on the weight path; block-8 NVFP4; FP8 weights for the input projections | 0 to 11% MSE for a 3 to 5x slower prefill; +0.40 GiB per token at rank 32 and a flat weight spectrum beyond rank 256; no native MMA and worse than Q5G64 on Gaussian weights; +1.26 GiB | chapter 04 |
| Low-rank error branches at 3 bits | rank 32 is +390 MiB and +3% decode for a gain the trained scale plane already covers | chapter 08 |
| A BF16 MTP layer; an MTP layer at Q5G64; a 2-bit LM head or embedding; a 65,536-row draft head | +0.57 GiB per draft row and about +5 to 10% step time against +2.9% tokens per verification for the 3.25-to-4.5-bit step; catch-up rows 2.0 against 1.4 ms; 0.15 GiB each with new kernels on a sensitive head; -4.2% / -6.9% accepted drafts | chapter 05 |
| BF16 storage of the GDN state | saves at most 1.2% of a token and 72 MiB; steady-state error 1.4 to 4% of the state (estimate) | chapter 06 |
| A uniform 2-bit KV cache or 2.5-bit keys for a reasoning fine-tune at 64K to 128K; FP4 attention for prefill | KIVI-2 loses 8.5 points on passage retrieval; TurboQuant 3-bit drops AIME25 from 0.683 to 0.443; attention is at most 2% of TTFT | chapter 06 |
| Large per-layer VQ codebooks (AQLM, VPTQ); FP16-valued trellis generators; AQLM-style codebook learning; GSQ at its default data volume locally | latency-bound L2 lookups; EXL3's `mul1` has 10 significant bits, not BF16-exact; days per model; about 150 hours on this card (estimate) | chapters 01, 03 |
| Pushing `gdn/output` (`ssm_out`) lower | the most KLD-sensitive tensor in the one per-tensor sweep on this family; pinned at Q5_K to Q8_0 by every community build | chapters 05, 09 |

## Open questions and gaps

- The weight-only share of every form's KL is unknown; the baseline's numbers bundle the NVFP4 KV cache, relaxed kernels and, for
  NVFP4 forms, two-term FP4 prefill activations; roadmap row 2 is the first measurement (baseline, chapters 01, 06, 09).
- Swift 1.5: the card is reported to say the published weights include the base model's MTP head, but whether the LoRA touched
  that block is not stated (baseline, chapters 08, 09); a Swift `q3` has no measurement, so the `q3` probes run on the base
  lineage and must be confirmed on Swift before shipping (chapters 01, 05, 08); the 58.5% thinking reduction is a median, the mean
  about 41% (chapter 09).
- The GDN sensitivity evidence is one 2026 preprint whose mechanism an independent review calls only partly supported; the
  long-context behaviour of 4-bit DeltaNet projections is contested without numbers (chapters 05, 09).
- Chapters 02, 04, 06 and 07 rest on synthetic Gaussian, Laplacian and Student-t weights, which the converter can replace with the
  real tensors in minutes (chapter 04). Argued but not run: signed-scale kernel compatibility (chapter 01); loosening P2E2's
  16-BIG fast-path limit to about 30 (chapter 07); `decode_rows`'s code-rate headroom at 2.5 bpw (chapter 03); a Q4 or Q5 linear
  inside an NVFP4 artifact (chapter 05).
- Whether a matrix containing Swift's own thinking-mode generations beats UkisAI's v1mix for the deployed form is untested
  (chapters 01, 08, 09).
- The web-search budget ran out in chapters 01, 03, 07, 08 and 09, and huggingface.co, arxiv.org and the PTX ISA page were
  unreachable, so Hugging Face cards were read through search summaries and these sources were not read:

| Chapter | Not retrieved |
|---|---|
| 01 | the AWQ paper's tables and its AWQ plus GPTQ claim; Four Over Six PTQ tables; GPTAQ, MagR and LeanQuant tables; "Pretraining LLMs with NVFP4"; any 3-bit number on Qwen2.5 or Qwen3-32B; the ISTA and UkisAI model cards |
| 03 | QTIP's 70B 3- and 4-bit cells; SqueezeLLM's 13B and 65B rows; Any-Precision perplexities; Proteus and CurveTQ beyond abstracts; an EXL3 decode measurement on a 4090 or 5090 at a named model |
| 04 | Four Over Six's own tables; the llm-compressor result that an expanded MSE search beats 4/6; the full text of the GDN NVFP4 paper; the PTX ISA page for the sparse FP4 pattern |
| 06 | the TurboQuant table and the vLLM study, cited only as relayed by community READMEs; the LMSYS Qwen3.8-27B NVFP4-KV table; any RULER-128K table for a 3-bit cache |
| 07 | DFloat11, ZipNN and NeuZip throughput tables; 2025 to 2026 in-kernel entropy decoders |
| 08 | OmniQuant's and BitDistiller's per-model tables; PV-Tuning's GPU hours; Medusa's training time; the QUASAR checkpoint; Qwen3-Next's MTP training notes |
| 09 | UkisAI's GSQ-RCO card (bit widths and quality of the most comparable public quantization of the served checkpoint); bartowski's full percentile table and its IQ3_S and IQ4_XS rows; unsloth Dynamic 3.0; the ISTA imatrix text; the Swift Open License text; Swift 1.5 sampling guidance |

- Stale engine docs the chapters and the baseline flagged, verified read-only against the checkout at commit 269a890:

| Document | Says | Current state | Source |
|---|---|---|---|
| `docs/MTP_CONTRACT.md`, `docs/MTP_SPECULATIVE.md` | MTP depth 3 on NVFP4 artifacts | `ArtifactProfile.draftLength` returns 2 for Q3 artifacts and 4 otherwise | baseline |
| `docs/ATTENTION_DECODE.md`, `docs/nvidia/tensor-cores.md` | an E2M1 x E4M3 product has at most five significant bits | six (2 for the E2M1 magnitude times 4 for E4M3), as `docs/NVFP4_NATIVE.md`, the baseline (section 5) and chapter 06 state; the row-exactness argument rests on it | baseline, chapter 06 |
| Any text predating PR 119 that attributes all four forms to the base checkpoint and ISTA-DASLab's `imatrix-qwen3.8-27b.gguf` | one lineage | two lineages: `q3` and `nvfp4` are base conversions with ISTA's matrix and `q3-compressed` a round-to-nearest base conversion, all scored against the base BF16; `nvfp4-compressed` is a Swift-1.5 conversion with UkisAI's `imatrix-swift15-v1mix.gguf` scored against Swift's BF16; the ISTA matrix's text is unknown and UkisAI's is a third its size (1.27 M against 4.1 M tokens), a confound whenever the lineages are compared | baseline, chapter 09 |

- The measurement protocol to adopt (chapter 09, section 6; holdout sizing from chapter 08, section 4.5):
  1. Four held-out texts: T1 the current document, T2 wikitext-2 `wiki.test.raw` (the only text with public KL at every width), T3
     code from outside the engine repository, T4 BF16 Swift-1.5 generations with thinking traces and template tokens; 32K scored
     tokens each (8 to 16 documents of 2K to 4K tokens, document-level s.e. near 0.005 nats on Q3-class forms).
  2. Every form converted from and scored against Swift-1.5 BF16 (`llama-perplexity --kl-divergence-base` at `-c 2560`); base
     conversions only as the bridge to public numbers on T2.
  3. Per text: mean KL with s.e.; median, 95%, 99% and maximum KL; paired dNLL with s.e.; same-top-1 with binomial s.e.; mean and
     RMS delta-p with 1% and 99% percentiles.
  4. Attribution per artifact: weight-only KL (dequantized GGUF, F16 KV), the drift floor, the full engine number.
  5. Thresholds: under 0.01 KL lossless; 0.010 to 0.020 the Q4_K_M class (top-1 95 to 96%); 0.05 to 0.08 the Q3_K_M class (about
     92%); above 0.15 the 2-bit class; a change counts only above two standard errors on T2 and T4 with the same sign on all four
     texts.
  6. Long context: teacher-forced KL after 32K, 64K and 128K prefixes and a needle set at nine or more depths, before and after
     any KV or GDN-state change; KL rising faster with depth than the reference's own NLL is a GDN or KV defect, not quantization.
  7. Downstream gates: GSM8K (s.e. 0.6 pp), a fixed 2,000-question MMLU-Pro subset (0.8 pp), LiveCodeBench v6 for code claims,
     AIME25 only with five seeds; completion and thinking tokens (mean, median, p90) against BF16 Swift on every run.
  8. Drafting stack: MTP tokens per verification with Swift's MTP block and with the base model's block grafted in.
  9. Record per imatrix and artifact: checkpoint hash, llama.cpp commit, matrix name and SHA-256, calibration text identity and
     token count, `--ctx-size`, chunk count, `--parse-special`, `--nextn`, `--process-output`, `llama-imatrix --show-statistics`.
