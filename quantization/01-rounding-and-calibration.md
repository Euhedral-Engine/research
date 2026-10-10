# Rounding and calibration inside the existing formats

Scope: methods that change *which codes and scales the converter writes* into Q3G64_F16S, Q4G64_F16S, Q5G64_F16S, NVFP4 and
NVFP4-SD4 without touching formats, layouts or kernels: error feedback (GPTQ/OBQ, GPTAQ, MagR, LeanQuant), learned rounding
(SignRound/AutoRound, BRECQ-style block reconstruction, OmniQuant clipping), channel scaling (AWQ, SmoothQuant), FP4 scale search
(Four Over Six, ModelOpt's E4M3 sweep and local-Hessian weighting), llama.cpp's imatrix quantizers, ISTA-DASLab's GSQ-RCO, and
calibration data. Everything is measured against `00-baseline.md`. My own estimates are marked "estimate"; everything else carries a
source. The web search budget of this session ran out after one round, so some sources named in the brief (the AWQ paper's tables, Four
Over Six's PTQ tables, GPTAQ's tables, "Pretraining LLMs with NVFP4", 3-bit numbers on Qwen2.5/Qwen3-32B) could not be retrieved and are
listed as gaps.

## 1. Summary

The converter already does one of the three things this family does: a per-group clipping and stretch search weighted by a diagonal
Hessian proxy (the imatrix), which took `q3` from +0.275 to +0.170 nats against BF16. It does not do the other two, cross-column error
feedback (GPTQ) and per-input-channel scaling (AWQ), and it does not use the full code range the way llama.cpp does (a signed binary16
scale puts every group's largest magnitude on code -4). Published 3-bit results on 13B to 70B dense models put RTN, GPTQ, AWQ, OmniQuant
and AutoRound in a narrow band: at W3 g128 on Llama-2-13B the WikiText2 perplexity is 5.51 / 5.42 / 5.32 / 5.28 (RTN / GPTQ / AWQ /
OmniQuant) [3] and the 11-task zero-shot average is 58.57 / 59.58 / 58.86 / 59.18 / 59.44 (RTN / GPTQ / AWQ / OmniQuant / AutoRound)
against 61.42 in FP16 [4]; the calibrated methods recover 30 to 60 percent of the RTN loss and none dominates. The most model-specific
number found: on Qwen3.8-27B itself at W4 g128, AutoRound's own imatrix-weighted scale search gives KL 0.0285 to BF16 and SignRound tuning
lowers it by 23 percent (symmetric, 57 minutes on an RTX 3090) to 33 percent (asymmetric, 200 iterations, 3 hours) [4]. Verdict: for `q3`
the chain "signed scales, then GPTQ with the existing group search as its quantizer, then AWQ scales folded into the norms" should cut the
weight-rounding error by roughly 25 to 40 percent (estimate: +0.170 to about +0.11 to +0.13 nats, KL 0.293 to about 0.20 to 0.24) for one
calibration pass of 30 to 90 minutes on the 16 GB card plus about an hour of converter time; learned rounding can add another 10 to 20
percent for 1 to 3 GPU hours. For `nvfp4` and `nvfp4-compressed` the same chain applies to the E2M1 grid, but published gains at 4.5 bpw are
small (AutoRound on NVFP4 Qwen3-32B: 0.7164 to 0.7306 average accuracy [4]) and the harness cannot resolve them on one document; the one free
improvement is widening the E4M3 scale search above round-to-nearest, worth a further 10 percent of weight MSE in my simulation.
`q3-compressed` cannot carry code -4 and keeps at most about three quarters of the calibrated gain (estimate); it profits least.

## 2. Mechanism

### 2.1 The current search in these terms

For weights W (N x K) and inputs X (T x K), every method minimizes `||(W - Q(W)) X^T||_F^2 = sum_rows e H e^T`, `H = X^T X`. The
imatrix is `diag(H)/T` (llama.cpp `in_sum2 / counts`), so `search_group_codes_torch` minimizes `sum_j H_jj e_j^2`: the diagonal
approximation, one scalar per 64 columns, 39 candidates. Left on the table: the off-diagonal of H (error feedback), per-column step sizes
(channel scaling) and one code of the grid (2.4).

### 2.2 Error feedback: GPTQ, GPTAQ, MagR, LeanQuant

GPTQ [1] (`gptq/gptq.py:fasterquant`): accumulate `H = 2/T sum x x^T` in FP32; damp by `0.01 x mean(diag H)`; upper Cholesky factor of
H^-1; walk columns in blocks of 128: `q = Q(w_i)`, `err = (w_i - q) / Hinv[i,i]`, `W[:, i:] -= err (x) Hinv[i, i:]`, then propagate the
block's errors to the remaining columns. With groups, the scale is set when the loop reaches a group's first column from the already
compensated weights (`static_groups=False`) or beforehand (`static_groups=True`, needed with `act_order` so groups stay contiguous [1]). Rows
are independent given Hinv, so the 248,320-row LM head runs in row chunks. Defaults: 128 x 2048 C4 tokens, `percdamp 0.01`, `blocksize 128`
[1]. In Euhedral `Q(.)` is the existing 39-candidate search over the 64 compensated columns, so format, group geometry and binary16 scale are
untouched; for NVFP4 the E4M3 block scale (2.6) is chosen when the loop reaches each 16-column block.

GPTAQ ("GPTQ v2", arXiv 2504.02692 [5]) matches the quantized layer's output to the *full-precision model's* output: it accumulates
`dXXT = (X_fp - X_q) X_q^T` next to H and adds `P = alpha x triu(dXXT Hinv^T, 1) Hinv`, alpha 0.25 (`gptaq_utils.py`). It needs BF16 and
partially quantized activations per layer, i.e. a sequential pass; GPTQModel ships it as experimental, "2-4x more VRAM", "results may not be
better than original" [17]. MagR (NeurIPS 2024, arXiv 2406.00800 [8]) pre-processes W by proximal gradient on `||W||_inf` while preserving WX
(alpha 0.0001 per group, 200 iterations) before GPTQ; in-format, but a later reproduction dropped its 13B W3 g128 result as anomalous [10].
LeanQuant (ICLR 2025, arXiv 2407.10032 [9]) changes the grid (non-uniform, or affine with a zero point): not representable here.

### 2.3 Channel scaling: AWQ, SmoothQuant

AWQ [2] multiplies input channel j of W by `s_j` and divides the activation by `s_j`, giving salient channels a finer effective step under a
shared group scale. `auto_scale.py`: `s = x_max^ratio`, 20 ratios in [0, 0.95], `x_max` the mean |x_j| over calibration tokens, chosen by the
output MSE of the *quantized* scaled weights; `auto_clip.py`: per-group max shrink, 20 steps down to 0.5, on 512 sampled tokens. The scale folds
into whatever produces the input: a norm for q/k/v and gate/up (`scale_ln_fcs`), the rows of `v_proj` for `o_proj` and of `up_proj` for
`down_proj` (`scale_fc_fc`). AWQ is a pre-processing step and composes with any quantizer: AutoRound runs `alg_configs=['awq','auto_round']`
[4], llm-compressor's Qwen3.8-27B NVFP4 recipe runs `AWQModifier` then `GPTQModifier` [16]. SmoothQuant moves outliers the other way, into the
weights, for activation quantization; for weight-only formats it is counterproductive and matters only to the native NVFP4 prefill's two-term
activation quantization, where AWQ's scaling already smooths activations.

Exact folding into a BF16 `(1 + w)` norm: store `w' = bf16((1 + w) s - 1)`, read back `s_real = (1 + w') / (1 + w)`, and divide the
weight columns by `s_real` before quantizing; the BF16 rounding of the norm then costs nothing.

### 2.4 The eighth code: signed scales and the P2E2 fallback

llama.cpp's `make_qx_quants` (ggml-quants.c:628) sets `iscale = -nmax / max` with `max` the *signed* value of largest magnitude: that value
always lands on code -4, and when it is positive the scale is negative. Q3G64_F16S's binary16 scale carries a sign; the Q3 decode kernel
converts it with `__half2float` and multiplies with `fmaf` (native/src/q3/contiguous.cuh:76,94), the reference kernel likewise
(native/src/reference/kernels.cu:38), P2E2 copies the scale plane byte for byte (q3_p2e2.py:120), and the MXFP8 prefill applies the scale in
FP32 after each group (baseline, section 5). Nothing I read takes an absolute value, so negative scales look format- and kernel-compatible;
only the converter forbids them (`_binary16_scales` clamps to [0, 65504]). AutoRound's symmetric quantizer uses the same trick and credits
"the llamacpp community" (`auto_round/data_type/int.py`); its docs show "full-range sym" beating GPTQ's symmetric convention by 8.8 points at
W2G32 on Llama-3.1-8B-Instruct and matching asymmetric within 0.3 points at W4G128 [4]. GPTQ's `sym=True` (`zero = 4`, step `2 amax / 7`) is a
third grid that clips positive values above 0.86 amax [1, quant.py].

The P2E2 fallback (-3..3) loses the code entirely: 6 intervals instead of 7, so the step grows by 7/6 and the rounding MSE by about 1.36 in the
groups that used -4. Measured with the converter's actual search on synthetic groups (Table G): restricting to -3..3 forfeits about one quarter
of the calibrated MSE reduction; signed scales add 9 to 13 percent on top of today's search. Published symmetric-versus-asymmetric numbers at 3
bits: none found. At W4 g128 on Qwen3.8-27B AutoRound's symmetric search is 6 percent worse in KL than its asymmetric one (0.02734 vs
0.02579), 4 percent after tuning [4].

### 2.5 Learned rounding and clipping

SignRound/AutoRound [4] learns `W_q = s x clamp(round(W/s + V), qmin, qmax)` with per-weight offsets V in [-0.5, 0.5] and per-group scale
multipliers by signed-gradient descent on the block output MSE: 128 x 2048 tokens (pile-10k), 200 iterations, lr 1/iters; `low_gpu_mem_usage`
offloads activations (+30 to 100 percent time). It is the modern BRECQ/AdaRound. `iters=0` is an "optimized RTN" scale search weighted by an
automatically collected imatrix, the analogue of Euhedral's search; `--enable_neuqi` adds a signed two-stage scale search (arXiv 2505.17595
[10]); `enable_alg_ext` (SignRoundV2) is the stronger tuner. AutoRound exports GGUF K-quants in-format and has `nv_fp4` (global scale
`448 x 6 / amax`, group 16) and MXFP4 data types. Costs on an A100, W4G128, torch.compile: 14B 23 min / 22 GB, 70B 120 min / 42 GB; with
`low_gpu_mem_usage`, `gradient_accumulate_steps=8, batch_size=1`: 14B 45 min / 7 GB, 70B 187 min / 19 GB [4]. Qwen3.8-27B W4 g128 on an RTX
3090: iters=0 9 to 13 min, 50 iterations 49 to 57 min, 200 iterations 2 h 57 min [4]. OmniQuant's LWC [3] learns a sigmoid clipping factor per
group (Adam, lr 1e-2, 20 epochs, 128 samples); it yields an ordinary scale and is in-format; its LET targets W4A4 and is irrelevant here.

### 2.6 FP4 scale search

ModelOpt [15] calibrates NVFP4 block scales by `mse` over amax multipliers 0.25 to 4.0 (step 0.1) or by `fp8_scale_sweep`, all 128 E4M3 codes
("the recommended setting for NVFP4"), optionally weighted by `local_hessian`: the block's 16 x 16 `H = X X^T`, loss `dw H dw^T`
(`config.py:LocalHessianCalibConfig`). Euhedral's search is the sweep restricted to -6..+2 codes with diagonal weights; the block Hessian is a
strict refinement needing K/16 x 256 floats per linear. Four Over Six (arXiv 2512.02010 [6]) tries two scales per block, block maximum to 6 or
to 4, and keeps the lower MSE (`ScaleRule.mse`, `fouroversix/utils.py`); since max-to-4 needs a 1.5x larger block scale, ModelOpt and the
reference code lower the per-tensor normalizer from 448 to 256 (`E4M3_MAX_46 = 256`, `numeric_utils.py:41`). In Euhedral: global scale
`amax / (6 x 256)` and a block search up to 5 or 6 codes above RTN; the kernels apply the global scale once per output (`nvfp4.cuh:172`), so
nothing else moves. ModelOpt also learns block scales (LSQ, Dual-LSQ) and ships `NVFP4_AWQ_LITE/CLIP/FULL` (alpha 0 to 1, step 0.1). The
MR-GPTQ paper (arXiv 2509.23202 [7]) reports that for NVFP4 plain GPTQ gives the highest recovery among its post-training methods and that
large models recover 98 to 99 percent of FP16 accuracy, while MXFP4 needs its micro-rotations; only abstract-level numbers were retrievable.

### 2.7 llama.cpp's imatrix quantizers

`quantize_row_q3_K_impl` weights each value by `imatrix_j x sqrt(sigma^2 + x^2)` (sigma^2 = 2 x mean square of the 256-block), then
`make_qx_quants(16, 4, ...)` per 16-value sub-block: signed scale, RTN plus 18 candidates `-(4 + 0.1 is)/max`, each refit by weighted least
squares, kept if weighted `sumlx^2/suml2` improves; the 16 sub-scales are quantized to 6 bits by the same routine. Q4_K uses `make_qkx3_quants`
(asymmetric with a min; `rmin -0.9, rdelta 0.05, 36 steps`). IQ2/IQ3 use 8-value codebooks, not representable here. `llama-imatrix`: chunk =
context length, `--chunks`, `--process-output` (also `output.weight`; off by default because the authors found it "typically better not to"),
`--nextn` (MTP statistics; needs one sequence per batch and an MTP with its own KV cache, which Qwen3.8 has per docs/MTP_CONTRACT.md),
`--parse-special`, `--in-file` merges [14]. The converter's search is this algorithm with a positive scale, 64-groups and no sub-scales.

### 2.8 GSQ-RCO (ISTA-DASLab, 2026)

GSQ = Gumbel-Softmax Quantization (arXiv 2604.18556, github.com/IST-DASLab/GSQ [11]): post-training scalar quantization that keeps a
symmetric group-wise integer grid (INT kernels, GGUF K-quants in-format) and learns every weight's discrete assignment. Per layer: GPTQ
initialization; per-weight logits over the grid (2-bit `{-2,-1,0,1} x s`, ternary, or an n-bit grid around the init) sampled with
Gumbel-Softmax, temperature annealed 2 to 0.05, logit scale 100 to 500, Lion with cosine LR, reconstruction loss `||f(x; w~) - f(x; w)||^2`,
bf16 logits; finished layers offloaded to `meta`. Default config: `open_thoughts` reasoning data, 4096 x 4096 tokens, 10 epochs, group 128,
GPTQ init with `percdamp 0.1` and 512 samples. Reported: Llama-3.1-70B-Instruct at 2.13 bpp, zero-shot average 75.57 vs GPTQ 57.38, QTIP
77.25, FP16 78.99; Qwen3-8B Q2_K GGUF refined in-format 50.03 to 56.28; Llama-3.1-8B on 1 x H100, 70B on 4 x H100 [11]. RCO = Riemannian
Constrained Optimization (arXiv 2605.00649, github.com/IST-DASLab/RCO [12]): one of K options (bit widths, GGUF types) per group (layer,
tensor) under an exact size budget, Adam on the budget manifold (tangent projection, retraction, vector transport) with a Gumbel-STE forward
pass solved by a knapsack DP; reported to match or beat EvoPress at 3 to 16x lower wall-clock. The Qwen3.8-27B release (IQ2_XS to IQ3_S, 8.4 to
11.8 GB, the 1000 x 4096-token imatrix) uses GSQ per tensor and RCO to pick each tensor's type under a budget [13]; community threads report
IQ3_XXS at about 10 GB keeping about 99 percent of the full model's scores, unverified (Hugging Face is unreachable here). UkisAI's Swift-1.5
GSQ-RCO GGUF reuses ISTA's allocation and ships `imatrix-swift15-v1mix.gguf`: 309 x 4096 tokens (about 1.27 M) of web, reasoning,
multilingual and code text, the matrix behind `nvfp4-compressed` [13]. In the converter: GSQ is in-format for Q3/Q4/Q5 and a heavier cousin of
AutoRound; at its default data volume it is out of reach on one 16 GB card (estimate: 6 x 0.4 G params x 1.7e8 token-passes x 10 epochs x 64
layers = 2.6e19 FLOP, about 150 hours at 50 TFLOPS), with 128 x 2048 tokens a few hours; its headline gains are at 2 bits where the GPTQ-to-VQ
gap is large, at 3 bits the gap is small (section 3). RCO belongs to the bit-allocation chapter but would run on the existing Q3/Q4/Q5 ladder.

### 2.9 Calibration data

In use: GPTQ 128 x 2048 C4 [1]; AWQ 128 x 512 Pile [4]; AutoRound 128 x 2048 pile-10k (512 in its paper) [4]; OmniQuant 128 x 2048
WikiText2 [3]; FP-Quant 128 x 2048 fineweb-edu [7]; llm-compressor's Qwen3.8-27B recipe 512 x 4096 "perfectblend" [16]; GSQ 4096 x 4096
open_thoughts [11]; RCO offers chat sets with answer-only loss masks [12]; Unsloth's imatrix sets 300 K to 1.5 M tokens of curated chat-oriented
text [22]; ISTA's Qwen3.8-27B imatrix 4.1 M tokens; UkisAI's 1.27 M. The llama.cpp imatrix discussion found text to beat pseudo-random data and
language-specific text to help when one language dominates [14]. Reasoning models: Liu et al. (arXiv 2504.04823 [18]) find W4 weight-only near
lossless and W3 costly (over 7 percent on LiveCodeBench for AWQ and GPTQ); arXiv 2606.00206 [19] calibrates GPTQ on *model-generated* reasoning
traces to reduce domain shift and finds 3-bit AWQ inflating overthinking errors from 19 to 139; LAQuant (arXiv 2605.08755 [20]) reports large
Pass@1 gains from swapping generic text for DeepSeek-R1 traces under the same GPTQ; for pruning, self-generated data helps reasoning models
(arXiv 2511.18864 [21]). The deployed artifact is converted from Swift 1.5 with UkisAI's mixed matrix, so there is no base/fine-tune mismatch
today; the open question is whether a matrix containing Swift's own thinking-mode generations does better. The evidence says yes for
reasoning-heavy use and "keep a mix" for general text.

### 2.10 Sequential Hessian collection on one 16 GB GPU

The 54.7 GB checkpoint never needs to be resident. Stage 1, one pass: stream layer l's BF16 weights from the safetensors mmap (about 0.85 GB
per layer); run T calibration tokens through it in micro-batches of whole sequences (the GDN recurrence needs them; use the Hugging Face Qwen3.5
implementation with its chunked scan kernels); hook each linear's input to accumulate `H += X^T X` in FP32 (gate_up and the attention/GDN
input projections share K = 5120: 105 MB; down K = 17408: 1.2 GB; o_proj/out_proj K = 6144: 151 MB), plus the diagonal (a Swift-1.5 imatrix
for free) and the 16 x 16 block diagonal; replace the input buffer by the layer output (T x 5120 BF16: 2.7 GB at T = 262 k, on the GPU or
pinned) and continue. Also collect the final-norm output (LM head) and the MTP inputs for the drafting stack. Compute: forward 2 x 25.6 G x T
and Hessian accumulation 2 x sum K^2 x T, each about 1.3e16 FLOP at T = 262 k; at 40 to 60 TFLOPS achieved 8 to 12 minutes, plus the GDN scan
and Python overhead: budget 30 to 60 minutes at T = 262 k, 2 to 4 hours at T = 1 M (estimate). Peak device memory: 0.85 + 1.5 + 2.7 GB plus
micro-batch temporaries (8 k tokens x 34816 x 4 B = 1.1 GB), under 7 GB. Persisting all H in FP32 costs about 94 GB of disk (estimate, 64 x
1.47 GB); the alternative quantizes layer by layer inside the pass (standard GPTQ, enables quantized-input propagation and GPTAQ). Stage 2 in the
converter: per object `O(N K^2 + K^3/3)`, 0.1 to 1.5e12 FLOP, in practice 2 to 10 s because the column loop is sequential (5120 to 17408
steps); LM head in 16 k-row chunks; 20 to 60 minutes total (estimate). AutoRound-style tuning instead: 1 to 3 hours for 50 to 200 iterations,
extrapolating the 3090 run [4]; memory about 10 GB with the saving settings (70B needs 19 GB [4]; estimate).

## 3. Published results

Table A. W3 g128 WikiText2 perplexity, Llama-2, as tabulated in the OmniQuant paper [3]. FP16 references 5.47 / 4.88 / 3.32 are the usual
values, not re-verified here; nats = ln(PPL / FP16).

| Model | RTN | GPTQ | AWQ | OmniQuant | RTN to GPTQ | RTN to OmniQuant |
|---|---|---|---|---|---|---|
| Llama-2-7B | 6.66 (+0.197 nats) | 6.29 (+0.140) | 6.24 (+0.132) | 6.03 (+0.097) | -29% | -51% |
| Llama-2-13B | 5.51 (+0.121) | 5.42 (+0.105) | 5.32 (+0.086) | 5.28 (+0.079) | -13% | -35% |
| Llama-2-70B | 3.97 (+0.179) | 3.85 (+0.148) | not reported | 3.78 (+0.130) | -17% | -27% |

Without grouping (per-channel W3) the spread explodes (13B: RTN 10.68, GPTQ 6.44, AWQ 10.45, OmniQuant 5.58) [3]: per-group scales, which
Euhedral has, are most of the battle. A later paper re-ran AWQ at 6.19 / 5.30 / 3.72 [3, search note], so the AWQ column is uncertain by 0.1.

Table B. LLaMA-1 WikiText2 perplexity, GPTQ repository README [1].

| Model | FP16 | 3-bit RTN | 3-bit GPTQ | 3-bit g128 GPTQ |
|---|---|---|---|---|
| 7B | 5.68 | 25.54 | 8.07 | 6.61 |
| 13B | 5.09 | 11.40 | 6.63 | 5.62 |
| 30B | 4.10 | 14.89 | 5.69 | 4.80 |
| 65B | 3.53 | 10.59 | 5.04 | 4.17 |

Table C. Average of 11 zero-shot tasks, Llama-2, W3G128 (AutoRound paper tables; 512 pile-10k samples; GPTQ with act-order and
true-sequential; AWQ at length 512) [4].

| Model | FP16 | RTN | GPTQ | AWQ | HQQ | OmniQuant | AutoRound |
|---|---|---|---|---|---|---|---|
| Llama-2-7B | 57.98 | 53.81 | 54.14 | 55.21 | 54.31 | 54.72 | 56.68 |
| Llama-2-13B | 61.42 | 58.57 | 59.58 | 58.86 | 58.10 | 59.18 | 59.44 |
| Llama-2-70B | 66.12 | 64.08 | 65.08 | 65.12 | 64.80 | 65.12 | 65.31 |
| 13B at W4G128 | 61.42 | 60.65 | 61.00 | 60.91 | 60.65 | 60.51 | 60.85 |
| 70B at W4G128 | 66.12 | 65.87 | 66.22 | 66.23 | 66.06 | 66.02 | 66.39 |

Table D. 3-bit GGUF in-format (Q3_K_S, about 3.44 bpw): AutoRound's imatrix-weighted scale search versus learned rounding on top of it,
average of 10 tasks, percent of BF16 in parentheses [4].

| Model | BF16 | Optimized RTN | AutoRound + alg_ext | Loss recovered | Time (Opt RTN / AutoRound) |
|---|---|---|---|---|---|
| Qwen3-8B | 0.6322 | 0.6128 (96.9%) | 0.6252 (98.9%) | 64% | 1 m 29 s / 48 min |
| Qwen2.5-7B-Instruct | 0.6571 | 0.6382 (97.1%) | 0.6503 (99.0%) | 64% | 1 m 20 s / 36 min |
| Llama-3.1-8B | 0.6295 | 0.6040 (96.0%) | 0.6081 (96.6%) | 16% | 1 m 25 s / 30 min |
| Qwen3-30B-A3B-Instruct | 0.6746 | 0.6598 (97.8%) | 0.6622 (98.2%) | 16% | 25 min / 12 h 48 min |

Table E. FP4 and Qwen-family 4-bit results.

| Model, format | Metric | Search or RTN | Improved | Method | Source |
|---|---|---|---|---|---|
| Qwen3.8-27B, W4A16 g128 sym | mean KL to BF16, openwebtext, 65 k positions | 0.02852 (imatrix search) / 0.02734 (NeUQI) | 0.02191 (50 it.) | SignRoundV2 | [4] |
| Qwen3.8-27B, W4A16 g128 asym | same | 0.02579 (NeUQI) | 0.02106 (50 it.) / 0.01921 (200 it.) | SignRoundV2 | [4] |
| Qwen3-32B, NVFP4 g16 | avg of 5 tasks, fake quant | 0.7164 | 0.7306 / 0.7295 (alg_ext) | AutoRound | [4] |
| Qwen3-32B, MXFP4 g32 | same | 0.6901 | 0.7211 | AutoRound | [4] |
| Qwen3-8B, NVFP4 | avg of 11 tasks; BF16 65.45 | 64.01 (RTN) | 64.40 | optimized RTN | [4] |
| Llama-3.1-8B-I, NVFP4 | same; BF16 65.17 | 62.65 | 63.91 | optimized RTN | [4] |
| Llama-3-70B, W4 g128 / W3 g128 | WikiText2 PPL; FP16 2.9 | | 3.3 / 5.2 | GPTQ | [23] |
| Llama-3.1-70B-Instruct, 2.13 bpp | zero-shot avg; FP16 78.99 | GPTQ 57.38 | 75.57 | GSQ | [11] |

Table F. Does AWQ stack on a scale search or on learned rounding? AutoRound's composition study, average of 5 tasks [4].

| Scheme, model | opt_rtn | AWQ + RTN | AutoRound (alg_ext) | AWQ + AutoRound (alg_ext) |
|---|---|---|---|---|
| MXFP4, Llama-3.1-8B-I | 62.75 | 62.36 | 65.83 | 66.20 |
| MXFP4, Qwen3-8B | 68.03 | 69.23 | 69.75 | 70.18 |
| W4A16, Llama-3.1-8B-I | 69.04 | 69.66 | 69.70 | 69.95 |
| W4A16, Qwen3-8B | 71.91 | 71.45 | 72.05 | 72.10 |

Table G. Estimate: relative weighted MSE of the converter's rounding variants on synthetic groups of 64 (60,000 groups; Gaussian, Student-t
with 4 degrees of freedom, Laplace; uniform or log-normal, sigma 1.5, per-column weights), `search_group_codes_torch` re-implemented in numpy.
Not model data; the ordering is the result, not the digits.

| Variant | Gaussian, uniform | Student-t, uniform | Gaussian, weighted | Groups using -4 |
|---|---|---|---|---|
| RTN `amax/3` | 1.000 | 1.000 | 1.000 | 0% |
| current search (ends 3 and 4, positive scale) | 0.570 | 0.588 | 0.463 | 65 to 72% |
| current search without the -4 end | 0.616 | 0.630 | 0.550 | |
| P2E2 fallback: search restricted to -3..3 | 0.683 | 0.713 | 0.595 | 0% |
| signed scale (llama.cpp trick) | 0.497 | 0.494 | 0.421 | 95 to 100% |
| NVFP4 block search -6..+2 (current) | 0.815 | 0.865 | | |
| NVFP4 block search -6..+6 (covers the 4/6 choice) | 0.730 | 0.812 | | 29 to 40% of blocks go above +2 |

The repository's own RTN-to-calibrated step (+0.275 to +0.170 nats, -38 percent) matches the simulated MSE drop (-43 percent) closely
enough to use "nats proportional to weighted MSE" as a first-order rule below.

Gaps: the AWQ paper's 70B INT3 g128 value and its AWQ+GPTQ table, Four Over Six PTQ perplexities, GPTAQ, MagR and LeanQuant tables,
"Pretraining LLMs with NVFP4", any 3-bit measurement on Qwen2.5/Qwen3-32B.

## 4. Fit with Euhedral

### 4.1 The eight constraints

1. Row-exact 1 to 8 row decode: unaffected; every method writes ordinary codes and binary16 / E4M3 scales. The one novelty, a negative
   binary16 scale, is a finite FP16 value multiplied in FP32 (2.4); verify with the exactness tests before trusting it.
2. Prefill at 9+ rows: unaffected; codes stay small integers or E2M1 x E4M3.
3. Decode instruction budget: unaffected.
4. Determinism: unaffected at run time. The converter becomes data-dependent; record calibration-set and Hessian hashes in the manifest next
   to `imatrix_sha256`.
5. Offline conversion only: where all the work lands. No new format or layout ordinals; converter time grows to about an hour (GPTQ) or
   several hours (learned rounding); the calibration pass is a separate tool.
6. The artifact decides everything: unchanged.
7. Memory is the product: byte-identical `q3`, `nvfp4`, `nvfp4-compressed`. Exception: P2E2 entropy-codes the Q3 codes, and rounding that
   uses the grid more fully produces more BIG codes; the 81.5 percent ratio was measured on RTN codes and must be re-measured.
8. Drafting precision separate: GPTQ/AWQ apply to the NVFP4 MTP layer and draft head with their own Hessians (the pass must run MTP rows);
   the effect is acceptance rate, one harness pass.

### 4.2 Which linears can take an AWQ scale for free

| Object | Input producer | Fold target | Free? |
|---|---|---|---|
| `attention/query_key`, `attention/gate_value` | `input_norm` (1 + w) | one s for q, k, v, gate | yes |
| `gdn/query_key`, `gdn/value_z`, `gdn/a_projection`, `gdn/b_projection` | `input_norm` | one s for all five (a/b are BF16, absorb 1/s) | yes |
| `mlp/gate_up` | `post_attention_norm` | one s for gate and up | yes |
| `mlp/down` | SwiGLU output | rows of `up` (silu(g) x u/s = (silu(g) x u)/s) | yes; search s on the quantized `up` |
| `attention/output` | attention output x sigmoid(gate) | v rows in `gate_value`; s shared by the 6 query heads of a KV head; passes through the NVFP4 KV cache | yes, with the GQA tie |
| `gdn/output` | gated RMSNorm (128-long weight shared by 48 heads) x silu(z) | only a 128-periodic pattern into `gdn/norm` | no (restricted) |
| `text/output_head` | `final_norm` | `final_norm` also feeds the MTP seed through `mtp/hidden_norm`, not scale-equivariant | no |
| `text/draft_head` | `mtp/final_norm` | the same vector is the next MTP row's hidden | no |
| `mtp/input_projection` | `mtp/embedding_norm`, `mtp/hidden_norm` | the two norms (columns 0..5119, 5120..10239) | yes |
| MTP block projections | its norms, v rows, up rows | as the base layers | yes |
| `text/token_embedding` | gather | none | no (no Hessian either) |

llm-compressor's AWQ mapping for `Qwen3_5ForConditionalGeneration` smooths exactly the input-norm and post-attention-norm groups above
(`dynamic_mappings.py:build_hybrid_attention_mappings`) [16]. The heads and `gdn/output` get error feedback only.

### 4.3 Converter stages that change

`importance.py` gains a second calibration source (per-object H, block diagonal, diagonal) and the AWQ scales; `inventory.py` passes scales to
the norm writers (`add_direct` must rewrite a norm) and row scalings to `gate_up`/`gate_value` before quantization; `grouped.py` gains the GPTQ
loop around `search_group_codes_torch` and signed scales; `nvfp4.py` gains the loop around the block search, a wider `SEARCH_ABOVE`, the 256
normalizer and block-Hessian weights. The parallel per-object writers survive if Hessians are precomputed to disk; fused objects already share
one input and so one H.

### 4.4 Interaction with the existing search and with MTP

The imatrix search is the quantizer *inside* GPTQ, so none of its gain is lost, and its weights can switch from the diagonal to the block
Hessian. Is it "most of AWQ's gain"? Partly: of AWQ's two parts, clipping is here in weighted-MSE form; channel scaling is not, and a shared
group scale cannot imitate it. What scaling adds on top of a weighted scale search is small and mixed at 4 bits (Table F: -0.4 to +1.2 points
on MXFP4, -0.5 to +0.6 on W4A16) and unmeasured at 3 bits in my sources; expect more where the step is coarser. MTP: a base model closer to
BF16 is what the MTP layer was trained against, so acceptance should not fall; the draft stack can take GPTQ; the heads take no AWQ fold
(4.2), which protects the seed hidden.

### 4.5 Expected quality (estimates, repository metrics)

| Form | Today | Signed scales | + GPTQ (own search as quantizer) | + AWQ folds | + learned rounding (50 to 200 it.) |
|---|---|---|---|---|---|
| `q3` | +0.170 nats, KL 0.293, top-1 76.2% | +0.15 to +0.16, KL about 0.27 | +0.13 to +0.15, KL 0.23 to 0.26 | +0.11 to +0.13, KL 0.20 to 0.24 | +0.10 to +0.12, KL 0.18 to 0.22 |
| `q3-compressed` | +0.275 (RTN) | not possible (-4) | calibrated -3..3 + GPTQ: about +0.17 to +0.19 | | |
| `nvfp4` | +0.029, KL 0.147 | n/a; wider E4M3 search: -5 to -10% of weight error | -10 to -20% of weight error | small | -10 to -20% of weight error |
| `nvfp4-compressed` | +0.028, KL 0.155 | as `nvfp4`; the SD4 table absorbs the wider search | as `nvfp4` | small | as `nvfp4` |

Basis: Table G (nats proportional to weighted MSE) for the first column; Tables A, C, D for GPTQ and learned rounding (RTN-to-GPTQ -13 to -29
percent at W3 g128; learned rounding recovering 16 to 64 percent of the scale-search loss on Qwen models; -23 to -33 percent KL on this model at
4 bits); Table F for AWQ. The P2E2 fallback alone (calibrated -3..3, no GPTQ) lands at about +0.20 to +0.22 nats, KL about 0.32 (three quarters
of the calibrated gain kept). For the NVFP4 forms the harness's KL includes the NVFP4 KV cache and kernel drift, so the weight-rounding share is
unknown (section 6); gains of 0.005 nats sit far below the 0.008 standard error.

## 5. Experiment plan

Ordered by cost; each step is go/no-go on the teacher-forced report plus a larger held-out KL run.

1. Fake-quant evaluation path (half a day). Dequantize a converted artifact's codes to BF16 safetensors or GGUF and run `llama-perplexity
   --kl-divergence` over 20 to 50 chunks against the matching BF16 `.kld`. This isolates weight rounding from the KV cache and gives the
   resolution the one-document harness lacks. Run it on today's `q3` and `nvfp4` first to learn the weight-only share.
2. Signed scales for Q3/Q4/Q5 (one day). Remove the clamp in `_binary16_scales`, flip a group's sign when its largest magnitude is positive,
   re-run the exactness and oracle tests on a converted artifact. Go if held-out dNLL improves by at least 0.02 nats (estimate -0.015 to
   -0.02) and the tests pass. Zero bytes, zero kernel work.
3. NVFP4 search width (one day). `SEARCH_ABOVE` 2 to 6 or the full 127-code sweep, global scale `amax / (6 x 256)`; re-run the SD4 DP on the
   wider costs. Go if weight-only KL falls at least 5 percent; expect no visible change in one-document NLL.
4. Calibration pass tool (one to two weeks). PyTorch, layer streaming, hooks; writes per-object diagonal (a Swift-1.5 imatrix including
   `output.weight` and MTP), 16 x 16 block diagonals and full Hessians; 256 x 2048 tokens, half Swift's own thinking-mode generations on varied
   prompts, half web/code/multilingual text. Check that the diagonal reproduces UkisAI's matrix within noise, then feed block-Hessian weights to
   the existing search (go if weight-only KL falls 3 percent).
5. GPTQ in the converter (one to two weeks). Group search as quantizer, `blocksize 128`, 1 percent damping, static groups if act-order is
   tried; all grouped and NVFP4 objects, LM head, draft head, MTP. Go for `q3` on dNLL -0.03 nats and KL -10 percent with top-1 up; go for NVFP4
   on weight-only KL -10 percent with MTP acceptance down by at most 2 percent relative. Measure the P2E2 payload of the GPTQ `q3`.
6. AWQ folds (one week). The free set of 4.2, 20-point alpha grid on the quantized output error, exact norm compensation, before GPTQ. Go on a
   further -0.02 nats / -7 percent KL for `q3`; skip for NVFP4 if step 5 is already within 10 percent of the fake-quant floor.
7. Learned rounding (two to three weeks). First run AutoRound itself on Qwen3.8-27B with `bits=3, group_size=64, sym=True` and `nv_fp4` to get
   the ceiling for free; if it beats step 6 by 0.03 nats, port SignRound (V offsets plus scale multipliers, 50 to 200 iterations, block output
   MSE) into the converter. Same thresholds as step 5, at 1 to 3 GPU hours.
8. P2E2 decision. With steps 2, 5 and 6 measured, convert the -3..3 variant and compare `q3-compressed` candidates by bytes and nats; this
   settles whether `docs/P2E2_FULL_RANGE.md`'s layout work is worth its kernel cost.

## 6. Risks and open questions

- The harness's KL for `nvfp4` (0.147) is 5x AutoRound's 0.0285 for W4 g128 on the same base model [4]. Text and metric details explain
  part of it, but the production path also quantizes the KV cache to NVFP4 and runs relaxed kernels. Until step 1 separates the shares, no
  NVFP4 weight-rounding gain can be sized and the NVFP4 thresholds stay soft.
- One 1279-token document resolves 0.02 (NVFP4) to 0.05 (Q3) nats; most steps promise less. The held-out KL run is not optional.
- Literature transfer: published 3-bit numbers are g128 with zero points on Llama-2; Q3G64 is symmetric with a binary16 scale on a heavily
  trained hybrid model. The ordering of methods should transfer; magnitudes may not.
- Calibration overfitting: GPTQ/AWQ fit 0.26 to 1 M tokens; chat-only or thinking-only sets can hurt general text [14]. Keep a mix and
  evaluate on text the set does not contain.
- GDN layers in PyTorch: the pass needs a correct Qwen3.5 forward with the recurrent scan; a slow reference multiplies stage-1 time by 5 to 10.
- Signed scales: compatibility is argued from reading the kernels, not from running them; P2E2 still cannot carry -4.
- P2E2 size: calibrated and GPTQ codes are less peaked around zero; the 10.16 GiB figure will grow by an unmeasured amount.
- GPTAQ's quantized-input propagation is incompatible with per-object parallel writers; it needs the layer-by-layer pipeline.
- The AWQ fold into v rows changes KV-cache per-channel statistics before the Hadamard rotation; measure KV error separately.
- Swift 1.5 versus base: the `q3` and `nvfp4` numbers are base conversions; a deployed Swift `q3` has no measurement yet. Whether Swift
  re-trained the MTP layer is undocumented and matters for the drafting-stack Hessians.
- Not retrieved: the AWQ paper's tables and AWQ+GPTQ claim, Four Over Six PTQ numbers, GPTAQ tables, "Pretraining LLMs with NVFP4",
  Qwen2.5/Qwen3-32B 3-bit numbers, the ISTA and UkisAI model cards. Treat the GSQ-RCO quality figures as community reports.

## 7. References

1. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", arXiv 2210.17323;
   https://github.com/IST-DASLab/gptq (README, gptq.py, quant.py, llama.py).
2. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", 2023 (arXiv ID not verified here);
   https://github.com/mit-han-lab/llm-awq (awq/quantize/auto_scale.py, auto_clip.py).
3. Shao et al., "OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models", ICLR 2024, arXiv 2308.13137;
   https://github.com/OpenGVLab/OmniQuant.
4. Intel AutoRound (Cheng et al., "Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs", EMNLP 2024 Findings;
   arXiv ID not verified here): https://github.com/intel/auto-round, docs/paper_acc.md, gguf_alg_ext_acc.md, mxnv_acc.md, opt_rtn.md,
   neuqi_acc.md, full_range_sym.md, awq_details.md, step_by_step.md; auto_round/data_type/int.py, nvfp.py.
5. Li et al., "GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration", ICML 2025, arXiv 2504.02692;
   https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ.
6. Cook et al., "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling", arXiv 2512.02010;
   https://github.com/mit-han-lab/fouroversix (src/fouroversix/utils.py).
7. Egiazarian et al., "Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization", arXiv 2509.23202;
   https://github.com/IST-DASLab/FP-Quant, https://github.com/IST-DASLab/qutlass.
8. Zhang et al., "MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization", NeurIPS 2024, arXiv 2406.00800;
   https://github.com/AozhongZhang/MagR.
9. Zhang et al., "LeanQuant: Accurate and Scalable LLM Quantization with Loss-error-aware Grid", ICLR 2025, arXiv 2407.10032;
   https://github.com/LeanModels/LeanQuant.
10. "NeUQI: Near-Optimal Uniform Quantization Parameter Initialization", arXiv 2505.17595.
11. Dadgarnia et al., "GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling", arXiv 2604.18556;
    https://github.com/IST-DASLab/GSQ (README tables, configs/local/config.yaml).
12. Helcig and Alistarh, "Model Compression with Exact Budget Constraints via Riemannian Manifolds", arXiv 2605.00649;
    https://github.com/IST-DASLab/RCO.
13. https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF, https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF,
    https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b (cards not fetched; sizes and scores from search summaries and the Euhedral docs).
14. llama.cpp, https://github.com/ggml-org/llama.cpp: ggml/src/ggml-quants.c (make_qx_quants, quantize_row_q3_K_impl, quantize_row_q4_K_impl,
    make_qkx3_quants), tools/imatrix/README.md, PR 4861, discussion 5263.
15. NVIDIA TensorRT Model Optimizer, https://github.com/NVIDIA/TensorRT-Model-Optimizer: modelopt/torch/quantization/config.py (MSE and
    LocalHessian calibration, LSQConfig, NVFP4 presets), utils/numeric_utils.py (E4M3_MAX_46),
    modelopt_recipes/configs/ptq/presets/model/nvfp4_four_over_six.yaml, nvfp4_awq_lite.yaml.
16. vLLM llm-compressor, https://github.com/vllm-project/llm-compressor: examples/quantization_w4a4_fp4/qwen3_8_gptq_awq_example.py,
    llama3_gptq_example.py, README.md; src/llmcompressor/modifiers/transform/awq/dynamic_mappings.py.
17. ModelCloud GPTQModel, https://github.com/ModelCloud/GPTQModel (README on GPTAQ; gptqmodel/quantization/gptq.py, fallback_smooth.py).
18. Liu et al., "Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models", arXiv 2504.04823.
19. "Quantized Reasoning Models Think They Need to Think Longer, but They Do Not", arXiv 2606.00206.
20. "LAQuant: A Simple Overhead-free Large Reasoning Model Quantization by Layer-wise Lookahead Loss", arXiv 2605.08755.
21. "Think Before You Prune: Selective Self-Generated Calibration for Pruning Large Reasoning Models", arXiv 2511.18864.
22. Unsloth, "Unsloth Dynamic 2.0 GGUFs", https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs.
23. Huang et al., "How Good Are Low-bit Quantized LLaMA3 Models? An Empirical Study", arXiv 2404.14047.
24. "Rethinking Residual Errors in Compensation-based LLM Quantization", arXiv 2604.07955 (GPTAQ follow-up; not read in full).
