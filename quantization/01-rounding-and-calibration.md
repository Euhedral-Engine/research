# Rounding and calibration inside the existing formats

Scope: methods that change *which codes and scales the converter writes* into Q3G64_F16S, Q4G64_F16S, Q5G64_F16S, NVFP4 and
NVFP4-SD4, without touching formats, layouts or kernels. Covered: error feedback (GPTQ/OBQ, GPTAQ, MagR, LeanQuant), learned
rounding (SignRound/AutoRound, BRECQ-style block reconstruction, OmniQuant clipping), activation-aware channel scaling (AWQ,
SmoothQuant), scale search for FP4 (Four Over Six, ModelOpt's E4M3 sweep and local-Hessian weighting), llama.cpp's imatrix
quantizers, ISTA-DASLab's GSQ-RCO, and calibration data. Everything is measured against `00-baseline.md`. Numbers that are my own
estimates are marked "estimate"; everything else carries a source. The web search budget of this session ran out after the first
round of searches, so a few sources named in the brief (the AWQ paper's own tables, the Four Over Six PTQ tables, the GPTAQ tables,
"Pretraining LLMs with NVFP4", 3-bit numbers on Qwen2.5/Qwen3-32B) could not be retrieved and are listed as gaps.

## 1. Summary

The current converter already does one of the three things these methods do: a per-group clipping and stretch search weighted by a
diagonal Hessian proxy (the imatrix), which took the `q3` artifact from +0.275 to +0.170 nats against BF16. It does not do the other
two: cross-column error feedback (GPTQ) and per-input-channel scaling (AWQ), and it does not use the format's full code range the way
llama.cpp does (a signed binary16 scale lets every group put its largest magnitude on code -4). Published 3-bit results on 13B to 70B
dense models put RTN, GPTQ, AWQ, OmniQuant and AutoRound within a narrow band: at W3 g128 on Llama-2-13B the WikiText2 perplexity is
5.51 / 5.42 / 5.32 / 5.28 (RTN / GPTQ / AWQ / OmniQuant) [3], and the 11-task zero-shot average is 58.57 / 59.58 / 58.86 / 59.18 /
59.44 for RTN / GPTQ / AWQ / OmniQuant / AutoRound against 61.42 in FP16 [4]; the calibrated methods recover 30 to 60 percent of the
RTN loss and none of them dominates. The best model-specific number I found is AutoRound's KL measurement on Qwen3.8-27B itself at
W4 g128: its own imatrix-weighted scale search gives KL 0.0285 to BF16 and SignRound tuning lowers that by 23 percent (sym, 57 minutes
on an RTX 3090) to 33 percent (asym, 200 iterations, 3 hours) [4]. Verdict: for `q3` the chain "signed scales, then GPTQ error
feedback with the existing group search as its quantizer, then AWQ scales folded into the norms" should cut the weight-rounding error by
roughly 25 to 40 percent (estimate: +0.170 to about +0.11 to +0.13 nats; KL 0.293 to about 0.20 to 0.24), at a one-time cost of one
calibration pass of 30 to 90 minutes on the 16 GB card plus a converter pass of about an hour; learned rounding (AutoRound-style) can add
another 10 to 20 percent for 1 to 3 GPU hours. For `nvfp4` and `nvfp4-compressed` the same chain applies to the E2M1 grid, but the
published gains at 4.5 bpw are small (AutoRound on NVFP4 Qwen3-32B: 0.7164 to 0.7306 average accuracy [4]) and the harness cannot
resolve them on one document; the one free improvement is widening the E4M3 scale search above the round-to-nearest code, which my
simulation puts at a further 10 percent of weight MSE. `q3-compressed` cannot take code -4 and therefore keeps at best 75 percent of the
calibrated gain (estimate); it is the form that profits least from this family.

## 2. Mechanism

### 2.1 What the current search is, in these terms

For a linear with weights W (N x K) and inputs X (T tokens x K), every method here minimizes the layer output error
`||(W - Q(W)) X^T||_F^2 = sum_rows e H e^T` with `H = X^T X` (K x K), e = row error. The imatrix is `diag(H)/T` (llama.cpp
`in_sum2 / counts`), so `search_group_codes_torch` minimizes `sum_j H_jj e_j^2`: the diagonal approximation, one scalar (the group
scale) per 64 columns, 39 candidates. Three things are left on the table: the off-diagonal of H (error feedback), the freedom to give
individual columns a finer step (channel scaling), and one code of the grid (section 2.4).

### 2.2 Error feedback: GPTQ, GPTAQ, MagR, LeanQuant

GPTQ [1] (reference loop in `gptq/gptq.py:fasterquant`): accumulate `H = 2/T sum x x^T` in FP32; add damping `0.01 x mean(diag H)`;
take the upper Cholesky factor of H^-1; walk columns in blocks of 128: `q = Q(w_i)`, `err = (w_i - q) / Hinv[i,i]`,
`W[:, i:] -= err (x) Hinv[i, i:]`, then propagate the block's errors to the remaining columns. With groups, the group's scale is set when
the loop reaches the group's first column, from the *already compensated* weights (`static_groups=False`), or from the original weights
beforehand (`static_groups=True`, required when `act_order` reorders columns so that groups stay contiguous [1, README]). Rows are
independent given Hinv, so the LM head (248,320 rows) is processed in row chunks. Defaults: 128 sequences x 2048 tokens of C4,
`percdamp 0.01`, `blocksize 128` [1]. For Euhedral, `Q(.)` is the existing 39-candidate weighted search applied to the 64 compensated
columns of the group, so the format, the group geometry and the binary16 scale are untouched; for NVFP4 the block scale (E4M3 code search,
section 2.6) is chosen when the loop reaches each 16-column block.

GPTAQ ("GPTQ v2", arXiv 2504.02692 [5]) matches the quantized layer's output to the *full-precision model's* output instead of its own
input: it accumulates `dXXT = (X_fp - X_q) X_q^T` alongside H and adds `P = alpha x triu((dXXT Hinv^T), 1) Hinv`, alpha = 0.25, to the
update (`GPTAQ/fake_quant/gptaq_utils.py`). It needs both the BF16 and the partially quantized model's activations per layer, i.e. a
sequential pass. GPTQModel ships it as experimental, "requires 2-4x more VRAM", "results may not be better than original" [17]. MagR
(NeurIPS 2024, arXiv 2406.00800 [8]) is a pre-processing step: proximal gradient on W to reduce `||W||_inf` while keeping `WX` (alpha
0.0001 per-group, 200 iterations) before GPTQ; in-format, but its 3-bit g128 results were partly omitted by a later reproduction as
anomalous [10]. LeanQuant (ICLR 2025, arXiv 2407.10032 [9]) changes the grid (non-uniform, or affine with a zero point): not
representable in Q3G64_F16S, out of scope.

### 2.3 Activation-aware channel scaling: AWQ, SmoothQuant

AWQ [2] multiplies input channel j of W by `s_j` and divides the activation by `s_j`, so that salient channels get a finer effective step
inside a shared group scale. `auto_scale.py`: `s = x_max^ratio` for 20 values of `ratio` in [0, 0.95], `x_max` the mean |x_j| over
calibration tokens, the ratio chosen by the output MSE of the *quantized* scaled weights on the calibration batch; `auto_clip.py`: per group
max shrink over 20 steps down to 0.5 on 512 sampled tokens. The scale is folded into whatever produces the input: a norm for q/k/v and
gate/up (`scale_ln_fcs`), the rows of `v_proj` for `o_proj`, the rows of `up_proj` for `down_proj` (`scale_fc_fc`). AWQ is a
pre-processing step and composes with any quantizer; AutoRound runs it as `alg_configs=['awq','auto_round']` [4], llm-compressor's Qwen3.8-27B
NVFP4 recipe runs `AWQModifier` then `GPTQModifier` [16]. SmoothQuant moves outliers the other way (into the weights) for activation
quantization; for weight-only formats it is counterproductive and only matters to the native NVFP4 prefill's two-term activation
quantization, where AWQ's scaling already pushes activations in the smoothing direction.

Folding with exact compensation in a BF16 `(1 + w)` norm: store `w' = bf16((1 + w) s - 1)`, read back `s_real = (1 + w') / (1 + w)`,
and divide the weight columns by `s_real` (not by s) before quantizing. The BF16 rounding of the norm then costs nothing; the only
approximation left is the quantization itself.

### 2.4 The eighth code: signed scales (llama.cpp) and the P2E2 fallback

llama.cpp's `make_qx_quants` (ggml-quants.c:628) sets `iscale = -nmax / max` where `max` is the *signed* value of largest magnitude: the
largest magnitude always lands on code -4, and when it is positive the returned scale is negative. The binary16 scale plane of Q3G64_F16S
carries a sign; the Q3 decode kernel converts the scale with `__half2float` and multiplies with `fmaf` (native/src/q3/contiguous.cuh:76,94),
the reference kernel does the same (native/src/reference/kernels.cu:38), P2E2 copies the scale plane byte for byte (q3_p2e2.py:120) and the
MXFP8 prefill route applies the binary16 scale in FP32 after each group (00-baseline section 5). Nothing I read takes an absolute value, so
negative scales look format- and kernel-compatible; the converter is what forbids them (`_binary16_scales` clamps to [0, 65504]). AutoRound's
symmetric quantizer uses the same trick and credits "the llamacpp community" (`auto_round/data_type/int.py:quant_tensor_sym`); its docs show
"full-range sym" beating GPTQ's symmetric convention by 8.8 points at W2G32 on Llama-3.1-8B-Instruct and matching asymmetric within 0.3
points at W4G128 [4]. GPTQ's own `sym=True` convention (`zero = 4`, step `2 amax / 7`) is a third grid that clips positive values above 0.86
amax [1, quant.py].

The P2E2 fallback (codes -3..3) loses that code entirely. Reasoning from the grid: 7 levels cover the group with 6 intervals instead of 7, so
the step grows by 7/6 and the rounding MSE by about (7/6)^2 = 1.36 for the groups that used -4. I measured the converter's actual search
on synthetic groups (section 3, table G): restricting to -3..3 forfeits about one quarter of the calibrated MSE reduction, signed scales add
9 to 13 percent on top of today's search. Published symmetric-versus-asymmetric numbers at 3 bits: none found; at W4 g128 on Qwen3.8-27B
AutoRound's symmetric search is 6 percent worse in KL than its asymmetric one (0.02734 vs 0.02579), 4 percent after tuning [4].

### 2.5 Learned rounding and clipping: SignRound/AutoRound, BRECQ, OmniQuant LWC

SignRound/AutoRound [4] keeps the format and learns `W_q = s x clamp(round(W/s + V), qmin, qmax)` with a per-weight rounding offset
V in [-0.5, 0.5] and per-group scale multipliers, by signed-gradient descent on the block output MSE: defaults 128 samples x 2048 tokens
(NeelNanda/pile-10k), 200 iterations, lr 1/iters, `low_gpu_mem_usage` offloads activations (+30 to 100 percent time). It is the modern form
of BRECQ/AdaRound block reconstruction. `iters=0` is an "optimized RTN" scale search weighted by an automatically collected activation imatrix,
the analogue of Euhedral's search; `--enable_neuqi` adds a joint (scale, zero) or two-stage signed-scale grid search (arXiv 2505.17595 [10]);
`enable_alg_ext` (SignRoundV2) is the stronger tuner. AutoRound exports GGUF K-quants in-format and has NVFP4 (`nv_fp4`, global scale
`448 x 6 / amax`, group 16) and MXFP4 data types. Costs on an A100 for W4G128 with torch.compile: 14B 23 min / 22 GB, 70B 120 min / 42 GB;
with `low_gpu_mem_usage`, `gradient_accumulate_steps=8, batch_size=1`: 14B 45 min / 7 GB, 70B 187 min / 19 GB [4]. On Qwen3.8-27B W4 g128
on an RTX 3090 (24 GB): iters=0 9 to 13 min, 50 iterations 49 to 57 min, 200 iterations 2 h 57 min [4]. OmniQuant's LWC [3] learns a
sigmoid clipping factor per group with Adam (lr 1e-2, 20 epochs, 128 samples); it produces an ordinary scale and is in-format; its LET
(equivalent transformations) targets W4A4 and is irrelevant here. GSQ (section 2.8) is the heaviest member of this family.

### 2.6 Scale search for FP4: E4M3 sweep, local Hessian, Four Over Six

ModelOpt [15] calibrates NVFP4 block scales by (a) `mse` with amax multipliers 0.25 to 4.0 in steps of 0.1, or (b) `fp8_scale_sweep`: try
all 128 E4M3 codes per block ("the recommended setting for NVFP4"), optionally weighted by (c) `local_hessian`: the block's 16 x 16
`H = X X^T`, loss `dw H dw^T` (config.py:LocalHessianCalibConfig). Euhedral's search is (b) restricted to -6..+2 codes around RTN with
diagonal weights; (c) is a strict refinement that needs only the 16 x 16 block diagonal of H per linear (K/16 x 256 floats). Four Over Six
(arXiv 2512.02010 [6]) tries two scales per block, mapping the block maximum to 6 or to 4, and keeps the lower MSE (`ScaleRule.mse` in
`fouroversix/utils.py`); because the max-to-4 choice needs a 1.5x larger block scale, both ModelOpt and the reference code lower the
per-tensor normalizer from 448 to 256 (`E4M3_MAX_46 = 256`, `numeric_utils.py:41`; `get_maximum_value` returns 256 for non-static rules).
In Euhedral terms: set the FP32 global scale to `amax / (6 x 256)` and let the block search go up to 5 to 6 codes above RTN; the decode
kernels apply the global scale once per output (`nvfp4.cuh:172`), so nothing else changes. ModelOpt also learns block scales by LSQ /
Dual-LSQ (`LSQConfig`) and has `NVFP4_AWQ_LITE/CLIP/FULL` presets (alpha grid 0 to 1 step 0.1). The MR-GPTQ paper (arXiv 2509.23202 [7])
reports that for NVFP4 plain GPTQ gives the highest recovery among its post-training methods and that large models recover 98 to 99 percent
of FP16 accuracy, while MXFP4 needs its micro-rotations; its abstract-level numbers are all I could retrieve.

### 2.7 llama.cpp's imatrix quantizers

`quantize_row_q3_K_impl` weights each value by `imatrix_j x sqrt(sigma^2 + x^2)` (sigma^2 = 2 x mean square of the 256-block), then
`make_qx_quants(16, 4, ...)` per 16-value sub-block: signed scale (section 2.4), RTN candidate plus 18 candidates `-(4 + 0.1 is)/max`, each
refit by weighted least squares, kept if the weighted `sumlx^2/suml2` improves; the 16 sub-scales are themselves quantized to 6 bits by the same
routine. Q4_K uses `make_qkx3_quants` (asymmetric with a min, `rmin -0.9, rdelta 0.05, 36 steps`). IQ2/IQ3 use 8-value codebooks (not
representable here). `llama-imatrix`: chunk = context length (`n_ctx / n_parallel`), `--chunks`, `--process-output` (also collects
`output.weight`, off by default because the tool's authors found it "typically better not to" use it for embeddings and output),
`--nextn` (collects MTP layer statistics; needs one sequence per batch and an MTP with its own KV cache, which Qwen3.8 has per
docs/MTP_CONTRACT.md), `--parse-special`, `--in-file` to merge runs [14]. The current converter's search is this algorithm with a positive
scale, a 64-group and no sub-scales.

### 2.8 GSQ-RCO (ISTA-DASLab, 2026)

GSQ = Gumbel-Softmax Quantization (arXiv 2604.18556, github.com/IST-DASLab/GSQ [11]): post-training scalar quantization that keeps a
symmetric group-wise integer grid (INT kernels, GGUF K-quants in-format) and learns the discrete assignment of every weight. Pipeline: GPTQ
initialization per layer; per-weight logits over the grid (2-bit `{-2,-1,0,1} x s`, ternary, or an n-bit grid around the init), sampled with
Gumbel-Softmax, temperature annealed 2 to 0.05 and logit scale 100 to 500, Lion optimizer with cosine LR, layer-wise reconstruction loss
`||f(x; w~) - f(x; w)||^2`, bf16 logits; layers offloaded to `meta` after completion so the model need not fit the GPU. Default config:
`open_thoughts` reasoning data, 4096 samples x 4096 tokens, 10 epochs, group 128, GPTQ init with `percdamp 0.1`, 512 samples. Reported:
Llama-3.1-70B-Instruct at 2.13 bpp, zero-shot average 75.57 vs GPTQ 57.38, QTIP 77.25, FP16 78.99; Qwen3-8B Q2_K GGUF refined in-format
50.03 to 56.28; Kimi-K2.5 to 2.13 bpp; dense Llama-3.1-8B on 1 x H100, 70B on 4 x H100 [11]. RCO = Riemannian Constrained Optimization
(arXiv 2605.00649, github.com/IST-DASLab/RCO [12]): assigns one of K options (bit widths or GGUF types) to each of N groups (layers or
tensors) under an exact size budget by Adam on the budget manifold (tangent projection, retraction, vector transport) with a Gumbel-STE
forward pass solved by a multiple-choice knapsack DP; reported to match or beat EvoPress at 3 to 16x lower wall-clock. The Qwen3.8-27B release
(IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, 8.4 to 11.8 GB, with the 1000 x 4096-token imatrix) uses GSQ for the per-tensor low-bit quantization and RCO
to pick each tensor's type under a size budget [13]; community threads report IQ3_XXS at about 10 GB keeping about 99 percent of the full
model's scores, which I could not verify (Hugging Face is unreachable from this session). UkisAI's Swift-1.5 GSQ-RCO GGUF reuses ISTA's
per-tensor allocation and ships `imatrix-swift15-v1mix.gguf`, 309 chunks x 4096 tokens (about 1.27 M tokens) of web, reasoning, multilingual
and code text, the matrix behind `nvfp4-compressed` [13]. Reproducibility in the converter: GSQ's optimization is in-format for Q3/Q4/Q5
(a symmetric grid with a per-group scale) and is a heavier cousin of AutoRound; at its default data volume it is out of reach on one 16 GB card
(estimate: 6 x 0.4 G params x 1.7e8 token-passes x 10 epochs x 64 layers = 2.6e19 FLOP, about 150 hours at 50 TFLOPS); with 128 x 2048
tokens it is a few hours, but the README's gains are reported at 2 bits, where the GPTQ-to-VQ gap is large; at 3 bits the gap is small
(section 3). RCO is a mixed-precision allocator and belongs to the bit-allocation chapter, but it would run on the converter's existing Q3/Q4/Q5
ladder unchanged.

### 2.9 Calibration data

Sizes in use: GPTQ 128 x 2048 C4 [1]; AWQ 128 x 512 Pile [4, paper_acc]; AutoRound 128 x 2048 pile-10k, 512 for its paper [4]; OmniQuant
128 x 2048 WikiText2 [3]; FP-Quant 128 x 2048 fineweb-edu [7]; llm-compressor's Qwen3.8-27B recipe 512 x 4096 "perfectblend" [16];
GSQ 4096 x 4096 open_thoughts [11]; RCO offers chat sets with answer-only loss masks (evol_codealpaca, tulu_math) [12]; Unsloth's imatrix
sets 300 K to 1.5 M tokens of hand-curated chat-oriented text [22]; ISTA's Qwen3.8-27B imatrix 1000 x 4096 = 4.1 M tokens; UkisAI's 1.27 M
tokens. The llama.cpp imatrix discussion found text to beat pseudo-random data and language-specific text to help when one language dominates
[14, discussion 5263]. For reasoning models: Liu et al. (arXiv 2504.04823 [18]) find W4 weight-only near-lossless and W3 costly (over 7 percent on
LiveCodeBench for AWQ and GPTQ); arXiv 2606.00206 [19] calibrates GPTQ on *model-generated* reasoning traces (NuminaMath-1.5) to reduce domain
shift and finds 3-bit AWQ inflating overthinking errors from 19 to 139; LAQuant (arXiv 2605.08755 [20]) reports large Pass@1 gains from swapping
generic pretraining text for DeepSeek-R1 reasoning traces under the same GPTQ; for pruning, self-generated calibration data helps reasoning
models (arXiv 2511.18864 [21]). Since the deployed artifact is converted from Swift 1.5 with UkisAI's mixed matrix, there is no base/fine-tune
mismatch today; the open question is whether a matrix containing Swift's own thinking-mode generations would do better. The evidence above says
yes for reasoning-heavy use and "keep a mix" for general text.

### 2.10 Sequential Hessian collection on one 16 GB GPU

The BF16 checkpoint (54.7 GB) never needs to be resident. Stage 1, one pass: stream layer l's weights from the safetensors mmap to the GPU
(about 0.85 GB per layer), run the T calibration tokens through it in micro-batches of a few sequences (the GDN recurrence needs whole
sequences; use the Hugging Face Qwen3.5 implementation with its chunked scan kernels), hook each linear's input to accumulate `H += X^T X`
in FP32 (gate_up and the attention/GDN input projections share K = 5120: 105 MB; down K = 17408: 1.2 GB; o_proj/out_proj K = 6144: 151 MB),
also the diagonal (an imatrix for Swift 1.5 for free) and the 16 x 16 block diagonal; replace the layer input buffer with its output (T x 5120
BF16: 2.7 GB at T = 262 k, kept on the GPU or in pinned host memory) and move on. Also collect the final-norm output (LM head Hessian) and the
MTP inputs `[e; n]` and the MTP block's own inputs for the drafting stack. Compute: forward 2 x 25.6 G x T plus Hessian accumulation
2 x sum K^2 x T per layer, both about 1.3e16 FLOP at T = 262 k; at 40 to 60 TFLOPS achieved that is 8 to 12 minutes, plus the GDN scan and
Python overhead: budget 30 to 60 minutes at T = 262 k, 2 to 4 hours at T = 1 M (estimate). Peak device memory: weights 0.85 GB + H 1.5 GB +
activations 2.7 GB + micro-batch temporaries (8 k tokens x 34816 x 4 B = 1.1 GB) < 7 GB. Persisting every H in FP32 for repeated converter
runs costs about 94 GB of disk (estimate: 64 x 1.47 GB); the alternative is quantizing layer by layer inside the pass (standard GPTQ, also
enables quantized-input propagation and GPTAQ). Stage 2 in the converter: per object `O(N K^2 + K^3/3)`, 0.1 to 1.5e12 FLOP, in practice
2 to 10 s per object because the column loop is sequential (5120 to 17408 steps); the LM head in row chunks of 16 k; total 20 to 60 minutes
(estimate). AutoRound-style tuning instead: 1 to 3 hours for 50 to 200 iterations, extrapolating the 3090 measurement [4] and the cost table's
19 GB for 70B with the memory-saving settings (27B: about 10 GB, estimate).

## 3. Published results

Table A. W3 g128 WikiText2 perplexity, Llama-2, as tabulated in the OmniQuant paper (ICLR 2024) [3]. FP16 references 5.47 / 4.88 / 3.32 are
the usual values, not re-verified in this session; nats = ln(PPL / FP16).

| Model | RTN | GPTQ | AWQ | OmniQuant | RTN to GPTQ | RTN to OmniQuant |
|---|---|---|---|---|---|---|
| Llama-2-7B | 6.66 (+0.197 nats) | 6.29 (+0.140) | 6.24 (+0.132) | 6.03 (+0.097) | -29% | -51% |
| Llama-2-13B | 5.51 (+0.121) | 5.42 (+0.105) | 5.32 (+0.086) | 5.28 (+0.079) | -13% | -35% |
| Llama-2-70B | 3.97 (+0.179) | 3.85 (+0.148) | not reported | 3.78 (+0.130) | -17% | -27% |

Without grouping (W3A16 per channel) the spread is far larger (13B: RTN 10.68, GPTQ 6.44, AWQ 10.45, OmniQuant 5.58) [3]: per-group
scales, which Euhedral has, are most of the battle. A later paper re-ran AWQ at 6.19 / 5.30 / 3.72 for the three sizes [3, search note], so
the AWQ column is uncertain by about 0.1.

Table B. LLaMA-1 WikiText2 perplexity from the GPTQ repository README [1].

| Model | FP16 | 3-bit RTN | 3-bit GPTQ | 3-bit g128 GPTQ |
|---|---|---|---|---|
| 7B | 5.68 | 25.54 | 8.07 | 6.61 |
| 13B | 5.09 | 11.40 | 6.63 | 5.62 |
| 30B | 4.10 | 14.89 | 5.69 | 4.80 |
| 65B | 3.53 | 10.59 | 5.04 | 4.17 |

Table C. Average of 11 zero-shot tasks, Llama-2, W3G128 (AutoRound paper tables; 512 pile-10k samples, GPTQ with act-order and true-sequential,
AWQ at sequence length 512) [4, docs/paper_acc.md].

| Model | FP16 | RTN | GPTQ | AWQ | HQQ | OmniQuant | AutoRound |
|---|---|---|---|---|---|---|---|
| Llama-2-7B | 57.98 | 53.81 | 54.14 | 55.21 | 54.31 | 54.72 | 56.68 |
| Llama-2-13B | 61.42 | 58.57 | 59.58 | 58.86 | 58.10 | 59.18 | 59.44 |
| Llama-2-70B | 66.12 | 64.08 | 65.08 | 65.12 | 64.80 | 65.12 | 65.31 |
| 13B at W4G128 | 61.42 | 60.65 | 61.00 | 60.91 | 60.65 | 60.51 | 60.85 |
| 70B at W4G128 | 66.12 | 65.87 | 66.22 | 66.23 | 66.06 | 66.02 | 66.39 |

Table D. 3-bit GGUF in-format (Q3_K_S, about 3.44 bpw): AutoRound's imatrix-weighted scale search ("Optimized RTN") versus learned rounding
on top of it, average of 10 tasks, percent of BF16 in parentheses [4, docs/gguf_alg_ext_acc.md].

| Model | BF16 | Optimized RTN | AutoRound + alg_ext | Share of the loss recovered | Time (Opt RTN / AutoRound) |
|---|---|---|---|---|---|
| Qwen3-8B | 0.6322 | 0.6128 (96.9%) | 0.6252 (98.9%) | 64% | 1 m 29 s / 48 min |
| Qwen2.5-7B-Instruct | 0.6571 | 0.6382 (97.1%) | 0.6503 (99.0%) | 64% | 1 m 20 s / 36 min |
| Llama-3.1-8B | 0.6295 | 0.6040 (96.0%) | 0.6081 (96.6%) | 16% | 1 m 25 s / 30 min |
| Qwen3-30B-A3B-Instruct | 0.6746 | 0.6598 (97.8%) | 0.6622 (98.2%) | 16% | 25 min / 12 h 48 min |

Table E. FP4 and Qwen-family 4-bit results.

| Model, format | Metric | RTN or search | Improved | Method | Source |
|---|---|---|---|---|---|
| Qwen3.8-27B, W4A16 g128 sym | mean KL to BF16, openwebtext 65 k positions | 0.02852 (imatrix search) / 0.02734 (NeUQI) | 0.02191 (50 it.) / 0.02203 (no NeUQI) | SignRoundV2 | [4, neuqi_acc] |
| Qwen3.8-27B, W4A16 g128 asym | same | 0.02579 (NeUQI) | 0.02106 (50 it.) / 0.01921 (200 it.) | SignRoundV2 | [4] |
| Qwen3-32B, NVFP4 g16 | avg of 5 tasks (fake quant) | 0.7164 | 0.7306 / 0.7295 (alg_ext) | AutoRound | [4, mxnv_acc] |
| Qwen3-32B, MXFP4 g32 | same | 0.6901 | 0.7211 | AutoRound | [4] |
| Qwen3-8B, NVFP4 | avg of 11 tasks; BF16 65.45 | 64.01 (RTN) | 64.40 | optimized RTN (iters 0) | [4, opt_rtn] |
| Llama-3.1-8B-I, NVFP4 | same; BF16 65.17 | 62.65 | 63.91 | optimized RTN | [4] |
| Llama-3-70B, W4 g128 / W3 g128 | WikiText2 PPL; FP16 2.9 | | 3.3 / 5.2 | GPTQ | [23] |
| Llama-3.1-70B-Instruct, 2.13 bpp | zero-shot avg; FP16 78.99 | GPTQ 57.38 | 75.57 | GSQ | [11] |

Table F. Does AWQ stack on a scale search or on learned rounding? AutoRound's composition study, average of 5 tasks [4, docs/awq_details.md].

| Scheme, model | opt_rtn | AWQ + RTN | AutoRound (alg_ext) | AWQ + AutoRound (alg_ext) |
|---|---|---|---|---|
| MXFP4, Llama-3.1-8B-I | 62.75 | 62.36 | 65.83 | 66.20 |
| MXFP4, Qwen3-8B | 68.03 | 69.23 | 69.75 | 70.18 |
| W4A16, Llama-3.1-8B-I | 69.04 | 69.66 | 69.70 | 69.95 |
| W4A16, Qwen3-8B | 71.91 | 71.45 | 72.05 | 72.10 |

Table G. My estimate: relative weighted MSE of the converter's rounding variants on synthetic groups of 64 (60,000 groups; Gaussian,
Student-t with 4 degrees of freedom, Laplace; uniform or log-normal (sigma 1.5) per-column weights), with `search_group_codes_torch`
re-implemented in numpy. Not model data; the ordering, not the digits, is the result.

| Variant | Gaussian, uniform | Student-t, uniform | Gaussian, weighted | Groups using -4 |
|---|---|---|---|---|
| RTN `amax/3` | 1.000 | 1.000 | 1.000 | 0% |
| current search (ends 3 and 4, positive scale) | 0.570 | 0.588 | 0.463 | 65 to 72% |
| current search without the -4 end | 0.616 | 0.630 | 0.550 | |
| P2E2 fallback: search restricted to -3..3 | 0.683 | 0.713 | 0.595 | 0% |
| signed scale (llama.cpp trick) | 0.497 | 0.494 | 0.421 | 95 to 100% |
| NVFP4 block search -6..+2 (current) | 0.815 | 0.865 | | |
| NVFP4 block search -6..+6 (covers the 4/6 choice) | 0.730 | 0.812 | | 29 to 40% of blocks go above +2 |

The repository's own RTN-to-calibrated step (+0.275 to +0.170 nats, a 38 percent drop) matches the simulated MSE drop (43 percent) well
enough to use "nats proportional to weighted MSE" as a first-order rule below.

Gaps: the AWQ paper's Llama-2 70B INT3 g128 value and its AWQ+GPTQ table, Four Over Six's PTQ perplexities, GPTAQ's tables, MagR and
LeanQuant tables, "Pretraining LLMs with NVFP4", and any 3-bit measurement on Qwen2.5/Qwen3-32B could not be retrieved in this session.

## 4. Fit with Euhedral

### 4.1 The eight constraints

1. Row-exact 1 to 8 row decode: unaffected. Every method writes ordinary codes and binary16 / E4M3 scales; the only novelty, a negative
   binary16 scale, is a finite FP16 value the kernels multiply in FP32 (section 2.4). Verify with the exactness tests before trusting it.
2. Prefill route at 9+ rows: unaffected; codes stay small integers (exact E4M3 expansion) or E2M1 x E4M3.
3. Decode instruction budget: unaffected; no new decoding work.
4. Determinism: unaffected at run time. The converter becomes data-dependent (calibration set, Hessians); record their hashes in the
   manifest next to `imatrix_sha256`.
5. Offline conversion only: this is exactly where all the work lands. No new format or layout ordinals. Converter time grows from minutes
   to about an hour (GPTQ) or several hours (learned rounding); the calibration pass is a separate tool that writes Hessians or an imatrix.
6. The artifact decides everything: unchanged; `ArtifactProfile` sees the same formats.
7. Memory is the product: byte-identical artifacts for `q3`, `nvfp4`, `nvfp4-compressed`. Exception: P2E2 entropy-codes the Q3 codes, and
   any rounding that uses the grid more fully (calibrated search, GPTQ) produces more BIG codes and a larger payload; the 81.5 percent ratio
   was measured on RTN codes and must be re-measured.
8. Drafting precision is a separate knob: GPTQ/AWQ apply to the NVFP4 MTP layer and the draft head with their own Hessians (the calibration
   pass must run MTP rows); the effect shows up only as acceptance rate, which the harness measures in one pass.

### 4.2 Which linears can take an AWQ scale for free

| Object | Input producer | Fold target | Free? |
|---|---|---|---|
| `attention/query_key`, `attention/gate_value` | `input_norm` (1 + w) | one s for q, k, v, gate (fused objects share the input) | yes |
| `gdn/query_key`, `gdn/value_z`, `gdn/a_projection`, `gdn/b_projection` | `input_norm` | one s for all five (a/b are BF16 and absorb 1/s) | yes |
| `mlp/gate_up` | `post_attention_norm` | one s for gate and up | yes |
| `mlp/down` | SwiGLU output | rows of `up` (silu(g) x u/s = (silu(g) x u)/s) | yes; `up` rows are themselves quantized, so search s on the quantized `up` |
| `attention/output` | attention output x sigmoid(gate) | rows of v inside `gate_value`; s shared by the 6 query heads of each KV head (GQA 6:1); passes through the NVFP4 KV cache | yes, with the GQA tie |
| `gdn/output` | gated RMSNorm (weight of length 128 shared by 48 heads) x silu(z) | only a 128-periodic pattern into `gdn/norm`; z is inside silu | no (restricted) |
| `text/output_head` | `final_norm` | `final_norm` also feeds the MTP seed hidden through `mtp/hidden_norm`, which is not scale-equivariant | no |
| `text/draft_head` | `mtp/final_norm` | the same vector is the next MTP row's hidden | no |
| `mtp/input_projection` | `mtp/embedding_norm`, `mtp/hidden_norm` | the two norms (columns 0..5119 and 5120..10239) | yes |
| MTP block projections | its norms, its v rows, its up rows | as the base layers | yes |
| `text/token_embedding` | gather | none | no (and no Hessian either) |

llm-compressor's AWQ mapping for `Qwen3_5ForConditionalGeneration` smooths exactly the input-norm and post-attention-norm groups above
(`dynamic_mappings.py:build_hybrid_attention_mappings`) [16], consistent with this table. The heads and `gdn/output` get error feedback only.

### 4.3 What changes in the converter

`importance.py` grows a second source of calibration (per-object Hessian, block diagonal, diagonal) and the AWQ scales; `inventory.py`
passes the scales to the norm writers (`add_direct` must be able to rewrite a norm) and the row scalings to `gate_up`/`gate_value` before
quantization; `grouped.py` gains the GPTQ loop around `search_group_codes_torch` and signed scales; `nvfp4.py` gains the same loop around
the block scale search, a wider `SEARCH_ABOVE`, the 256 normalizer and the block-Hessian weighting. The parallel per-object writers stay as
they are if Hessians are precomputed (stage 1 writes them to disk); the fused objects already share one input and hence one H.

### 4.4 Interaction with the existing search and with MTP

The imatrix search is the quantizer *inside* GPTQ: at each group start it picks the scale from the compensated columns, so nothing of the
current gain is lost and the search's weights can switch from the diagonal to the block Hessian. Is the search "most of AWQ's gain"? Partly:
of AWQ's two parts, clipping is already here in a weighted-MSE form; channel scaling is not, and it is the part a shared group scale cannot
imitate. Published evidence of what channel scaling adds on top of a weighted scale search is mixed and small at 4 bits (Table F: -0.4 to
+1.2 points on MXFP4, -0.5 to +0.6 on W4A16) and unmeasured at 3 bits in my sources; expect more at 3 bits, where the step is coarser. For
MTP: a base model closer to BF16 is what the MTP layer was trained against, so acceptance should not fall; the draft stack can take GPTQ with
its own Hessians; neither AWQ fold on the heads is allowed (section 4.2), which protects the seed hidden.

### 4.5 Expected quality (estimates, repository metrics)

| Form | Today | Signed scales | + GPTQ (own search as quantizer) | + AWQ folds | + learned rounding (50 to 200 it.) |
|---|---|---|---|---|---|
| `q3` | +0.170 nats, KL 0.293, top-1 76.2% | +0.15 to +0.16, KL about 0.27 | +0.13 to +0.15, KL 0.23 to 0.26 | +0.11 to +0.13, KL 0.20 to 0.24 | +0.10 to +0.12, KL 0.18 to 0.22 |
| `q3-compressed` | +0.275 (RTN) | not possible (-4) | calibrated -3..3 + GPTQ: about +0.17 to +0.19 | | |
| `nvfp4` | +0.029, KL 0.147 | n/a; wider E4M3 search: -5 to -10% of weight error | -10 to -20% of weight error | small | -10 to -20% of weight error |
| `nvfp4-compressed` | +0.028, KL 0.155 | as `nvfp4`; the SD4 table absorbs the wider search | as `nvfp4` | small | as `nvfp4` |

Basis: Table G for the first column (nats proportional to weighted MSE); Tables A, C, D for GPTQ and learned rounding (RTN-to-GPTQ -13 to
-29 percent at W3 g128; learned rounding recovering 16 to 64 percent of the scale-search loss on Qwen models; -23 to -33 percent KL on this
very model at 4 bits); Table F for AWQ. The P2E2 fallback alone (calibrated -3..3, no GPTQ) lands at about +0.20 to +0.22 nats, KL about
0.32 (estimate: three quarters of the calibrated gain kept). For the NVFP4 forms the harness's KL includes the NVFP4 KV cache and kernel
drift, so the weight-rounding share is unknown (section 6); gains of 0.005 nats are far below the 0.008 standard error.

## 5. Experiment plan

Each step is go/no-go on the repository's teacher-forced report plus a larger held-out KL run; order is by cost.

1. Build a fake-quant evaluation path (half a day). Dequantize a converted artifact's codes back to BF16 safetensors or GGUF and run
   `llama-perplexity --kl-divergence` over 20 to 50 chunks against the matching BF16 `.kld`. This isolates weight rounding from the KV cache
   and gives the resolution the single-document harness lacks. Also run it on the current `q3` and `nvfp4` to learn the weight-only share.
2. Signed scales for Q3/Q4/Q5 (one day). Remove the clamp in `_binary16_scales`, flip the sign of the group when its largest magnitude is
   positive, re-run the exactness and oracle tests on a converted artifact. Go if dNLL improves by at least 0.02 nats on the held-out KL run
   (estimate -0.015 to -0.02) and the exactness tests pass. Zero bytes, zero kernel work.
3. NVFP4 scale search width (one day). `SEARCH_ABOVE` 2 to 6 (or the full 127-code sweep) and the global scale `amax / (6 x 256)`; for SD4
   re-run the DP with the wider costs. Go if weight-only KL falls by at least 5 percent; expect no visible change in the one-document NLL.
4. Calibration pass tool (one to two weeks). PyTorch, layer streaming, hooks; outputs per-object diagonal (a Swift-1.5 imatrix, also
   `output.weight` and MTP), 16 x 16 block diagonals and full Hessians to disk; 256 x 2048 tokens of a 50/50 mix of Swift's own thinking-mode
   generations on varied prompts and web/code/multilingual text. First check: the diagonal reproduces UkisAI's matrix within noise, then
   block-Hessian weights in the existing search (go if weight-only KL falls by 3 percent on `nvfp4` or `q3`).
5. GPTQ in the converter (one to two weeks). Group search as quantizer, `blocksize 128`, damping 1 percent, static groups if act-order is
   tried; all grouped objects, NVFP4 objects, LM head, draft head, MTP. Go for `q3` if dNLL improves by at least 0.03 nats and KL by 10
   percent with top-1 up; go for NVFP4 if weight-only KL falls 10 percent and MTP acceptance does not drop by more than 2 percent relative.
   Measure the P2E2 payload of the GPTQ `q3`.
6. AWQ folds (one week). The free set of section 4.2, 20-point alpha grid on the quantized output error, exact norm compensation, searched
   before GPTQ. Go on a further 0.02 nats / 7 percent KL for `q3`; skip for NVFP4 if step 5's weight-only KL is already within 10 percent of
   the fake-quant floor.
7. Learned rounding (two to three weeks). Run AutoRound itself on Qwen3.8-27B with `bits=3, group_size=64, sym=True` (full-range) and
   `nv_fp4` to get the ceiling for free before writing anything; if it beats step 6 by 0.03 nats, port SignRound (V offsets plus scale
   multipliers, 50 to 200 iterations, block output MSE) into the converter. Go/no-go: the same thresholds as step 5, at 1 to 3 GPU hours.
8. P2E2 decision. With steps 2, 5 and 6 measured, convert the -3..3 variant and compare `q3-compressed` candidates by bytes and nats;
   this settles whether `docs/P2E2_FULL_RANGE.md`'s layout work is worth its kernel cost.

## 6. Risks and open questions

- The harness's KL (0.147 for `nvfp4`) is 5x AutoRound's measured 0.0285 for W4 g128 on the same base model [4]. Different text and
  metric details explain part of it, but the production path also quantizes the KV cache to NVFP4 and runs relaxed kernels. Until step 1
  separates the shares, no NVFP4 weight-rounding gain can be sized, and go/no-go thresholds for the NVFP4 forms are soft.
- One 1279-token document resolves 0.02 (NVFP4) to 0.05 (Q3) nats; most steps above promise less. The held-out KL run is not optional.
- Literature transfer: the published 3-bit numbers are g128 with zero points on Llama-2; Q3G64 is symmetric with a binary16 scale on a
  heavily trained hybrid model. The ordering of methods should transfer; the magnitudes may not.
- Calibration overfitting: GPTQ/AWQ fit 0.26 to 1 M tokens; a chat-only or thinking-only set can hurt general text (llama.cpp discussion [14]).
  Keep a mixed set and evaluate on text the set does not contain.
- GDN layers in PyTorch: the calibration pass needs a correct Qwen3.5 forward with the recurrent scan; a slow reference implementation
  multiplies stage-1 time by 5 to 10.
- Signed scales: compatibility is argued from reading the kernels, not from running them; the P2E2 layout still cannot carry -4.
- P2E2 size: calibrated and GPTQ codes are less peaked around zero; the 10.16 GiB figure will grow by an unmeasured amount.
- GPTAQ's propagation of quantized inputs is incompatible with per-object parallel writers; it needs the layer-by-layer pipeline.
- The AWQ fold into v rows changes the KV cache's per-channel statistics before the Hadamard rotation; measure KV error separately.
- Swift 1.5 versus base: the `q3` and `nvfp4` numbers in the baseline are base-model conversions; a deployed Swift `q3` has no measurement
  yet. Whether Swift re-trained the MTP layer is still undocumented and matters for the drafting-stack Hessians.
- Not retrieved: the AWQ paper's own tables and its AWQ+GPTQ claim, Four Over Six PTQ numbers, GPTAQ tables, "Pretraining LLMs with NVFP4",
  Qwen2.5/Qwen3-32B 3-bit numbers, the ISTA and UkisAI model cards. Treat the GSQ-RCO quality figures as community reports.

## 7. References

1. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", arXiv 2210.17323; code and LLaMA
   table: https://github.com/IST-DASLab/gptq (README, gptq.py, quant.py, llama.py).
2. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", 2023 (arXiv ID not verified here); code:
   https://github.com/mit-han-lab/llm-awq (awq/quantize/auto_scale.py, auto_clip.py).
3. Shao et al., "OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models", ICLR 2024, arXiv 2308.13137
   (https://arxiv.org/abs/2308.13137); code: https://github.com/OpenGVLab/OmniQuant.
4. Intel AutoRound (Cheng et al., "Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs", EMNLP 2024
   Findings; arXiv ID not verified here): https://github.com/intel/auto-round, docs/paper_acc.md, docs/gguf_alg_ext_acc.md,
   docs/mxnv_acc.md, docs/opt_rtn.md, docs/neuqi_acc.md, docs/full_range_sym.md, docs/awq_details.md, docs/step_by_step.md,
   auto_round/data_type/int.py, auto_round/data_type/nvfp.py.
5. Li et al., "GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration", ICML 2025, arXiv 2504.02692
   (https://arxiv.org/abs/2504.02692); code: https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ.
6. Cook et al., "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling", arXiv 2512.02010
   (https://arxiv.org/abs/2512.02010); code: https://github.com/mit-han-lab/fouroversix (src/fouroversix/utils.py).
7. Egiazarian et al., "Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization", arXiv 2509.23202
   (https://arxiv.org/abs/2509.23202); code: https://github.com/IST-DASLab/FP-Quant, https://github.com/IST-DASLab/qutlass.
8. Zhang et al., "MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization", NeurIPS 2024, arXiv 2406.00800;
   code: https://github.com/AozhongZhang/MagR.
9. Zhang et al., "LeanQuant: Accurate and Scalable LLM Quantization with Loss-error-aware Grid", ICLR 2025, arXiv 2407.10032;
   code: https://github.com/LeanModels/LeanQuant.
10. "NeUQI: Near-Optimal Uniform Quantization Parameter Initialization", arXiv 2505.17595 (https://arxiv.org/abs/2505.17595).
11. Dadgarnia et al., "GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling", arXiv 2604.18556;
    code and README tables: https://github.com/IST-DASLab/GSQ.
12. Helcig and Alistarh, "Model Compression with Exact Budget Constraints via Riemannian Manifolds", arXiv 2605.00649;
    code: https://github.com/IST-DASLab/RCO.
13. Model releases: https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF and
    https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF (cards not fetched; sizes and scores from search summaries and the
    Euhedral repository), https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b.
14. llama.cpp: https://github.com/ggml-org/llama.cpp (ggml/src/ggml-quants.c: make_qx_quants, quantize_row_q3_K_impl,
    quantize_row_q4_K_impl, make_qkx3_quants; tools/imatrix/README.md; imatrix PR 4861; discussion 5263 on imatrix calibration text).
15. NVIDIA TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer (modelopt/torch/quantization/config.py:
    MSE and LocalHessian calibration, LSQConfig, NVFP4 presets; utils/numeric_utils.py: E4M3_MAX_46;
    modelopt_recipes/configs/ptq/presets/model/nvfp4_four_over_six.yaml, nvfp4_awq_lite.yaml).
16. vLLM llm-compressor: https://github.com/vllm-project/llm-compressor (examples/quantization_w4a4_fp4/qwen3_8_gptq_awq_example.py,
    llama3_gptq_example.py, README.md; src/llmcompressor/modifiers/transform/awq/dynamic_mappings.py).
17. ModelCloud GPTQModel: https://github.com/ModelCloud/GPTQModel (README: GPTAQ notes; gptqmodel/quantization/gptq.py,
    fallback_smooth.py).
18. Liu et al., "Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models", arXiv 2504.04823.
19. "Quantized Reasoning Models Think They Need to Think Longer, but They Do Not", arXiv 2606.00206.
20. "LAQuant: A Simple Overhead-free Large Reasoning Model Quantization by Layer-wise Lookahead Loss", arXiv 2605.08755.
21. "Think Before You Prune: Selective Self-Generated Calibration for Pruning Large Reasoning Models", arXiv 2511.18864.
22. Unsloth, "Unsloth Dynamic 2.0 GGUFs", https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs (calibration set 300 K to 1.5 M tokens).
23. Huang et al., "How Good Are Low-bit Quantized LLaMA3 Models? An Empirical Study", arXiv 2404.14047 (70B GPTQ rows).
24. "Rethinking Residual Errors in Compensation-based LLM Quantization", arXiv 2604.07955 (GPTAQ follow-up; not read in full).
