# Rotations and outlier-smoothing transforms: QuIP/QuIP#, QuaRot, SpinQuant, FlatQuant, DuQuant, OSTQuant, HIGGS, Hadamard FP4 recipes

Measured against `00-baseline.md` (q3 +0.170 nats, nvfp4 +0.029, nvfp4-compressed +0.028 against its own BF16). Sources are
the papers' public repositories (cloned and read; links in section 7) plus search-result abstracts; arXiv pages were not reachable
from this environment, so numbers come from repository READMEs and result tables, and every number that is my own estimate is
marked as such. Simulation code: `scratchpad/sim/rot_sim.py`, `rot_sim2.py` (numpy, synthetic rows; reproduced in section 4).

## 1. Summary

The family multiplies the inputs of a linear layer by an orthogonal (or invertible) matrix and its weights by the inverse, so the
product is unchanged while activations lose their channel outliers and weights become approximately Gaussian ("incoherent").
Most of the rotation is free: a residual-stream rotation R1 folds into the embedding, the norm weights, the LM head and every linear
that reads from or writes to the residual. The rest runs online as fast Hadamards before down_proj, o_proj and (here) GDN
out_proj. The best-supported results are for W4A4 with INT4 per-token activations: with GPTQ weights, QuaRot/SpinQuant/FlatQuant keep
Llama-2 70B within 0.2 to 0.5 WikiText-2 perplexity and Qwen2.5-32B within 0.5 to 1.6 (FlatQuant 5.80, QuaRot 6.54 to 6.95, BF16
5.32); for
weight-only 4-bit per-channel quantization rotation plus GPTQ closes the per-channel RTN gap (Llama-2 7B 7.2 to 5.6, FP 5.5). For
Euhedral the verdict is narrow. (a) Weight-only: rotation cannot be applied without giving up the imatrix, because a signed Hadamard
makes the diagonal importance exactly flat; group-64 Q3 gains only if the model's groups are heavy-tailed, and the measured P2E2
ratio implies they are already near-Gaussian; block-16 NVFP4 gains nothing and loses the calibrated-scale leverage (toy estimate:
true weighted error 0.47% with imatrix against 0.73% rotated). Expected gain: none to small; a converter-only experiment decides it.
(b) Prefill activations: a single NVFP4 term is bounded by the Gaussian floor of E2M1 at block 16, 9.5% RMS, which is exactly what
the engine measured; rotation cannot lower it, so the two-term FP4 route stays. A single MXFP8 term (2.65% RMS, estimate) needs no
rotation at all and is the real prefill experiment for the Q3 forms. (c) KV: the engine's H256 is already SpinQuant R3/QuaRot's
key-value rotation; what remains is random signs (free), a free offline V rotation, and calibrated or learned per-head rotations, all
small. Decode cost of the online transforms, if ever needed: about 0.3 to 0.4 ms per token unfused (2% of 16 ms), near zero fused
into the SwiGLU, gated-norm and attention-merge kernels; they are row-exact by construction.

## 2. Mechanism

### 2.1 The transforms

**Computational invariance (QuaRot, SpinQuant).** For a pre-norm block, RMSNorm(xR) = RMSNorm(x)R for orthogonal R once the
norm's elementwise weight is folded into the following linear (`fuse_ln_linear`, then the norm weight is set to ones). The residual
stream is rotated once: embedding rows E become ER, every norm-fed linear W becomes WR^T (QuaRot `rotate_attention_inputs`,
`rotate_mlp_input`, `rotate_head`), every linear that writes the residual becomes R^T W (`rotate_attention_output`,
`rotate_mlp_output`). Nothing runs online for R1. QuaRot uses a randomized Hadamard (Sylvester H_{2^k} with random signs, or
H_K ⊗ H_{2^k} with a hand-coded Hadamard of order K in {12, 20, 28, 36, 40, 52, 60, 108, 140, 156, 172}); SpinQuant learns R1
(and a per-head R2) on the Stiefel manifold with Cayley SGD, keeping R3 and R4 as fixed Hadamards.

**The two places that need online work.** The SwiGLU output has no linear predecessor, so the down_proj input gets an online full
Hadamard R4, with H folded into down_proj's columns (`apply_exact_had_to_linear(W, had_dim=-1)`). The attention output is
rotated per head by folding H_{d_head} into the rows of W_v (R2, offline; attention is linear in V), and QuaRot completes a full
Hadamard with a small online transform of size num_heads across heads (`OnlineHadamard(self.num_heads)` on the transposed
attention output in `e2e/quantized_llama/modeling_llama.py`), with the full inverse folded into o_proj. Keys and queries get an
online per-head Hadamard after RoPE (R3) so the KV cache can be quantized; q·k is preserved.

**QuIP / QuIP# incoherence processing.** QuIP (2023) multiplies W by random orthogonal matrices on both sides (Kronecker-factored)
and rounds with LDLQ (GPTQ-like error feedback) against the layer's input Hessian H. QuIP# replaces this with the randomized
Hadamard transform: W' = H_m S_V W S_U H_n, H' = H_n S_U H S_U H_n with random sign vectors S_U, S_V, optionally preceded by a
diagonal rescaling (`scaleWH = (diag H / diag W^T W)^(1/4)`, a SmoothQuant-like balance), then E8-lattice codebooks and
fine-tuning (`lib/algo/quip.py`). QTIP keeps the incoherence step and replaces the lattice with trellis coding. The input-side
rotation must be applied to activations online (or folded into R1 when the input is the residual); the output-side rotation is free
only when the consumer is linear.

**HIGGS.** Hadamard rotation makes weights approximately N(0, σ²), so one MSE-optimal grid for the unit Gaussian serves every
layer: pairs of rotated weights (vector size 2) are rounded to a 2^(2b) point grid with b in {2, 3, 4} and a per-group scale
(FLUTE's `integrations/higgs.py`: grid shape [2^(bits·vector_size), vector_size], scales per `group_size`). The "linearity
theorem" relates the per-layer squared error to the perplexity increase; the method is data-free and runs on FLUTE lookup-table
kernels. Accuracy tables were not retrievable here (section 3 notes the one secondary number).

**FlatQuant.** Per-linear learned affine transforms P = P1 ⊗ P2 (Kronecker, n1·n2 = n, SVD-parameterised with Cayley orthogonal
factors and a learnable diagonal; `flatquant/trans_utils.py`) applied online to every linear input, including the normalized
hidden state before q/k/v and gate/up (`ln_trans`), a transform across heads before o_proj (`o_trans`, size num_heads) and per-head
transforms for the K and V caches; learned layer by layer in hours with learnable clipping (LWC/LAC). Because P is not folded into
the residual, FlatQuant has online work before every linear; its kernel is a fused Kronecker matmul plus quantization. Any
factorization works, so odd widths such as 17408 are no obstacle.

**DuQuant, OSTQuant.** DuQuant builds block-diagonal rotations (block 128) greedily around measured activation outliers, zigzag-permutes
channels so blocks have balanced outliers, and smooths; it identifies "massive outliers" at down_proj. DuQuant++ (2026) uses block
32, aligned to the MXFP4 group, with optional GPTQ. OSTQuant learns one global orthogonal R_res plus scaling pairs S_attn, S_ffn in
four equivalence pairs with a KL-top loss; everything fuses into weights.

**FP4 recipes with Hadamards.** NVIDIA's NVFP4 pretraining recipe uses 16×16 random Hadamards only on the weight-gradient GEMM
operands; TransformerEngine's `NVFP4BlockScaling` states RHT "is currently only applied to column-wise usage so that it can be used
for wgrad GEMM"; forward activations are not rotated. Quartet and Quartet II (NVFP4: MS-EDEN unbiased rounding, "four-over-six"
forward scaling, randomized block rotations in the backward pass) ship QuTLASS kernels that fuse transform, quantization and scale
computation with rotation sizes 16/32/64/128. MR-GPTQ (ICLR 2026) quantizes weights for MXFP4/NVFP4 with block-wise Hadamards whose
block equals the microscaling group (FP-Quant: `hadamard_group_size` in {16, 32, 64, 128} for NVFP4, default 128), MSE grids and
static activation reordering; its stated finding is that rotations help MXFP4 but can hurt NVFP4 under round-to-nearest. ModelOpt
exposes block-granular RHT (`RotateConfig`) and an NVFP4 KV recipe that rotates K and Q but not V. SageAttention3 quantizes
attention operands to FP4 with 1×16 groups and a two-level scaling of P.

### 2.2 Architecture mapping for Qwen3.8-27B

Notation: R1 is the 5120-wide residual rotation (H_20 ⊗ H_256 with random signs; 5120 = 20·256 is a supported `20N` size of the
Dao kernel and of QuaRot's `had20`). T_d, T_o, T_g are online transforms on the inputs of down, o_proj and GDN out_proj. "(1+w)"
is the engine's unit-offset RMSNorm weight (`RMS_NORM_UNIT_OFFSET`, `norm/kernels.cu`: `weight + weight_offset`), folded by storing
w = 0. Column folds are W' = W diag(1+w) R1^T; row folds are W' = R1 W.

| Object (per layer unless noted) | Input side | Output side | Status |
|---|---|---|---|
| `text/token_embedding` (Q3) | one-hot: nothing | rows E R1 | free, provable |
| `input_norm`, `post_attention_norm`, `final_norm`, MTP norms | weight folded into consumer, stored as zeros | | free, provable (RMSNorm is rotation-equivariant without its weight) |
| `attention/query_key` (q 6144 + k 1024 rows, Q4) | R1 and (1+w) folded | q,k → per-head RMSNorm with weight, RoPE: no offline rotation; online H256 after RoPE already exists | free on input; output blocked, provable |
| `attention/gate_value` (gate 6144 + v 1024 rows, Q5) | R1 folded | gate → sigmoid: none. v → cache: per-head R_v foldable into W_v rows because attention is linear in V and the merge kernel un-rotates before gating (`nvfp4_attention.cuh`: `hadamard256` then `/ (1 + expf(-gate))`) | free on input; V-side fold free, provable |
| `attention/output` o_proj 5120×6144 (Q3) | T_o online after the sigmoid gate; W' = R1 W_o T_o^T. SpinQuant's offline R2 is blocked: diag(sigmoid(g)) does not commute with a per-head rotation | R1 folded | input online (per-head H256 fits the merge kernel); output free |
| `gdn/query_key`, `gdn/value_z` (in_proj_qkv, in_proj_z) | R1 and (1+w) folded | conv1d (per channel, 4 taps) then SiLU: no rotation of any output; z → SiLU gate: none | free on input; output blocked, provable |
| `gdn/a_projection`, `gdn/b_projection` (BF16, 48 rows) | R1 folded (BF16 store) | softplus/sigmoid scalars: none | free, provable |
| delta rule (state 48×128×128 FP32) | a rotation of v commutes with the recurrence (S ← αS + δk^T is linear in v) and with the per-head RMS, but conv1d+SiLU before it and the (128-vector) norm weight and SiLU(z) gate after it block any offline use | | blocked, provable |
| `gdn/norm` weight (128, shared by 48 heads) | foldable into out_proj columns as a diagonal; irrelevant to rotation | | free |
| `gdn/output` out_proj 5120×6144 (Q3) | T_g online after the z gate; W' = R1 W_out diag(w_norm) T_g^T | R1 folded | input online (per-head H128 fits the gated-norm kernel); output free |
| `mlp/gate_up` 34816×5120 (Q3) | R1 and (1+w) folded | SwiGLU: none | free, provable |
| `mlp/down` 5120×17408 (Q3) | T_d online after SwiGLU; W' = R1 W_down T_d^T. 17408 = 17·1024 is not a Dao/QuaRot size; options: 17 blocks of H_1024, H_68 ⊗ H_256 (order 68 exists by the Paley construction, q = 67), or a FlatQuant 128×136 Kronecker | R1 folded | input online; output free |
| `text/output_head`, `text/draft_head` | R1 and (1+w_final) folded | logits: none | free, provable |
| `mtp/input_projection` fc 5120×10240 | embedding half: (1+w_e) and R1^T folded. Hidden half: the seed is `(1+w_f) ⊙ h_u` and its RMS is taken after the weight, so h_u = R1^T h'_u must be un-rotated online (one H_5120 per MTP row) before `hidden_norm` | rows R1 if the MTP residual is rotated | needs one small online transform per MTP row |
| `mtp/final_norm` → shared LM head | the head absorbed (1+w_f), the MTP output carries (1+w_m): multiply by (1+w_m)/(1+w_f) in the original basis, then rotate (H_5120) | | provable, one small online transform per draft row |
| DFlash2 drafter | taps are R1 x_l: fold R1^T into its `fc` (BF16, 5 blocks); its embedding input must be un-rotated (8 rows per block); its output needs the same ratio-and-rotate before the output head | | provable, small online work |
| KV cache K, Q | existing online H256 after RoPE; random signs D are free (same D on Q and K) | | in place |
| KV cache V | existing online H256; can move offline into W_v rows (R_v = H256 D_v); the merge then applies D_v then H256 (not self-inverse once signs are added) | | free |

Two provable negatives for this architecture: SpinQuant's R2 (offline per-head v/o rotation) is unavailable because the sigmoid
output gate sits between attention and o_proj, and nothing on the GDN value path can be rotated offline because conv1d+SiLU
precede the recurrence and an elementwise weight and gate follow it. Everything that QuaRot does online stays online here, plus
one more online transform per GDN layer (48 of 64 layers).

### 2.3 Online transform cost (estimates)

Per token at decode: 64 T_d (17408 wide), 16 T_o and 48 T_g (6144 wide), plus 2 H_5120 per draft row. Arithmetic with
block-diagonal power-of-two Hadamards: 64·17408·10 + 64·6144·8 to 9 adds ≈ 14 to 15 Mflop per token, negligible. Launch latency
dominates if the transforms are separate kernels: 128 kernels at 2 to 3 us inside a CUDA graph ≈ 0.25 to 0.4 ms, 1.5 to 2.5% of
a 16.0 ms token. Fused into the kernels that already hold the row (SwiGLU epilogue holding a 17408 row in shared memory, the
per-(row, head) gated-norm CTA for H128, the per-head merge warp for H256), the cost is the arithmetic only. The engine's own data
point: the per-split query H256 in attention decode costs 2.5% at 16K and 0.7% at 64K (`ATTENTION_DECODE.md`), an upper bound for
a transform that runs once per row. Prefill: one extra read-write pass of the activation per transformed linear, about 1 to 2% of
the chunk's GEMM time (estimate). FlatQuant's measurement of unfused online transforms: Llama-2-7B decode at batch 1 on an RTX 3090
runs at 0.70× (QuaRot) and 0.71× (FlatQuant) of FP16 against 0.81× for plain Int4, so about 13% of a step in that implementation.

## 3. Published results

Reproduced from repository READMEs (FlatQuant Table 1 and 4, SpinQuant "Quantized Models", OSTQuant "Main Results",
Qwen3-Quantization Table 4). WikiText-2 perplexity unless noted; zero-shot averages are each paper's task set.

**Table 3.1: W4A4, INT4 per-token activations, WikiText-2 PPL (FlatQuant README Table 1; 4-bit weights and activations).**

| Method | W rounding | Llama-2 7B | 13B | 70B | Llama-3 8B | 70B |
|---|---|---|---|---|---|---|
| FP16 | | 5.47 | 4.88 | 3.32 | 6.14 | 2.86 |
| QuaRot | RTN | 8.56 | 6.10 | 4.14 | 10.60 | 55.44 |
| SpinQuant | RTN | 6.14 | 5.44 | 3.82 | 7.96 | 7.58 |
| FlatQuant | RTN | 5.79 | 5.12 | 3.55 | 6.98 | 3.78 |
| QuaRot | GPTQ | 6.10 | 5.40 | 3.79 | 8.16 | 6.60 |
| SpinQuant | GPTQ | 5.96 | 5.24 | 3.70 | 7.39 | 6.21 |
| FlatQuant | GPTQ | 5.78 | 5.11 | 3.54 | 6.90 | 3.77 |

**Table 3.2: Qwen2.5-Instruct, W4A4 (FlatQuant README Table 4): WikiText-2 / C4 PPL, zero-shot QA average.**

| Model | BF16 | QuaRot RTN | QuaRot GPTQ | FlatQuant RTN |
|---|---|---|---|---|
| 7B | 8.36 / 14.37, 70.75 | | | 8.46 / 13.94, 68.62 |
| 32B | 5.32 / 10.45, 75.10 | 6.95 / 12.17, 70.24 | 6.54 / 11.65, 72.25 | 5.80 / 10.86, 74.89 |

Qwen outlier evidence: a plain Hadamard (QuaRot) leaves +23% perplexity on Qwen2.5-32B even with GPTQ, against +33% on Llama-3 8B
and +14% on Llama-2 70B; the learned affine transform recovers most of it. Secondary snippets (not read in full): TwinQuant reports
SpinQuant W4A8 at 10.9 WikiText-2 PPL on Qwen3-32B; MosaicQuant reports that QuaRot and Atom "still degrade noticeably" at W4A4 on
LLaMA3 and Qwen3; the Qwen3 quantization study attributes large-model sensitivity at W4A8 to activation outliers.

**Table 3.3: weight-only 4-bit, per-channel, WikiText-2 PPL (SpinQuant README W4A16KV16; OSTQuant README 4-16-16 in parentheses).**

| Method | Llama-2 7B | Llama-2 13B | Llama-2 70B | Llama-3 8B | Llama-3 70B |
|---|---|---|---|---|---|
| FP | 5.5 (5.47) | 5.0 (4.88) | 3.3 | 6.1 (6.14) | 2.8 |
| RTN | 7.2 (7.02) | 6.4 (6.39) | 4.6 | 7.8 (8.13) | 1e5 |
| GPTQ | 11.3 (9.84) | 5.6 (5.79) | 3.9 | 7.2 (7.43) | 1e5 |
| AWQ / OmniQuant (OSTQuant table) | (5.83 / 5.74) | (5.07 / 5.02) | | (7.36 / 7.19) | |
| QuaRot (GPTQ in rotated basis) | 5.6 (5.62) | 5.0 (5.00) | 3.5 | 6.4 (6.53) | 7.9 |
| SpinQuant | 5.6 (5.58) | 5.0 (5.00) | 3.5 | 6.4 (6.49) | 4.8 |
| OSTQuant | (5.64) | (4.94) | | (6.53) | |

Read: with per-channel 4-bit weights, rotation plus GPTQ reaches 0.1 to 0.3 PPL above FP where RTN loses 1 to 2; it is on par with
AWQ and OmniQuant, which use calibration instead. No rotation-only W3A16 table exists in these sources; the 3-bit evidence is for
Hadamard plus non-uniform codebooks (QuIP#, HIGGS, QTIP; see the codebook chapter). Group-wise scales already capture most of the
per-channel gap, which is the setting Euhedral is in.

With the KV cache also at 4 bits (W4A4KV4, SpinQuant README), SpinQuant reaches 5.9 / 7.3 / 3.8 PPL on Llama-2 7B, Llama-3 8B
and Llama-2 70B against QuaRot's 6.4 / 8.0 / 3.9; OSTQuant reports 5.91 and 7.29 on the first two.

**Table 3.4: Qwen3-8B weight-only, group 128 (Qwen3-Quantization README, Table 4 of the 2026 journal version).**

| Method | Bits | WikiText-2 PPL | MMLU 5-shot |
|---|---|---|---|
| FP16 | 16 | 9.71 | 74.7 |
| GPTQ | W4 | 9.96 | 73.4 |
| AWQ | W4 | 11.3 | 69.3 |
| AWQ | W3 | 27.5 | 33.2 |
| GPTQ | W2 | 52.1 | 25.0 |

For scale: Euhedral's calibrated Q3 on the 27B model is +0.170 nats, perplexity ×1.19; Qwen3-8B at W3 g128 with AWQ is ×2.8
(smaller model, different method), and at W4 g128 GPTQ is +0.025 nats, close to the nvfp4 artifact's +0.029 at 4.5 bpw.

**Table 3.5: FP4 activations and Hadamards in training and attention (abstract-level claims).**

| Source | Claim |
|---|---|
| NVFP4 pretraining (NVIDIA, arXiv 2509.25149) | 12B model, 10T tokens, loss and accuracy comparable to FP8; 16×16 RHT only on weight-gradient GEMM operands; stochastic rounding on gradients; 2D 16×16 weight scales; about 15% of layers kept in higher precision |
| TransformerEngine `NVFP4BlockScaling` | RHT applied to the column-wise (wgrad) operands only; forward activations not rotated; weights 2D-scaled |
| Quartet II (arXiv 2601.22813, ICML 2026) | MS-EDEN unbiased rounding with randomness in the microscale, four-over-six forward scaling, randomized block rotations in backward; up to 4.2× over BF16; validated to 1.9B params / 38B tokens |
| MR-GPTQ (arXiv 2509.23202, ICLR 2026) | naive MXFP4 W4A4 loses about 10% accuracy; block-wise Hadamard (block = microscaling group) plus GPTQ brings MXFP4 close to NVFP4; rotations help MXFP4 but can hurt NVFP4 with RTN; QuTLASS: up to 3.6× layer / 2.2× end-to-end on B200, 6× / 4× on RTX 5090 over FP16 |
| SageAttention3 (arXiv 2505.11594) | FP4 attention with 1×16 groups and two-level scaling of P, 1038 TOPS on RTX 5090 (5× FlashAttention); the repository recommends SageAttention2 for precision-sensitive applications |
| RotateKV (arXiv 2501.16383) | 2-bit KV via outlier-aware (channel-reordered) FWHT, pre-RoPE grouped-head rotation, attention-sink-aware quantization; under 0.3 PPL loss on Llama-2-13B |
| Quamba2 (arXiv 2503.22879) | W4A8/W4A16 for Mamba2 with Hadamards fused into in_proj and out_proj (`fuse_hadamard`), a gated-SSM precedent for T_g |

## 4. Fit with Euhedral

### 4.1 What rotation does to the engine's grids (simulation, my estimates)

Synthetic rows of 4096 (256 rows), quantized with the engine's recipes (Q3/Q4 group 64 RTN and the 39-candidate search with
least-squares refit; NVFP4 block 16 with E4M3 scale, RTN and the -6..+2 scale-code search; MXFP8 E4M3 with UE8M0 per 32), before
and after a random-sign Sylvester Hadamard along the row. Relative MSE in %. "t5", "laplace" and "mix" (90% narrow, 10% wide
Gaussians) stand in for heavy-tailed weight rows; after rotation every distribution is Gaussian.

| Quantizer | Gaussian | Laplace | Laplace rotated | t5 | t5 rotated | mix | mix rotated |
|---|---|---|---|---|---|---|---|
| Q3 g64 RTN | 5.50 | 8.44 | 5.50 | 9.10 | 5.49 | 13.17 | 5.51 |
| Q3 g64 search (engine) | 3.60 | 5.81 | 3.59 | 6.23 | 3.60 | 9.53 | 3.60 |
| Q4 g64 search (engine) | 0.90 | 1.56 | 0.90 | 1.66 | 0.90 | 2.59 | 0.90 |
| NVFP4 b16 RTN | 0.90 | 0.86 | 0.91 | 0.86 | 0.91 | 0.82 | 0.91 |
| NVFP4 b16 search (engine) | 0.74 | 0.73 | 0.74 | 0.73 | 0.74 | 0.73 | 0.74 |
| Lloyd-Max 8-level grid, RMS scale per 64 (HIGGS-style 3-bit) | 2.93 | 4.96 | 2.93 | 5.36 | 2.93 | 9.02 | 2.94 |
| Lloyd-Max 16-level grid, RMS scale per 64 | 0.69 | 1.24 | 0.69 | 1.57 | 0.69 | 3.06 | 0.70 |

Three readings. Group-64 integer grids gain 35 to 60% lower MSE from Gaussianization when rows are heavy-tailed and nothing when
they are Gaussian. Block-16 NVFP4 is already insensitive to tails and rotation makes RTN slightly worse (0.82 to 0.91), the effect
MR-GPTQ reports. Rotation changes the optimal grid: on Gaussian data the 8-level Lloyd-Max grid beats the searched uniform Q3 grid by
19% and the 16-level grid beats searched Q4 by 23% and searched NVFP4 (4.5 bpw) at 4.25 bpw; whether such grids can satisfy
constraints 1 and 2 (E4M3-exact codes, BF16-exact products) is the codebook chapter's question. Are this model's groups heavy-tailed?
The P2E2 coding of RTN Q3 costs 2 bits plus 2 per BIG code; the measured 81.5% of row-split bytes (2.65 of 3.25 bpw with scales byte
for byte) implies 2.40 bits per code, so about 20% BIG codes. A Gaussian group under amax/3 scaling gives 20.5% (simulation); t5 and
mix rows give 11.5% and 6%. The real tensors therefore already look Gaussian in this statistic, and Gaussianization has little left to
give on Q3 (inference, to be checked directly on the weights in experiment E1).

**Importance.** With R1 a signed Hadamard, diag(R1 H R1^T) = mean(diag H) for any diagonal H, exactly; with the real (non-diagonal)
H the diagonal is flat up to fluctuations of order ||offdiag H||_F / n. The imatrix is this diagonal, so in the rotated basis the
engine's weighted scale search degenerates to the unweighted search, and the information moves to the off-diagonals of H', which
only error-feedback rounding (GPTQ/LDLQ, rounding chapter) can use. Toy (t5 weights, lognormal importance with max/mean 150 to 520,
true error tr(ΔW H ΔW^T) in %): Q3 unrotated RTN 8.9 to 9.5, unrotated search with imatrix 4.4 to 4.6, rotated search 3.5 to 3.8
(rotation wins); NVFP4 unrotated search with imatrix 0.44 to 0.47, unweighted 0.71 to 0.75, rotated 0.73 to 0.78 (rotation loses
about 60%). Output-side rotations (R1 on down/o_proj/out_proj rows, R_v on W_v rows) mix rows, which Gaussianizes those weights along
K while leaving their input-column importance exactly meaningful; but R1 cannot be applied to outputs without also rotating the
inputs of the norm-fed linears, because the residual basis is shared.

### 4.2 Activations in prefill (simulation against the engine's measurements)

Per-element relative RMS error, Gaussian rows versus rows with 8 of 4096 channels 20× larger (massive activations), before and after
rotation: single NVFP4 term 9.51% (Gaussian), 9.28% (outliers), 9.55% (outliers, rotated); two NVFP4 terms 0.87%; single MXFP8 term
2.65% everywhere; INT4 per-row absmax 15.4% (Gaussian), 67.7% (outliers), 14.5% (rotated). The engine measured 9.5% output RMS
error for one NVFP4 term and 0.88% for two (`00-baseline.md` section 5), the same numbers: the single-term error is the Gaussian
floor of E2M1 at block 16, not an outlier effect, because the per-16 scale already isolates outliers. Rotation can remove what sits
above that floor, which here is nothing. The W4A4 literature lives at this error level (their INT4 per-token activations are worse
still, and rotation is what makes them usable); the engine's "double the KL" is the same quality class, rejected by a stricter bar.
A single E4M3 term has a scale-free 3-bit mantissa error that rotation does not touch; its 2.65% RMS is one twelfth of the variance of
the FP4 term. If KL scales with error variance (the one-term FP4 run added about one unit of the artifact's own KL), one-term MXFP8
would add roughly (2.65/9.5)² ≈ 8% of that: about +0.01 KL (estimate, prefix-only exposure in the harness).

### 4.3 The eight constraints

1. **Row-exact 1 to 8 rows.** Weight formats and decode kernels are unchanged by offline rotation. An online Hadamard is a per-row
   butterfly with a fixed order (`nvfp4kv::hadamard256` already is), so each row's bits depend on its own inputs only; the row twin
   runs the same code per row and is bitwise the one-row result. Confirmed by construction; the determinism and exactness claims in
   the prompt hold provided the M-row kernel uses the one-row code path per row (an MMA-based Hadamard also qualifies, since a
   tensor-core row never depends on other rows).
2. **Prefill at 9+ rows.** Formats unchanged, so the MXFP8 (Q3) and native FP4 (NVFP4) routes keep working with rotated weights after
   the online transform. Rotation does not unlock single-term FP4 activations (4.2). Single-term MXFP8 on the Q3 route is viable
   without rotation: the MMA work halves (204 → 408 TFLOPS logical), the producer-side weight expansion and per-group FP32 scale FFMAs
   do not shrink, so 1.3 to 1.7× on gate/up and down (estimate; 126 TFLOPS achieved today), and nothing changes for attention and GDN
   prefill. The nvfp4 forms have no FP8 path on the FP4 tensor cores and stay two-term.
3. **Decode instruction budget.** The GEMV is untouched; transforms are separate passes or epilogues (2.3). Fused, they cost
   arithmetic only; unfused, about 2% of a token.
4. **Determinism.** Fixed butterfly order, graph-captured, same hashes run to run; no atomics.
5. **Offline conversion.** The folds are FP32/FP64 matmuls in the converter (gate_up: 34816×5120 by 5120×5120, about 1 s on the GPU per
   tensor; minutes in total). New tensor-descriptor fields: "input expects transform T with sign vector s" per linear, and sign vectors
   (about 3 KB per layer); existing artifacts carry no flags and load as before.
6. **The artifact decides.** The profile reads the flags and selects fused kernels with transforms; no user option.
7. **Memory.** Rotation adds no bytes. P2E2: Gaussianized groups code at 2.41 bits (simulation), the same as today's 2.40, so
   `q3-compressed` size would not change; heavy-tailed tensors, if any, would compress worse. SD4: Gaussian blocks spread their
   E4M3 scale codes over fewer values (top 16 codes cover 99.7% of blocks for Gaussian rows against 79 to 94% for heavy-tailed rows),
   so rotation would make the 16-entry table fit better, but SD4 already costs only +0.005 nats against plain NVFP4.
8. **Drafting precision separate.** The MTP layer and DFlash2 drafter can stay in the original basis; they need one un-rotation of the
   seed hidden (or embedding) and one rotation with the final-norm ratio before the shared output head, both 5120-wide per draft row.

### 4.4 Kernels, layouts and converter stages that change

Converter: a `--rotate` stage producing R1 (signed H_20 ⊗ H_256), folding norms and R1 in FP64 before quantization, zeroing norm
weights, writing sign vectors and per-linear transform flags; importance handling switches to uniform for rotated inputs (the flat
diagonal) and keeps the imatrix for non-rotated inputs (down, o_proj, out_proj in E1). Engine, only if online transforms are adopted:
a 17408-wide transform after SwiGLU (blockwise H_1024 in-CTA, or H_68 ⊗ H_256), per-head H256 after the gate in the attention merge,
per-head H128 after the gate in the GDN gated norm, two H_5120 kernels for MTP/DFlash2 rows, and the matching column folds in the
converter. Layout: none. Verifier: the same per-row transform code in row-exact mode.

### 4.5 Expected gains in the repository's metrics

Weight-only: none to small for q3 (Gaussianization of already near-Gaussian groups against the loss of imatrix leverage; sign
unknown, resolvable only above the harness's 0.02 to 0.05 nat floor) and likely negative for nvfp4 and nvfp4-compressed (block-16
grids gain nothing; the calibrated scale search loses its weights). Rotation becomes valuable only paired with error-feedback rounding
or with Gaussian-optimal codebooks (QuIP#/HIGGS/QTIP), where it is the enabling step: one fixed grid for every group, data-free.
Prefill: no gain attributable to rotation; the single-term MXFP8 experiment is independent of it. KV: the engine already has the
rotation; random signs remove the one structural weakness of a plain Sylvester transform (row 0 of H256 sums all 256 channels, so a
key with a nonzero channel mean μ puts 16μ into one coordinate), and moving the V rotation offline is free.

### 4.6 MTP, DFlash2 and the verifier

The MTP seed is the post-final-norm hidden, i.e. (1+w_f) ⊙ h_u; the rotated model only has R1 h_u because the LM head absorbed
(1+w_f). Exact recovery is R1^T (one H_5120 per MTP row), then the weight, then `hidden_norm` as today. The MTP's own final norm
weight w_m differs from w_f, so its output must be scaled by (1+w_m)/(1+w_f) and rotated before the head; the recursion (m_k as the
next seed) stays in the original basis. DFlash2: fold R1^T into `fc` (free), un-rotate its 8 embedded block rows, ratio-and-rotate its
7 output rows. The drafter's taps "read only the residual" and remain the one-row values in verification. None of this touches
correctness: drafts are verified exactly whatever basis they are computed in, and the verifier's rows go through the same per-row
transforms as one-row decode. Acceptance may shift slightly because the drafter's BF16 `fc` rounds differently after the fold.

## 5. Experiment plan

E1. **R1-only rotated artifacts, converter only, no kernel change.** Fold R1 into embedding, norms, all norm-fed inputs, all
residual-writing outputs, LM head, draft head; uniform importance on rotated inputs, imatrix elsewhere; disable speculation. Build
`q3` and `nvfp4` variants (base model, same imatrix as today) and run `TeacherForcedQualityCudaIntegrationTest` plus
`compare_reference.py`. Also dump, per tensor, the fraction of BIG codes and the group kurtosis before and after folding to test the
near-Gaussian inference of 4.1. Go if KL falls by 10% or more (q3 0.293 → ≤ 0.264; nvfp4 0.147 → ≤ 0.132) or dNLL improves beyond
two standard errors; otherwise stop the weight-only track and hand the rotated-basis question to the rounding chapter (GPTQ/LDLQ in
the rotated basis, where it belongs). Cost: a day of converter work, two conversions.
E2. **Online transforms for down, o_proj, out_proj** (only after E1 passes): per-head H256 in the merge kernel, per-head H128 in the
gated norm, blockwise H_1024 after SwiGLU; column folds; MTP/DFlash2 un-rotate and re-rotate kernels. Measure decode tok/s at 4K
and 32K (accept ≤ 2% loss), `SpeculativeVerifyCudaIntegrationTest` and `SpeculativeDecodeCudaIntegrationTest` (must pass bitwise),
MTP acceptance (within 0.05 tokens per verification), and quality as in E1.
E3. **Single-term MXFP8 activations on the Q3 prefill route, no rotation.** A kernel variant that drops the lo term; measure gate/up
and down TFLOPS on 512-row chunks and the teacher-forced report with the 1281-token prefix through the new route. Go if gate/up
reaches ≥ 1.3× and KL rises by ≤ 0.01 (the drift-floor test resolves 3e-3). Separately report TTFT at 32K and 128K.
E4. **KV rotation study, offline.** Dump K and V of the chat corpus from the engine (per layer, per head), compare per-head NVFP4
block-16 error for H256, H256 with random signs, and a per-layer rotation calibrated on the dumps (Cayley, FP32, minutes); check the
channel-mean concentration of plain H256. Adopt signs if the RMS error falls ≥ 10%; move the V rotation offline if the Q5 and NVFP4
error of the rotated W_v rows is not worse. Bit widths stay with the KV chapter.
E5. **Codebook handoff.** If E1 shows Gaussian groups, measure the Lloyd-Max 8-level and 16-level grids with RMS scales on the real
rotated tensors against searched Q3/Q4/NVFP4 (the codebook chapter decides E4M3-exact grid points and kernels).

## 6. Risks and open questions

- The near-Gaussian inference from the P2E2 ratio is indirect; E1's code statistics settle it. If groups are heavy-tailed in a few
  tensors only, a per-tensor rotation decision is not available for R1 (shared basis) but is for the output-side folds.
- Learned rotations (SpinQuant, OSTQuant) need the quantized 27B model in the loop; a 16 GB card cannot hold it in BF16, so only
  Hadamard-grade (QuaRot) rotations are feasible locally; FlatQuant-style layer-wise calibration fits per layer but adds online
  transforms before every linear, which is the cost profile Euhedral's decode does not want. See the fine-tuning chapter.
- 17408 has no fast Hadamard in the usual libraries (Dao: 12N/20N/28N/40N; QuaRot: K in {12..172}); the Dao kernel's zero padding to
  32768 is not a 17408-point orthogonal transform. H_68 ⊗ H_256 (Paley, q = 67), 17 blocks of H_1024, or a Kronecker 128×136 are the
  options; blockwise is what ModelOpt and FP-Quant do for odd widths.
- Random-sign variance: SpinQuant found the best of 100 random rotations beat the worst by 13 points at W4A4 on Llama-2 7B; weight-only
  effects are smaller but a seed should be fixed and recorded in the artifact.
- BF16 storage of the rotated residual changes rounding patterns of the residual stream; the relaxed-drift test should be rerun on a
  rotated artifact.
- Folding R1 into `in_proj_a` and `in_proj_b` (BF16, FP32 control path) changes alpha and beta slightly; the GDN state is FP32 and
  unquantized, so no further interaction is expected.
- 2025-2026 follow-ups seen only as titles and abstracts (reference 19) were not read; none is known to change the conclusions above,
  but "Transforms for LLM Quantization: The Great Inversion and Format Co-Design" (2608.25188) bears on the grid question of 4.1.
- Open: does the Swift-1.5 fine-tune change the outlier structure relative to the base model (its imatrix differs)? E1 can be run on
  both checkpoints at nvfp4-compressed cost.

## 7. References

1. QuaRot: Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", arXiv 2404.00456 (NeurIPS 2024). Code read:
   https://github.com/spcl/QuaRot (`fake_quant/rotation_utils.py`, `e2e/quantized_llama/modeling_llama.py`).
2. SpinQuant: Liu et al., "SpinQuant: LLM Quantization with Learned Rotations", arXiv 2405.16406 (ICLR 2025). Code and results table:
   https://github.com/facebookresearch/SpinQuant.
3. FlatQuant: Sun et al., "FlatQuant: Flatness Matters for LLM Quantization", arXiv 2410.09426 (ICML 2025). Code and Tables 1-7:
   https://github.com/ruikangliu/FlatQuant.
4. DuQuant: Lin et al., "DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs", arXiv 2406.01721
   (NeurIPS 2024 oral). https://github.com/Hsu1023/DuQuant. DuQuant++: "Fine-grained Rotation Enhances Microscaling FP4 Quantization",
   arXiv 2604.17789 (2026), https://github.com/Hsu1023/DuQuant-v2.
5. OSTQuant: Hu et al., "OSTQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better
   Distribution Fitting", ICLR 2025, arXiv 2501.13987. Results table: https://github.com/BrotherHappy/OSTQuant.
6. QuIP: Chee et al., "QuIP: 2-Bit Quantization of Large Language Models With Guarantees", NeurIPS 2023,
   https://openreview.net/pdf?id=xrk9g5vcXR (arXiv ID not verified here).
7. QuIP#: Tseng et al., "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks", arXiv 2402.04396
   (ICML 2024). Code: https://github.com/Cornell-RelaxML/quip-sharp (`lib/algo/quip.py`, `lib/utils/matmul_had.py`).
8. QTIP: Tseng et al., "QTIP: Quantization with Trellises and Incoherence Processing", arXiv 2406.11235 (NeurIPS 2024).
   https://github.com/Cornell-RelaxML/qtip.
9. HIGGS: Malinovskii et al., "Pushing the Limits of Large Language Model Quantization via the Linearity Theorem", arXiv 2411.17525
   (NAACL 2025). Integration read: https://github.com/HanGuo97/flute (`flute/integrations/higgs.py`, README) and
   https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/quantization/higgs.md. FLUTE: arXiv 2407.10960.
10. NVIDIA, "Pretraining Large Language Models with NVFP4", arXiv 2509.25149. Recipe code:
    https://raw.githubusercontent.com/NVIDIA/TransformerEngine/main/transformer_engine/common/recipe/__init__.py (`NVFP4BlockScaling`).
11. Quartet: Castro et al., "Quartet: Native FP4 Training Can Be Optimal for Large Language Models", arXiv 2505.14669 (NeurIPS 2025),
    https://github.com/IST-DASLab/Quartet. Quartet II: Panferov et al., arXiv 2601.22813 (ICML 2026), https://github.com/IST-DASLab/Quartet-II.
12. MR-GPTQ: Egiazarian et al., "Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization", arXiv 2509.23202
    (ICLR 2026). Code: https://github.com/IST-DASLab/FP-Quant; kernels: https://github.com/IST-DASLab/qutlass; HF doc:
    https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/quantization/fp_quant.md.
13. SageAttention3: Zhang et al., "SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training",
    arXiv 2505.11594 (NeurIPS 2025). https://github.com/thu-ml/SageAttention (`sageattention3_blackwell/README.md`).
14. RotateKV: Su et al., arXiv 2501.16383. https://github.com/ZunhaiSu/RotateKV.
15. Quamba2: Chiang et al., "Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models",
    arXiv 2503.22879 (ICML 2025). https://github.com/enyac-group/Quamba.
16. Zheng et al., "An Empirical Study of Qwen3 Quantization", arXiv 2505.02214; Visual Intelligence 4:11 (2026).
    https://github.com/Efficient-ML/Qwen3-Quantization.
17. Dao, fast-hadamard-transform: https://github.com/Dao-AILab/fast-hadamard-transform (12N/20N/28N/40N interfaces).
18. NVIDIA TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer (`RotateConfig`, `normalized_hadamard_transform`,
    `modelopt_recipes/configs/ptq/units/kv_nvfp4_rotate.yaml`, `plugins/vllm_indexer.py`, `plugins/gated_delta_net`).
19. Titles seen in search results only (not read): arXiv 2511.04214, 2511.22316, 2602.17681, 2608.25188, 2510.27527, 2609.02846,
    2604.11080, 2606.01556, 2606.15652.
20. Euhedral sources cited: `docs/PREFILL_MX.md`, `NVFP4_NATIVE.md`, `ATTENTION_DECODE.md`, `MTP_CONTRACT.md`, `MTP_VERIFIER.md`,
    `DFLASH2.md`; `native/src/attention/{nvfp4_kv,nvfp4_attention,kv_append}.cuh`, `native/src/gdn/{gated_norm,convolution,recurrence}.cuh`,
    `native/src/norm/kernels.cu`; `tools/euhedral_artifacts/{inventory,grouped,nvfp4}.py`.
