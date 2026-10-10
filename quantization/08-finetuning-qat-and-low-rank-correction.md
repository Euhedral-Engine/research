# Fine-tuning after rounding: QAT, block reconstruction, scale-only tuning, low-rank branches, drafter training

Methods that train something after (or during) low-bit rounding: block-wise reconstruction (OmniQuant, EfficientQAT Block-AP, GSQ),
end-to-end QAT and distillation (EfficientQAT E2E-QP, ParetoQ, LLM-QAT, BitDistiller, PV-Tuning, NVIDIA QAD), scale-and-norm-only
tuning, low-rank error correction kept at inference (LoRC, LQER, QERA, CALDERA, SVDQuant-style branches), and the lever unique to an
exactly verified speculative decoder: training the drafter against the quantized target. Codebook formats and rotations have their own
chapters. Everything is argued against `00-baseline.md`: `q3` at +0.170 nats (KL 0.293, 76.2% top-1; MLP at 3.25 bpw),
`q3-compressed` at +0.275 (RTN, 2.65 bpw), `nvfp4` at +0.029, `nvfp4-compressed` at +0.028 against Swift-1.5 BF16, one RTX 5070 Ti
with 16 GB, and the eight constraints of section 6. Numbers marked *estimate* are mine; the rest are cited.

## 1. Summary

Fine-tuning-based recovery is the only family in this folder that improves quality at zero bytes and zero kernel changes: its output
is Q3/Q4/Q5 codes with binary16 scales, or NVFP4 codes with E4M3 scales, in the layouts the kernels already serve. The best-supported
published recovery at 3 bits is block-wise reconstruction plus end-to-end scale tuning: EfficientQAT takes Llama-2-70B W3g128 to 3.61
against 3.32 FP16 perplexity (0.084 nats) and W2g64 to 4.52 (0.31 nats) in 40.9 hours on one A100 [1]; GSQ's learned grid
assignments land within 1.7 zero-shot points of the vector-quantized frontier at 2.13 bits on Llama-3.1-70B, far ahead of GPTQ and
EfficientQAT [4]; PV-Tuning reaches 3.78 against 3.32 at 2 bits but needs a codebook format [3]. The cheapest tier that plausibly pays
for `q3` runs on the 16 GB card: one decoder block at a time against cached BF16 activations (6 GB of optimizer state per block, 11 to
16 minutes per block, 15 to 20 hours for 65 blocks at EfficientQAT's data budget, *estimate*), and I expect it to roughly halve the
`q3` gap, from +0.17 to about +0.08 to +0.11 nats (*estimate*). The same training restricted to codes -3..3 is the route to a viable
2.5 to 3.0 bpw form: it would lift the lossless P2E2 form (2.65 bpw) from RTN quality to about today's `q3` quality or better
(*estimate*), saving 1.56 GiB, about 90K tokens of KV. A uniform 2-bit MLP is not viable: every published scalar W2 result on a
heavily trained model loses 0.3 to 0.8 nats even after QAT. The expensive tier (end-to-end QAD over 1 to 10 B tokens on 8xH100 or
B200) costs about $400 to $6,000 at 2026 on-demand prices and buys perhaps another 0.01 to 0.03 nats (*estimate*). Low-rank branches
add 0.06 to 0.26 bpw-equivalent and a row-exact fused epilogue; they help most at 2 bits and are dominated at 3 bits by training the
400 M scales the format already carries. Swift 1.5 ships the base model's MTP head unchanged [23], so fine-tuning the 0.42 B MTP layer
on the quantized target's own greedy generations, loop-unrolled as HASS and EAGLE-3 do, is a cheap local experiment that changes
speed and never output: HASS added 8.5% throughput to DeepSeek-R1's official MTP with two epochs of off-distribution data [19].

## 2. Mechanism

### 2.1 The family

| Method | Trainable set | Objective | Data and compute | Source |
|---|---|---|---|---|
| OmniQuant (2023) | per-block learnable clipping (LWC) and equivalent transforms (LET), weights fixed | block output MSE against the FP block | 128 x 2048 tokens; 20 epochs, 40 with `--aug_loss` at W2g64; one GPU, block by block | [7] |
| EfficientQAT Block-AP (2024) | all weights of one block (FP32 masters) plus step sizes and zero points | MSE of the quantized block's output (on the quantized model's inputs) against the FP block's output (on FP inputs) | 4096 x 2048 RedPajama, 2 epochs, batch 2, `quant_lr` 1e-4, `weight_lr` 2e-5 (W2) or 1e-5 (W3/W4); one GPU, caches on disk | [1] |
| EfficientQAT E2E-QP | step sizes only (`module.scales.requires_grad = True`) | next-token cross-entropy | 4096 x 4096, 1 epoch, batch 32, lr 2e-5 / 1e-5; whole model resident | [1] |
| GSQ (2026) | per-weight logits over a 5-level window around a GPTQ init plus per-group scales, Gumbel-softmax relaxation, Lion | layer output reconstruction | 4096 x 4096 (FineWeb-Edu; OpenThoughts reasoning traces for Kimi), 10 epochs per layer; layer by layer with offloading, 1xH100 (8B) to 4xH100 (70B) | [4] |
| ParetoQ (Meta 2025) | all weights, STE; learnable range; SEQ grid at 1.58/2 bits, grid with zero at 3/4 bits | next-token loss | about 10% of the pretraining budget; 3/4-bit "compensation" saturates fast, 2-bit and below needs "reconstruction" | [2] |
| LLM-QAT (2023) | all weights | logit distillation from the FP model | 100K self-generated samples, data-free | [6] |
| BitDistiller (2024) | all weights after asymmetric clipping | confidence-aware KL (CAKLD) to the FP teacher | 8,000 teacher-generated samples of 1024 tokens (3000 WikiText + 5000 Alpaca), lr 8e-6, 2 to 4 epochs, 8 GPUs | [5] |
| PV-Tuning (2024) | codes, codebooks, scales, non-quantized parameters | cross-entropy with teacher logits | RedPajama, batch 128 x 4096, 1 epoch, lr 3e-4, `code_lr` 1e-2, beam 5; multi-GPU | [3] |
| QuEST (2025) | all weights and activations, from scratch | pretraining loss, Hadamard-normalized MSE-optimal quantizer, trust gradient estimator | pretraining data; not a recovery method | [8] |
| Norm Tweaking (2023) | LayerNorm parameters only, per layer | channel-wise distribution loss against the float layer | self-generated text; one GPU | [9] |
| NVIDIA QAD (2026) | all weights, or LSQ `amax` scales only, or LoRA on a frozen quantized base | KL to the BF16 teacher's top-k logits | 20K x 8192 (HF recipe) to 8.4 B tokens (Megatron); 1 node to 128 GPUs | [17] |

Two facts organize the family. ParetoQ's learning transition: at 3 and 4 bits a fine-tuned model stays close to its pretrained
distribution and saturates with few tokens, while 2 bits and below must rebuild representations [2]; `q3` is in the cheap regime, a
uniform 2-bit form is not. And every block-wise method keeps one block on the GPU, so model size enters as disk and time, not VRAM.

### 2.2 Block-wise reconstruction on one 16 GB card

EfficientQAT's loop (`quantize/block_ap.py`): cache block l's inputs for the FP model and for the quantized model, run the FP block over
the FP cache to make targets, replace the block's linears with fake-quant linears, train the block in FP32 under autocast with AdamW
for 2 epochs at batch 2, then run both the trained quantized block and the FP block forward to produce block l+1's caches
(`--off_load_to_disk` keeps them on disk). For this model (shapes from `00-baseline.md` section 1):

| Item | GDN block | Attention block | MTP block |
|---|---|---|---|
| Parameters | 382.7 M (gate_up 178.3, down 89.1, value_z 62.9, output 31.5, query_key 21.0) | 372.2 M | 424.7 M |
| FP32 master + grad + Adam (16 B/param) | 6.1 GB | 6.0 GB | 6.8 GB |
| BF16 weights and grads with 8-bit Adam | 2.3 GB | 2.2 GB | 2.6 GB |
| Activations at 2 x 2048 tokens (*estimate*) | 1.5 to 3 GB (gate_up output 285 MB BF16; chunked GDN states) | same | same |
| Forward+backward FLOPs, 4096 x 2048 x 2 epochs | 3.85e16 | 3.75e16 | |
| Minutes per block at 40 to 60 TFLOPS achieved (*estimate*; peak 102) | 11 to 16 | 10 to 16 | |
| Total for 65 blocks (*estimate*) | 12 to 17 h compute, 15 to 20 h with cache I/O; about 5 h at 1024 samples | | |

Everything fits in the 14.6 GiB the desktop leaves. The caches are the constraint: 4096 x 2048 x 5120 BF16 is 85.9 GB per cache, 172 GB
for two, read twice per block (about 500 GB of NVMe traffic per block, 3 minutes at 3 GB/s); 43 GB at 1024 samples. Host RAM already
swaps at `--jobs 16`, so NVMe it is. The BF16 teacher never sits on the GPU: each block's weights (at most 766 MB) are read once from
the 54.7 GB checkpoint. EfficientQAT measured Block-AP on Llama-2-70B at 26.6 h and 29.9 GB on one A100-80GB [1]; scaled by parameters
a 27B model is about 10 A100-hours, 3 to 6 hours on an H100 (*estimate*). The GDN block needs a PyTorch backward (the chunked gated
delta rule of `flash-linear-attention`, which Hugging Face's `Qwen3_5` modeling code uses).

GSQ is the same loop with a per-weight logit vector over five integer levels around the GPTQ init and a learned scale per group, 10
epochs over 4096 x 4096 tokens per layer [4]: 3.9e17 FLOPs per block here, 2.5e19 for the model, about 17 hours on one H100 or a week
on the 5070 Ti (*estimate*; the repository notes about 12 hours on 4 GPUs for the 40 layers of Qwen3.5-35B-A3B). It is the method
behind ISTA's Qwen3.8-27B GSQ-RCO GGUFs at 2.50 to 3.50 bpw and it supports the Qwen3.5/3.6 hybrid wrappers.

### 2.3 End to end: scales and norms, all parameters, the objective

**Scales and norms only.** The Q3/Q4/Q5 formats carry one binary16 scale per 64 weights: 400 M scales over the 25.6 B streamed
parameters (267 M in the MLP), plus about 0.66 M RMSNorm weights. With codes frozen the dequantized weight is `code x s`, so the gradient
to `s` is exact and only the binary16 rounding of `s` needs a straight-through estimator; EfficientQAT's E2E-QP trains step sizes this
way and NVIDIA ships `qad_scale_only.yaml`, LSQ `amax` scales at lr 1e-4 against the BF16 teacher [17]. The engine's own history says
scales matter at 3 bits: the 39-candidate weighted scale search took RTN Q3 from +0.275 to +0.170 with no other change. End-to-end
scale training on the 16 GB card is marginal: packed codes (about 10 GiB), a BF16 scale copy and gradient (1.6 GB), 1.3 GB of
activation checkpoints at 2048 tokens, FP32 masters and Adam (4.8 GB) offloaded to the host (*estimate*), about 19 hours of compute
for 16.8 M tokens. Block-wise scale training (6.25 M scales per block) or one rented H100 (6 x 27 B x 16.8 M = 2.7e18 FLOPs, about 2
hours, $5 to $8, *estimate*) is simpler.

**All parameters.** Memory moves this to rented hardware: 27 B parameters with FP32 master and Adam is 432 GB, so full-parameter QAT
needs 8x80 GB with FSDP regardless of token count, and a resident teacher adds 54.7 GB (ModelOpt reports 124 GB per GPU for a
35B-A3B student plus teacher at 32K tokens [17]). LoRA-QAD (ModelOpt `--lora_rank`: backbone frozen, `quantize(W + (alpha/r) BA)`
per linear) or scale-only QAD fits on 1 to 2 H100s or one H200/B200.

**Objective.** Teacher logits beat hard labels at low bits: LLM-QAT found logit distillation superior to label training [6];
BitDistiller's CAKLD mixes forward and reverse KL by teacher confidence [5]; ModelOpt's QAD uses KL over the teacher's top-4096 [17];
OnlineQAT's baseline is forward KL over the top-20 plus a 0.2-weighted cross-entropy [16f]. Two 2026 results go further: on-policy
distillation (the student generates, the teacher scores) gained 4.32 points in 20 steps where 800 teacher-forced QAD steps gained 4.09
on Qwen3-4B at 2.79 bits, and a larger teacher did not help from 1.7B up [16e]; OnlineQAT initializes block-wise and then distills on
policy [16f]. With a 248,320-row vocabulary, store the teacher's top-k offline (64 entries as int32 id plus fp16 value is 384 B per
token, 3.8 GB per 10 M tokens); the repository's `bf16.kld` format (16-bit log-probabilities over the 16 nats below each maximum) is
already such a store.

### 2.4 Simulating the engine's formats in PyTorch

Round where `grouped.py` and `nvfp4.py` round; accumulation order is irrelevant for training (the kernel drift floor is KL 3.4e-3 to
4.7e-3, 30 to 85 times below the quantization gap).

- **Q3G64_F16S, Q4G64_F16S, Q5G64_F16S.** Groups of 64 along K, K zero-padded to a multiple of 128. `W_hat = clamp(round(W / s), qmin,
  qmax) x s` with (qmin, qmax) = (-4, 3), (-8, 7), (-16, 15); `s` a trainable FP32 scalar per group rounded to binary16 in the forward
  (`s16 = s + (fp16(s) - s).detach()`), initialized from the converter's search so training starts at today's artifact. Weights use the
  STE (ParetoQ, EfficientQAT), scales the LSQ gradient. For a P2E2-compatible result clamp to (-3, 3): the compressed layout cannot store
  -4. Export writes codes and scales into the row-split planes with `pack_codes_torch` and the scale writer; no round trip through BF16
  (`code x fp16 scale` has up to 14 significant bits). Mirror `recipes.py` tensor for tensor: Q4 `query_key`, Q5 `value_z` and
  `gate_value`, Q3 elsewhere including embedding and LM head.
- **NVFP4.** Per tensor `g = amax / (6 x 448)` (FP32, fixed), per 16-block an E4M3 code `s_b`, `W_hat = sign x e2m1(|W| / (s_b g)) x
  s_b g`. Either train weights only and recompute `s_b` from the current weights each step as the converter does (imatrix search once at
  export), or learn a continuous block scale snapped to E4M3 with an STE, which is ModelOpt's LSQ-NVFP4 recipe (`learnable_amax: pre,
  post`, MSE init, FP8 scale sweep) [17]. `INT_vs_FP` has a compact reference fake-quant for the E4M3 scale (`s_of_s = amax(s)/448;
  (s / s_of_s).to(float8_e4m3fn)`) to cross-check against `nvfp4.py` byte for byte [13]. For SD4 keep the 16-entry table fixed and
  re-run the 6-minute table DP at export.
- **Stochastic rounding.** In the NVFP4 pretraining recipe it applies to gradients in the backward GEMMs; weights and activations use
  round-to-nearest-even, weights carry 2D 16x16 scales, and the random Hadamard transform touches only the weight-gradient GEMM inputs
  [14]. None of that is a weight-only fine-tune. Use RTN plus STE with learnable scales; stochastic rounding of weights is an option
  against STE dead zones but must anneal back to RTN before export, because the artifact stores rounded codes.
- **Not simulated:** Hadamard-rotated NVFP4 KV, the two-term activation quantization of the prefill routes, FP32 accumulation order. The
  harness measures their effect at the 4e-3 KL level.

### 2.5 Low-rank branches: the arithmetic for this model

A branch stores `W ~ Q + B A` with `A` (r x K), `B` (N x r): LoRC takes the top-r SVD of the rounding error [12a], LQER/L2QER scale
the error by activation statistics first [12b], QERA derives the factors from the layer-output error in closed form [12c], CALDERA
alternates re-quantizing Q with refitting LR and optionally LoRA-trains the factors [12d], SVDQuant moves outliers into a rank-32
16-bit branch before 4-bit rounding [12e]. Over the 320 linears of the 64 blocks, sum(N + K) is 6.39 M:

| Rank | Parameters | BF16 bytes | bpw-equivalent over 25.6 B | FLOPs per token | Decode time at 855 GB/s | Stored as NVFP4 |
|---|---|---|---|---|---|---|
| 16 | 102 M | 195 MiB (+8 MiB LM head) | 0.064 | 0.20 GFLOP | +0.24 ms (+1.5% of the 16 ms Q3 step) | 55 MiB, +0.07 ms |
| 32 | 204 M | 390 MiB (+15 MiB) | 0.128 | 0.41 GFLOP | +0.48 ms (+3%) | 110 MiB, +0.13 ms |
| 64 | 409 M | 780 MiB (+31 MiB) | 0.256 | 0.82 GFLOP | +0.96 ms (+6%) | 220 MiB, +0.27 ms |

Bytes are the real cost at one row; FLOPs are 0.8% of the model's at rank 32. Launches are not free: two extra GEMVs per linear are 640
launches per token, 1 to 2 ms even from captured graphs, so the branch must be fused: compute `y = A x` once per distinct block input
(four per block, stacking the A's of linears that share an input) inside the kernel that produces that input, and add `B[row, :] . y`
in the GEMV epilogue (r FMAs per output row against a K of 5120 to 17408). "Q3 plus rank-32 BF16 branch" is 3.25 + 0.13 = 3.38 bpw on
the MLP, 11.1 GiB streamed instead of 10.7. The branch has half the trainable parameters of the scale plane (204 M against 400 M), and
the scale plane costs nothing at inference, so a branch is justified only by its expressivity after the scales have been trained.

### 2.6 Training the drafter

EAGLE-1 trains a one-layer head on the target's features with `1.0 x feature regression + 0.1 x token loss`, 20 epochs, lr 3e-5,
"within 1 to 2 days on 8x RTX 3090" [18a]; EAGLE-3 drops the feature loss and unrolls the head over its own outputs in training
(SpecForge: `ttt_length 7`, 10 epochs, lr 1e-4, ShareGPT) [18b][18c]; HASS harmonizes objective and context between training and
decoding [19]; Medusa-1 trains only new heads at lr 1e-3 for 2 epochs, Medusa-2 adds self-distillation [20]; DFlash trains a 5-layer
block drafter on responses regenerated by the frozen target [21]. SpecForge's DFlash2 recipe for Qwen3.8-27B is the closest datapoint
to this engine: a 1.9 B drafter reading the hidden states after layers 5, 19, 33, 47 and 61, trained on 1.07 M conversations whose
assistant turns Qwen3.8-27B regenerated itself with reasoning kept (mean 4.5K tokens, about 4.8 B tokens), hard-target cross-entropy
with a strict top-k selector objective, lr 5e-4, at 9 to 22.65 samples/s on 8x H200 (about 13 hours per epoch, computed from the
recipe); acceptance on off-policy answers "plateaus far below" the same draft on regenerated data [18c]. DeepSeek-V3 reports 85 to
90% second-token acceptance and that its module was trained for one step and reused autoregressively [22a]; GLM-5 shares the
parameters of 3 MTP layers in training to remove that discrepancy [22b].

For Euhedral the drafter is the base model's MTP block (0.42 B: `fc` 52 M, packed q/k/gate/v 73 M, `o` 31 M, gate_up 178 M, down 89 M),
NVFP4 in every artifact, applied recursively (depth 2 on Q3, 4 on NVFP4). Fine-tuning it is EAGLE-style training with three specifics:
the input hidden `h_p` comes from the quantized target (the row-exact verifier produces exactly the hidden the drafter sees in
deployment; a fake-quant replica matches up to kernel-order noise); the recursion is unrolled to depth 2 to 4 with the drafted or
gold token fed back; and the MTP weights are fake-quantized to NVFP4 during training so the exported layer is the trained one. The loss
that matches a greedy verifier is cross-entropy against the target's argmax, optionally plus a KL term for a denser signal. Cost on the
16 GB card (*estimate*): optimizer state 6.8 GB (FP32) or 2.6 GB (8-bit Adam), the frozen LM head in BF16 (2.5 GB) with chunked
logits, 7.6 GFLOP per token at depth 1 and 23 GFLOP with three unrolled drafts (the LM head dominates), so 10 M tokens take 0.5 to
1.6 hours and 100 M tokens 5 to 16 hours. Hidden states are 10 KB per token (102 GB per 10 M tokens) if dumped, or 2 x 27 B FLOPs per
token (about 4 hours per 10 M tokens) if recomputed from the resident Q3 codes. The DFlash2 drafter (1.9 B) does not fit
full-parameter training on 16 GB; LoRA on its 25 projections would, but `q3` artifacts have no DFlash2 drafter (`docs/DFLASH2.md`
section 9), so the MTP layer is the local target.

## 3. Published results

Perplexities are WikiText-2; "nats" is `ln(ppl_quant / ppl_fp16)`, the repository's dNLL.

**Uniform scalar reconstruction and QAT** (EfficientQAT model zoo [1]):

| Model | FP16 | W3g128 | W2g64 | Nats at W3g128 / W2g64 | Cost (paper) |
|---|---|---|---|---|---|
| Llama-2-7B | 5.47 | 5.81 | 6.86 | 0.060 / 0.226 | 4.8 h, one A100-80GB |
| Llama-2-13B | 4.88 | 5.12 | 5.96 | 0.048 / 0.200 | |
| Llama-2-70B | 3.32 | 3.61 | 4.52 | 0.084 / 0.309 | 40.9 h on one A100-80GB: Block-AP 26.6 h / 29.9 GB, E2E-QP 14.3 h / 34.2 GB at 2 bits |
| Llama-3-8B | 6.14 | 7.09 | 9.41 | 0.144 / 0.427 | |
| Llama-3-70B | 2.85 | 4.19 | 6.08 | 0.385 / 0.758 (W4g128: 3.17, 0.106) | |

Llama-3 loses 2 to 4 times more than Llama-2 at the same width, as "Low-Bit Quantization Favors Undertrained LLMs" predicts [11].
Qwen3.8-27B is in that class, and the engine's `q3` (0.171 nats at an average 3.59 bpw over the streamed tensors) already sits between
EfficientQAT's Llama-3-8B and Llama-3-70B W3g128 results without any error feedback. OmniQuant's tables were not retrievable this
session beyond one snippet (LLaMA-13B W2A16 13.21 against GPTQ 3832; W2A16g64 7.34 against GPTQ 10.06, column unverified) [7];
BitDistiller's per-model tables and GPU-hours were not retrievable (its abstract claims +3.54 points over LLM-QAT and +12.43 over the
best PTQ at 2 bits) [5].

**Codebook fine-tuning, learned assignments and low-rank decomposition at about 2 bits** ([3], [4], [12d]):

| Model | Method | bpw | Result | Baselines | Cost |
|---|---|---|---|---|---|
| Llama-2-7B / 13B / 70B | AQLM + PV-Tuning 1x16g8 | about 2 | ppl 5.68 / 5.05 / 3.78 (0.038 / 0.034 / 0.130 nats) | FP16 5.47 / 4.88 / 3.32 | PV-Tuning hours not retrieved; AQLM calibration about 1 day per 7B, 10 to 14 days per 70B on one A100 |
| Llama-3.1-8B-Instruct | GSQ | 2.13 | zero-shot avg 68.55 | FP16 73.71; GPTQ 37.53; QuIP 39.20; EfficientQAT 63.79; QTIP 69.88; PV-Tuning 69.83 | 1xH100 |
| Llama-3.1-70B-Instruct | GSQ | 2.13 | 75.57 | FP16 78.99; GPTQ 57.38; EfficientQAT 71.43; QTIP 77.25; PV-Tuning 76.27 | 4xH100 |
| Qwen3-8B Q2_K GGUF | GSQ in-format refinement | Q2_K | avg 56.28 | Unsloth init 50.03 | |
| Llama-2-70B | CALDERA rank 128 / 256, 4-bit factors | 2.1 / 2.2 | WikiText-2 4.11 / 3.98 (0.213 / 0.181 nats); C4 5.95 / 5.76 | QuIP# 2-bit without fine-tuning 5.37 / 7.51 (earlier paper version) | |

GSQ's two rows calibrate expectations: a learned scalar assignment closes most of the distance from GPTQ and EfficientQAT to the VQ
frontier, and it produced the public Qwen3.8-27B GGUFs (IQ2_XS 2.50 bpw 8.4 GB, IQ2_S 2.75, IQ3_XXS 3.00, IQ3_S 3.50 "recommended"; a
third-party summary reports IQ3_S matching the base on AIME25 and LiveCodeBench v6, 0.51 behind on GPQA-Diamond) [4][24]. Those files
use llama.cpp codebook types (other chapter); the learned-rounding part applies to uniform grids.

**Industrial QAD on a Qwen hybrid-attention MoE** (ModelOpt, Qwen3.6-35B-A3B, W4A4 NVFP4 with FP8 KV) [17]:

| Model | MMMU-Pro | GPQA-D | SciCode | AA-LCR | IFBench | tau2 | Avg | Cost |
|---|---|---|---|---|---|---|---|---|
| BF16 teacher | 74.6 | 84.7 | 39.9 | 69.1 | 60.0 | 94.2 | 70.4 | |
| W4A4 NVFP4 PTQ | 73.4 | 84.7 | 39.1 | 70.0 | 57.9 | 94.2 | 69.9 | 1.2 GPU-h |
| + QAD 500 iterations (8.4 B tokens, KL top-4096, lr 1e-5) | 73.9 | 84.2 | 40.2 | 69.4 | 59.6 | 93.4 | 70.1 | 735 GB200 GPU-h, 5.7 h on 128 GPUs |

Only IFBench (-2.6 to -0.3 pp) and MMMU-Pro (-1.2 to -0.7) had real losses. The side effect matters for Swift 1.5: QAD raised SciCode
output length 90.9% (non-terminating `<think>` loops from 0.7% to 3.6% of sub-steps) at unchanged score, 4-bit weights alone lengthened
outputs 23 to 25%, and a decode-time `presence_penalty` 1.5 removed it [17].

**Reasoning models and data** ([16a] to [16f]):

| Finding | Setting | Source |
|---|---|---|
| Self-generated reasoning traces (NuminaMath-1.5, 128 x 2048) as the GPTQ calibration set, because of the domain gap to WikiText2 | R1-Distill 1.5B to 70B, QwQ-32B | [16a] |
| Own traces beat DeepSeek-R1-671B's traces as calibration data by +25.54 pp; C4/WikiText2 calibration collapsed | Qwen3-4B, 1.58-bit | [16b] |
| Reasoning calibration: up to +38.45 pp AIME25 pass@1, largest at small scale and low bits | Qwen3-4B W3G128 GPTQ | [16c] |
| PTQ lowers accuracy and lengthens chains of thought; in up to 52% of failures the right answer appears mid-trace; a logit penalty on 50 hesitation markers shortens CoT 12 to 23% | 1.5B to 32B, 3 PTQ methods | [16d] |
| On-policy distillation: 20 steps gain 4.32 points against 4.09 from 800 teacher-forced QAD steps; larger teacher does not help from 1.7B up | Qwen3-4B, 2.79 bits | [16e] |
| Block-wise QAT init then on-policy distillation: Qwen3-1.7B 57.28 at W3A16, 32.52 at W2A16 (+2.90 / +0.44 over ReasoningQAT) | Qwen3-1.7B | [16f] |

**Drafter training** ([18] to [22]): EAGLE-1 heads train in 1 to 2 days on 8x RTX 3090 [18a]; EAGLE-3's original SFT is 20 epochs and
roughly 80 hours, or 8 hours on 8x H800 for 2 epochs of 8x-augmented UltraChat-200K [18d]; about 48 H200-hours per 8 to 9 B draft on
1.4 M regenerated samples [18e]; HASS continual training of DeepSeek-R1's official MTP for 2 epochs of ShareGPT lifted mean throughput
from 62.46 to 67.75 tok/s at T=0 (1.76x to 1.91x, +8.47%) with data "extremely less than the official MTP" and off-distribution, and
reaches acceptance length 4.06 on R1-Distill-Qwen-32B [19]; SpecForge's DFlash2 for Qwen3.8-27B used 4.8 B self-regenerated tokens on
8x H200 [18c]; DeepSeek-V3's MTP accepts its second token 85 to 90% of the time [22a].

**Hybrid SSM/attention models.** None of EfficientQAT, PV-Tuning, ParetoQ, BitDistiller, OmniQuant or QuEST reports one. The evidence
is GSQ's Qwen3.5/3.6 wrappers and the Qwen3.8-27B GGUFs [4]; "Why Gated DeltaNet Survives 4-Bit Quantization", where calibration-only
PTQ reaches BF16-level task accuracy for NVFP4 W4A4 on the recurrent half of a 27 B hybrid and a concurrent QUASAR checkpoint
quantizes all 496 projections by QAT [15]; and NVIDIA's NVFP4 pretraining of a 12 B hybrid Mamba-Transformer over 10 T tokens [14]. No
paper reports W2/W3 QAT recovery for a GDN hybrid.

## 4. Fit with Euhedral

### 4.1 The eight constraints

| Constraint | Fine-tuned codes and scales | Low-rank branch | Fine-tuned MTP |
|---|---|---|---|
| 1. Row-exact 1 to 8 row decode | Unchanged: same formats, kernels and twins. | `B[row,:] . y` in the epilogue is per row and deterministic; `y = A x` does not depend on the row count. Row-exact by construction; needs twin tests, no new proof. | Not involved: the verifier never reads the drafter's arithmetic. |
| 2. Prefill off the BF16 path | Unchanged. | A BF16 GEMM pair of 2 x T x r x (N + K) FLOPs, 0.8% of the main GEMM at r = 32, plus an add over the output tile: about 2% of prefill (*estimate*). | Unchanged. |
| 3. Decode instruction budget | Unchanged. | +r FMAs per output row (0.2 to 0.6%); `A x` is r x K per block input. | Unchanged. |
| 4. Determinism | Weights are data. | Deterministic kernels. | Deterministic. |
| 5. Offline conversion only | An offline step that emits codes and scales; the converter needs a `--codes-from` entry that writes pre-rounded codes into the existing planes (new recipe, same ordinals). | New EDRL object class and layout ordinal; old artifacts keep loading. | Replaces the `mtp/*` objects only. |
| 6. The artifact decides | A fine-tuned `q3` is a recipe (`q3-ft`) the engine cannot tell from `q3`. | `ArtifactProfile` must detect branch objects and pick the fused kernels. | `speculativeDepth` was screened for the shipped head; a better head may move the best depth and needs re-screening. |
| 7. Memory is the product | Zero bytes. Restricted to codes -3..3 it gives the 2.65-bpw P2E2 form calibrated quality: 1.56 GiB less than `q3`, about 90K tokens of KV at 18 KiB. | +195 to +780 MiB resident; on the NVFP4 forms that pushes weights to the host at 2 to 2.5% decode per 128 MiB (net negative); `q3` has the room (2.4 GiB free at 48K). | Zero bytes. |
| 8. Drafting precision is separate | n/a | n/a | Exactly this knob: speed, never correctness. |

### 4.2 Which forms benefit, expected quality

At 3 bits the model is in the compensation regime, EfficientQAT's W3 results stay within 0.06 to 0.14 nats on Llama-2 and Llama-3-8B,
GSQ recovers most of the GPTQ-to-VQ gap, and the engine's rounding has no error feedback today, so there is room. The figures below
are estimates in the repository's metric (dNLL against the BF16 of the same checkpoint); on the current document only the `q3` rows
are resolvable (s.e. 0.021), hence the holdout of section 4.5.

| Form | Today | Tier 0: scales + norms (16 GB) | Tier 1: block-wise all parameters (16 GB or 1xH100) | Tier 2: end-to-end QAD, 0.1 to 1 B tokens (rented) | Confidence |
|---|---|---|---|---|---|
| `q3` (3.25 bpw MLP) | +0.170, KL 0.293 | +0.13 to +0.15 | +0.08 to +0.11, KL about 0.15 to 0.18 | +0.07 to +0.09 | medium: W3 literature is consistent |
| `q3-compressed` (2.65 bpw, codes -3..3) | +0.275 (RTN) | +0.19 to +0.22 | +0.10 to +0.15 | +0.09 to +0.13 | medium: 7 levels is a mild restriction of 8 |
| Uniform 2.5 bpw (Q2 MLP) | not built | | +0.4 to +0.6 | +0.3 to +0.5 | high that it is not viable: every scalar W2 result on a Llama-3-class model loses 0.3 to 0.8 nats |
| `nvfp4`, `nvfp4-compressed` | +0.029 / +0.028 | within noise | +0.015 to +0.02 | +0.01 to +0.02 | low: below the harness resolution; ModelOpt's W4A4 QAD recovered 1 to 2 pp on two of six benchmarks |
| `q3` + rank-32 BF16 branch (3.38 bpw), factors trained | | | Tier 1, perhaps 0.01 to 0.02 better | | low: CALDERA's gains are at 2 bits; the trained scale plane already has more parameters |

Verdict on "Q3 plus rank-32 branch against NVFP4": not competitive. Nothing published suggests a branch takes a 3.25-bpw backbone from
+0.17 to +0.03; low-rank gains are largest on very coarse backbones (CALDERA at 2.1 bpw: 0.21 nats against about 0.48 for QuIP# 2-bit
without fine-tuning), and at W3 LQER needed rank about 600 for plain SVD-of-error on OPT-1.3B [12b]. The byte-efficient levers at 3.4
bpw are the trained scale plane (free) and, for context, the P2E2 form.

Which checkpoint: the deployed artifact is `nvfp4-compressed` from Swift-1.5 with UkisAI's `imatrix-swift15-v1mix.gguf` (a
reasoning, multilingual and code mix); `q3` and `nvfp4` are still base-model conversions with ISTA's matrix. Fine-tune the checkpoint
that will be served, with its BF16 as teacher and reference (Swift-1.5 BF16: perplexity 8.51, NLL 2.141 on the quality document).
Converting Swift-1.5 to `q3` with the v1mix matrix, with no training, is the baseline every experiment compares against.

### 4.3 Cost table

Assumptions (*estimates*): 40 to 60 TFLOPS achieved on the 5070 Ti, 400 on an H100, 900 on a B200; 2026 on-demand prices: H100 $2.49
to $3.99 per GPU-hour (RunPod community $2.69, secure $3.29 to $3.49; Lambda $2.49 to $3.99), B200 $6.69 to $6.79 [25]; spot tiers are
cheaper but interruptible. Teacher online adds about a third to QAD compute.

| Tier | Trained | Hardware | Time | Dollars | Data |
|---|---|---|---|---|---|
| 0a | scales + norms, block-wise | 5070 Ti, caches on NVMe | 5 h (1024 x 2048) to 15 to 20 h (4096 x 2048) | 0 | 2 to 8 M tokens |
| 0b | scales only, end to end, KL to teacher | 5070 Ti with host-offloaded optimizer (marginal) or 1xH100 | 19 h / 2 h | 0 / $5 to $8 | 16.8 M tokens; teacher top-k logits (6 to 7 h on the 5070 Ti by layer streaming, or llama.cpp on the CPU) |
| 1 | all parameters per block (Block-AP) | 5070 Ti / 1xH100 | 15 to 20 h / 3 to 6 h | 0 / $10 to $25 | 8.4 M tokens; 172 GB caches (43 GB at 1024 samples) |
| 1b | GSQ-style learned assignments | 1xH100 / 5070 Ti | about 17 h / about a week | $45 to $70 / 0 | 16.8 M tokens per layer |
| 2a | QAD, 100 M tokens, full parameters or LoRA | 8xH100 (memory-bound for full parameters) | about 2 h | $50 to $70 | half own generations |
| 2b | QAD, 1 B tokens | 8xH100 / 8xB200 | 15 to 20 h / 6 to 8 h | $400 to $650 / $340 to $450 | |
| 2c | QAD, 10 B tokens (ParetoQ scale) | 8xH100 / 8xB200 | 150 to 190 h / 65 to 85 h | $3,000 to $6,000 / $3,500 to $4,600 | reference: ModelOpt's 8.4 B-token W4A4 QAD took 735 GB200 GPU-h |
| 3 | MTP block, unrolled, NVFP4 fake-quant | 5070 Ti | 0.5 to 1.6 h per 10 M tokens; hidden states 4 h per 10 M tokens or dumped during generation | 0 | 5 to 50 M tokens of own greedy generations, thinking on |

### 4.4 Data: which text, how many tokens, own generations or not

- **Budgets in the literature:** Block-AP 8.4 M tokens, E2E-QP 16.8 M, GSQ 16.8 M per layer, OmniQuant 0.26 M over 20 to 40 epochs,
  BitDistiller about 8 M (teacher-generated), LLM-QAT 100K self-generated samples, ModelOpt HF QAD 20K x 8192 (up to 164 M), ModelOpt
  Megatron QAD 8.4 B for W4A4, ParetoQ about 10% of pretraining. For weight-only 3-bit recovery, 10 to 50 M tokens is the regime of
  diminishing returns; 16 to 32 M is the budget here.
- **Own generations, with thinking.** PTQ (Liu et al., AYOT, LAQuant), QAT (BitDistiller, LLM-QAT, OnlineQAT, on-policy distillation)
  and drafters (SpecForge, DFlash) agree: the model's own outputs are the right calibration and distillation data [16][5][6][18c][21].
  For Swift 1.5 that means Swift's generations, not the base's: the base thinks about twice as long (UkisAI reports a 41.9% mean and
  58.5% median token reduction on GPQA-Diamond [23]), and distilling toward base traces trains the quantized model back toward the
  behaviour the fine-tune removed. Swift BF16 generation on this machine means CPU llama.cpp, so generate with the deployed
  `nvfp4-compressed` artifact at 100 to 146 tok/s (10 M tokens is 19 to 28 single-stream hours; prompts from `tools/mtp_draft_prompts.py`
  and public prompt sets) and score the sequences teacher-forced with Swift BF16 top-k logits: the on-policy recipe of [16e][16f], with
  a small artifact-to-BF16 path mismatch (top-1 agreement 89.4%).
- **Mix and length.** Half own generations (chat, code, agentic with tool results, math; the sets of `MtpDraftConfidenceCudaIntegrationTest`),
  half general text of the v1mix kind (reasoning, multilingual, code), to keep the broad distribution and expose overfitting to
  self-generated style. 2048 to 4096 tokens per sequence; longer buys nothing for quantization and costs activation memory per block.
- The imatrix pipeline is already Swift-specific (`llama-imatrix` on a BF16 GGUF of Swift-1.5, UkisAI's v1mix); seed Tier 0/1 with the
  same text so calibration and fine-tune agree on what matters.

### 4.5 Evaluation holdout

One 1279-token document gives s.e. 0.021 nats on `q3` (per-token s.d. of the paired dNLL about 0.75 nats) and 0.008 on `nvfp4` (s.d.
0.29). An s.e. of 0.005, enough to resolve 0.01 nats at two standard errors, needs about 22,500 independent tokens for Q3-class and
3,300 for NVFP4-class artifacts; tokens within a document are correlated, so use 8 to 16 held-out documents of 2K to 4K tokens (32K
to 64K tokens) and a document-level standard error, with KL and top-1 as `compare_reference.py` reports them. Hold out two kinds: own
generations never used in training and external text (news, multilingual prose, code from other repositories); improving the first
while worsening the second is overfitting. Exclude the imatrix and training text. Add the metric Swift 1.5 is sold on: mean thinking
tokens and accuracy on 100 to 200 fixed reasoning prompts under greedy decoding, for BF16, the artifact and the fine-tuned artifact
(quantization inflates chain-of-thought length [16d]; ModelOpt's QAD doubled one benchmark's output length at unchanged score [17]).

## 5. Experiment plan

Ordered by information per hour; each step is one artifact through the existing harness (`TeacherForcedQualityCudaIntegrationTest`,
`compare_reference.py`, the acceptance pass, the speed benchmarks), on Swift-1.5 against its BF16.

1. **Baselines and holdout (1 day, no training).** Build the holdout of 4.5 and its Swift-1.5 BF16 reference (`llama-perplexity
   --kl-divergence-base`). Convert Swift-1.5 to `q3` and `q3-compressed` with the v1mix matrix. Record dNLL, KL, top-1 and thinking
   length for all forms. Go: holdout s.e. at or below 0.005 nats for `q3`.
2. **Fake-quant parity (1 to 2 days).** A PyTorch module reproducing `grouped.py` and `nvfp4.py` byte for byte on the `q3` and
   `nvfp4-compressed` recipes; export, load, compare teacher-forced reports. Go: KL between fake-quant model and artifact below 1e-2.
3. **Tier 0a on two blocks (hours).** Scale-and-norm training of block 10 (GDN) and 11 (attention) on 1024 x 2048 own-generation
   tokens, 2 epochs. Go: held-out block-output MSE down at least 30%; less means the scales are already near optimal and Tier 1 is the
   first real step.
4. **Tier 0a then Tier 1 on all blocks (1 to 2 days each on the 5070 Ti).** Produce `q3-ft-scales` and `q3-ft-blocks`. Go for the new
   `q3` recipe: dNLL better by at least 0.03 nats, thinking length within 10%, no external-text regression. The literature expectation
   is 0.06 to 0.09; below 0.02 the method does not pay on this model and this chapter's expectation was wrong.
5. **P2E2-compatible Tier 1 (same cost).** Codes clamped to -3..3, transcoded to P2E2; measure size and quality. Go: `q3-compressed` at
   or below today's `q3` dNLL (+0.170) with the P2E2 ratio at or below 0.85 of row-split bytes. This is the result that changes the
   product.
6. **Drafter fine-tune (1 to 2 days).** Dump post-final-norm hiddens and greedy tokens from the engine on 5 to 20 M tokens of own
   generations. Fine-tune the MTP block (NVFP4 fake-quant, unrolled to depth 4, cross-entropy on the target argmax plus KL), export,
   re-screen depth with `MtpDraftLengthScreenCudaIntegrationTest`. Measure tokens per verification on the chat, code and agentic sets of
   `MtpDraftConfidenceCudaIntegrationTest` (today 3.11 / 3.81 / 3.82 at MTP4) and per-position conditional acceptance (0.85 at d1 falling
   to 0.70 to 0.78 on chat). Go: +0.2 tokens per verification at the shipped depth, or a deeper depth now winning at 32K; output hashes
   equal greedy on every prompt (they must; a mismatch means the artifact writer is wrong, not the drafter). Repeat with the Q3 target's
   hiddens to learn whether each form needs its own head.
7. **Rented Tier 2 only if step 4 saturates.** One H100 for end-to-end scale tuning over the Tier 1 artifact (2 h, under $10); if that
   beats the holdout s.e., 8xH100 QAD at 0.1 to 1 B tokens ($50 to $650). Go: more than 0.02 nats over Tier 1 with the thinking-length
   guard intact.
8. **Low-rank branch last.** Train rank-32 factors on the frozen Tier 1 backbone in fake-quant; build kernels only if the gain exceeds
   0.03 nats, which is what 390 MiB buys as KV (about 22K tokens).

## 6. Risks and open questions

- **Eroding Swift 1.5's behaviour.** Its property is fewer thinking tokens; quantization lengthens reasoning [16d] and ModelOpt's QAD
  doubled output length on SciCode through non-terminating `<think>` loops at unchanged score [17]. Mitigations: Swift's own generations
  as data, the thinking-length guard in every gate, on-policy sequences from the deployed artifact, and the low learning rates of the
  recipes (1e-5 for weights, 1e-4 for scales: "adapt weights to quantization, not learn the task" [17]).
- **Overfitting the calibration text.** 8 to 17 M tokens through 65 blocks of 0.4 B parameters is ample capacity to fit the cache;
  EfficientQAT uses a 64-sample validation split and early stopping per block; the two-kind holdout is the external check. Scale-only
  training (400 M multiplicative parameters) is the safer first step.
- **The P2E2 ratio may move.** 0.815 depends on most codes being -1, 0, +1; a fine-tune that spreads codes grows the file, one that
  sharpens them shrinks it. Measure the histogram after step 5 before claiming the savings.
- **Base against Swift.** `q3` is a base-model artifact today, so the first Swift `q3` number (step 1) may differ from +0.170, and a
  trained MTP head belongs to one target.
- **The MTP head is the base's** (official card: "the published weights include the base model's MTP head"; third-party cards say UkisAI
  did not fine-tune it [23]). Open: how much of the 0.70 to 0.78 conditional acceptance beyond d1 is recursion mismatch (fixed by
  unrolled training), Swift-versus-base drift (fixed by own data) or drafter capacity (not fixable without a larger draft). The
  repository's finding that the head is calibrated at d1 and overconfident by 0.05 to 0.10 from d2 on points at the first.
- **Hybrid evidence is thin.** No W2/W3 QAT paper covers a GDN hybrid; the 48 GDN blocks hold 22% of `q3`'s bytes at 4.25 to 5.25 bpw,
  and whether block-wise training lets them drop a bit width is untested [15][4].
- **Unretrieved numbers.** OmniQuant's and BitDistiller's per-model tables, PV-Tuning's GPU-hours, Medusa's training time, the QUASAR
  checkpoint and Qwen3-Next's MTP training notes could not be fetched this session (search budget); no estimate above depends on them.
- **Teacher logits on this machine.** The 54.7 GB teacher runs on the CPU (llama.cpp; throughput unmeasured here) or by layer streaming
  over PCIe in PyTorch (about 6 to 7 hours per 16.8 M tokens, *estimate*); a one-time cost per data set.
- **Export path.** The converter has no entry for pre-rounded codes; `pack_codes_torch` and the scale writer exist, but a `--codes-from`
  recipe, manifest fields for the training data, and a byte-equality test against the fake-quant model are new work.

## 7. References

1. Chen et al., *EfficientQAT*, ACL 2025, arXiv:2407.11062; https://github.com/OpenGVLab/EfficientQAT (model zoo, `examples/`,
   `quantize/block_ap.py`, `main_e2e_qp.py`; Table 7 times as quoted in search results).
2. Liu et al. (Meta), *ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization*, arXiv:2502.02631;
   https://github.com/facebookresearch/ParetoQ; summary https://graphcore-research.github.io/paretoq/.
3. Malinovskii et al., *PV-Tuning*, NeurIPS 2024, arXiv:2405.14852; Egiazarian et al., *AQLM*, arXiv:2401.06118;
   https://github.com/Vahe1994/AQLM (PV-tuned model table, `finetune.py` recipe, quantization-time notes).
4. Dadgarnia et al., *GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling*, arXiv:2604.18556;
   https://github.com/IST-DASLab/GSQ; RCO arXiv:2605.00649 (ID from a search snippet, unverified);
   https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF.
5. Du et al., *BitDistiller*, ACL 2024, arXiv:2402.10631; https://github.com/DD-DuDa/BitDistiller (`train/train.sh`, `data/generation`).
6. Liu et al., *LLM-QAT: Data-Free Quantization Aware Training*, ACL Findings 2024, arXiv:2305.17888.
7. Shao et al., *OmniQuant*, ICLR 2024, arXiv:2308.13137; https://github.com/OpenGVLab/OmniQuant (`scripts/Llama-2/*/w2a16g64.sh`).
8. Panferov et al., *QuEST*, ICML 2025, arXiv:2502.05003; https://github.com/IST-DASLab/QuEST.
9. Li et al., *Norm Tweaking*, AAAI 2024, arXiv:2309.02784.
10. Chen et al., *Scaling Law for Quantization-Aware Training*, arXiv:2505.14302; Dremov et al., *Compute-Optimal QAT*, arXiv:2509.22935.
11. *Low-Bit Quantization Favors Undertrained LLMs*, arXiv:2411.17691 (title and ID from search results; contents not read).
12. (a) Yao et al., *ZeroQuant-V2* (LoRC), arXiv:2303.08302; (b) Zhang et al., *LQER*, ICML 2024, arXiv:2402.02446; (c) *QERA*,
    arXiv:2410.06040; (d) Saha et al., *CALDERA*, NeurIPS 2024, arXiv:2405.18886, https://github.com/pilancilab/caldera; (e) Li et al.,
    *SVDQuant*, ICLR 2025, arXiv:2411.05007.
13. Chen et al., *INT v.s. FP*, ICML 2026, arXiv:2510.25602; https://github.com/ChenMnZ/INT_vs_FP (`quant/quant_func.py`).
14. NVIDIA, *Pretraining Large Language Models with NVFP4*, arXiv:2509.25149; *Dissecting Outlier Dynamics in LLM NVFP4 Pretraining*,
    arXiv:2602.02047.
15. *Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM*, arXiv:2609.04098
    (abstract-level findings from search results).
16. (a) Liu et al., *Quantization Hurts Reasoning?*, COLM 2025, arXiv:2504.04823, https://github.com/ruikangliu/Quantized-Reasoning-Models;
    (b) *Attend to Your Own Thoughts*, arXiv:2608.01078; (c) *LAQuant*, arXiv:2605.08755; (d) Lotfi et al., *Quantized Reasoning Models
    Think They Need to Think Longer, but They Do Not*, arXiv:2606.00206; (e) *Train Where the Quantized Model Goes: On-Policy Distillation
    for Low-Bit Reasoning*, arXiv:2609.26708; (f) *OnlineQAT*, arXiv:2610.09346. Also arXiv:2608.03796 (offline top-k KD) and
    arXiv:2506.09104 (block-wise PTQ plus distillation QAT toward 2-bit instruction-tuned LLMs).
17. NVIDIA TensorRT Model Optimizer, https://github.com/NVIDIA/TensorRT-Model-Optimizer: `examples/llm_qat/README.md`,
    `configs/train/{qat_nvfp4,qad_nvfp4,qad_scale_only,qad_with_learnt_amax}.yaml`, `modelopt_recipes/general/qad/*.yaml`,
    `examples/megatron_bridge/README.md`, `examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/README.md`.
18. (a) Li et al., *EAGLE*, arXiv:2401.15077 and *EAGLE-2*, arXiv:2406.16858, https://github.com/SafeAILab/EAGLE (`eagle/train/main.py`,
    `eagle/traineagle3/`); (b) *EAGLE-3*, NeurIPS 2025, arXiv:2503.01840; (c) SpecForge, https://github.com/sgl-project/SpecForge
    (`docs/recipes/qwen3.8-27b-dflash2-disaggregated.md`, `examples/configs/offline/colocated/qwen3-8b-eagle3-offline.yaml`,
    `examples/data_regeneration/`), LK loss arXiv:2602.23881; (d) *Scaling Laws for Speculative Decoding*, arXiv:2505.07858; (e)
    *Attention Drift*, arXiv:2605.09992 (d, e: figures from search results).
19. Zhang et al., *Learning Harmonized Representations for Speculative Sampling* (HASS), ICLR 2025, arXiv:2408.15766;
    https://github.com/HArmonizedSS/HASS.
20. Cai et al., *Medusa*, arXiv:2401.10774; https://github.com/FasterDecoding/Medusa.
21. Chen, Liang, Liu, *DFlash*, arXiv:2602.06036; https://github.com/z-lab/dflash; *DFlash 2*, Inco AI blog, August 2026,
    https://inco.ai/blog/dflash2/ (not fetched).
22. (a) DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437, https://github.com/deepseek-ai/DeepSeek-V3; (b) *GLM-5*,
    arXiv:2602.15763 (from search results); *Nemotron 3 Super*, arXiv:2604.12374; DLoop, arXiv:2610.07659 (cited by
    `docs/MTP_DRAFT_CONFIDENCE.md`; not retrieved).
23. Swift 1.5: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b, https://ukisai.com/swift-1-5-27b,
    https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF (source of `imatrix-swift15-v1mix.gguf`),
    https://huggingface.co/BroLaurens/Swift-1.5-Qwen3.8-27B-NVFP4-GGUF (third-party statement on the MTP head).
24. Third-party summaries of the ISTA GGUFs: https://interfaze.ai/models/ista-daslabqwen38-27b-gsq-rco-gguf,
    https://openclawdc.com/blog/qwen-3-8-27b-gsq-rco-quants/.
25. 2026 GPU prices: https://dev.to/moksh/h100-pricing-in-2026-coreweave-vs-lambda-vs-runpod-gpu-hour-by-gpu-hour-ei0,
    https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/,
    https://effectstory.com/en/articles/cloud-gpu-h100-b200-pricing-comparison-2026.
26. Euhedral at `perf/mtp-gated-depth`: `docs/MTP_CONTRACT.md`, `docs/MTP_VERIFIER.md`, `docs/MTP_DRAFT_CONFIDENCE.md`, `docs/DFLASH2.md`,
    `docs/QUALITY.md`, `docs/P2E2_FULL_RANGE.md`, `tools/euhedral_artifacts/grouped.py`, `nvfp4.py`, `recipes.py`.
