# Fine-tuning after rounding: QAT, block reconstruction, scale-only tuning, low-rank branches, and drafter training

This chapter covers every method that trains something after (or during) low-bit rounding to recover quality: block-wise
reconstruction (OmniQuant, EfficientQAT Block-AP, GSQ), end-to-end quantization-aware training and distillation (EfficientQAT
E2E-QP, ParetoQ, LLM-QAT, BitDistiller, PV-Tuning, NVIDIA's QAD recipes), scale-and-norm-only tuning, low-rank error correction
kept at inference (LoRC, LQER, QERA, CALDERA, SVDQuant-style branches, LoRA on a quantized base), and the one lever unique to an
exactly verified speculative decoder: training the drafter against the quantized target. Codebook formats and rotations are in
their own chapters; QuIP#/QTIP/AQLM fine-tuning is referenced only where the recipe is the point. Everything is argued against
`00-baseline.md`: the `q3` form (+0.170 nats, KL 0.293, 76.2% top-1 at 3.25 bpw for the MLP), `q3-compressed` (+0.275, RTN,
2.65 bpw), `nvfp4` (+0.029) and `nvfp4-compressed` (+0.028 against Swift-1.5 BF16), one RTX 5070 Ti with 16 GB, and the eight
constraints of section 6. Numbers marked *estimate* are mine; everything else is cited.

## 1. Summary

Fine-tuning-based recovery is the only family in this folder that improves quality at **zero bytes and zero kernel changes**: the
output of every method below is a set of Q3/Q4/Q5 codes with binary16 scales (or NVFP4 codes with E4M3 scales) that the existing
converter layout and kernels already serve. The best-supported published recovery at 3 bits is block-wise reconstruction plus
end-to-end scale tuning: EfficientQAT closes a Llama-2-70B W3g128 model to 3.61 against 3.32 FP16 perplexity (0.084 nats) and
W2g64 to 4.52 (0.31 nats), trained in 40.9 hours on one A100 [1]; GSQ's learned grid assignments land within 1.7 zero-shot points
of the vector-quantized frontier at 2.13 bits on Llama-3.1-70B, far ahead of GPTQ and EfficientQAT [4]; PV-Tuning reaches 3.78
against 3.32 at 2 bits on Llama-2-70B but needs a codebook format [3]. The cheapest tier that plausibly pays for `q3` runs on the 16
GB card: block-wise training of one decoder block at a time against cached BF16 activations (6 GB of optimizer state per block,
about 11 to 16 minutes per block, 15 to 20 hours for 65 blocks at EfficientQAT's data budget, *estimate*), and my expectation is
that it roughly halves the `q3` gap (+0.17 to about +0.08 to +0.11 nats, *estimate*). The same training restricted to codes -3..3
is the way to make a 2.5 to 3.0 bpw form viable: it would lift the lossless P2E2 form (`q3-compressed`, 2.65 bpw) from RTN quality
(+0.275) to roughly today's `q3` quality or better (*estimate*), buying 1.56 GiB, about 90K tokens of KV. A uniform 2-bit MLP is
not viable: every published scalar result at W2 on a heavily trained model loses 0.3 to 0.8 nats even after QAT. The expensive
tier (end-to-end QAT or distillation over 1 to 10 B tokens on 8xH100/B200) costs about $400 to $6,000 at 2026 on-demand prices and
buys perhaps another 0.01 to 0.03 nats on `q3` (*estimate*); it is not where the leverage is. Low-rank branches add 0.06 to 0.26
bpw-equivalent and a row-exact fused epilogue; they help most at 2 bits and are dominated at 3 bits by training the 400 M scales
the format already carries. Swift 1.5 ships the base model's MTP head unchanged [23], so fine-tuning the 0.42 B MTP layer against
the quantized target's own greedy generations (loop-unrolled, as HASS and EAGLE-3 do) is a cheap local experiment that can only
change speed, never output: HASS added 8.5% throughput to DeepSeek-R1's official MTP with two epochs of off-distribution data [19].

## 2. Mechanism

### 2.1 The family at a glance

| Method | Trainable set | Objective | Data | Where it runs | Source |
|---|---|---|---|---|---|
| OmniQuant (2023) | per-block learnable clipping (LWC) and equivalent transforms (LET); weights fixed | block output MSE against the FP block | 128 x 2048 tokens; 20 epochs (W3/W4), 40 with `--aug_loss` (W2g64) | one GPU, block by block | [7] |
| EfficientQAT Block-AP (2024) | all weights of one block (FP32 masters) plus step sizes and zero points | MSE of the quantized block's output on the quantized model's inputs against the FP block's output on FP inputs | 4096 x 2048 RedPajama, 2 epochs, batch 2, `quant_lr` 1e-4, `weight_lr` 2e-5 (W2) or 1e-5 (W3/W4) | one GPU, block by block, caches on disk | [1] |
| EfficientQAT E2E-QP | step sizes (scales) only, `module.scales.requires_grad = True` | next-token cross-entropy | 4096 x 4096 RedPajama, 1 epoch, batch 32, lr 2e-5 / 1e-5 | whole quantized model resident | [1] |
| GSQ (2026) | per-weight logits over a 5-level window around a GPTQ init plus per-group scales, Gumbel-softmax relaxation | layer output reconstruction | 4096 x 4096 (FineWeb-Edu for Llama, OpenThoughts reasoning traces for Kimi), 10 epochs per layer, Lion | layer by layer with offloading; 1xH100 for 8B, 4xH100 for 70B | [4] |
| ParetoQ (Meta 2025) | all weights with STE; learnable range (LSQ-style); SEQ grid for 1.58/2-bit, grid with zero for 3/4-bit | next-token loss | about 10% of the pretraining token budget; 3/4-bit "compensation" saturates fast, 2-bit and below needs "reconstruction" | multi-GPU | [2] |
| LLM-QAT (2023) | all weights (W4A8KV4 setting) | logit distillation from the FP model | 100K self-generated samples, data-free | multi-GPU | [6] |
| BitDistiller (2024) | all weights after asymmetric clipping init | confidence-aware KL (CAKLD) to the FP teacher | 8,000 teacher-generated samples of 1024 tokens (3000 WikiText + 5000 Alpaca), lr 8e-6 constant, 2 to 4 epochs, 8 GPUs | DeepSpeed, 8 GPUs | [5] |
| PV-Tuning (2024) | codes, codebooks, scales and non-quantized parameters | cross-entropy with teacher logits | RedPajama slice, batch 128 x 4096, 1 epoch, lr 3e-4, `code_lr` 1e-2, beam 5 | multi-GPU | [3] |
| QuEST (2025) | all weights and activations, from scratch | pretraining loss with Hadamard-normalized MSE-optimal quantizer and a trust gradient estimator | pretraining data | multi-GPU | [8] |
| Norm Tweaking (2023) | LayerNorm parameters only, per layer | channel-wise distribution loss against the float layer | self-generated text | one GPU | [9] |
| NVIDIA QAD (2026) | all weights, or LSQ `amax` scales only, or LoRA adapters on a frozen quantized base | KL to the BF16 teacher's logits (top-k) | 20K x 8192 (HF recipe) up to 8.4 B tokens (Megatron) | 1 node to 128 GPUs | [17] |

Two facts organize the family. First, ParetoQ's learning transition: at 3 and 4 bits a fine-tuned model stays close to its
pretrained distribution and saturates with few tokens ("compensation"), while 2-bit and below must rebuild representations and
needs far more tokens ("reconstruction") [2]. `q3` sits in the cheap regime; a uniform 2-bit form does not. Second, every
block-wise method keeps only one block on the GPU, so model size enters as disk and time, not as VRAM.

### 2.2 Block-wise reconstruction on one 16 GB card

EfficientQAT's loop (`quantize/block_ap.py`): cache the inputs of block l for the FP model and for the quantized model (two
caches), run the FP block over the FP cache to make the targets, replace the block's linears with fake-quant linears, train the
block in FP32 under autocast with AdamW for 2 epochs on batches of 2 sequences, then run the trained quantized block and the FP
block forward to produce the two caches of block l+1. `--off_load_to_disk` keeps the caches on disk. Memory per block is the
block's optimizer state plus one micro-batch of activations; the whole model never sits in VRAM.

For this model (shapes from `00-baseline.md` section 1):

| Item | GDN block | Attention block | MTP block | Note |
|---|---|---|---|---|
| Parameters | 382.7 M (gate_up 178.3, down 89.1, value_z 62.9, output 31.5, query_key 21.0, small objects) | 372.2 M | 424.7 M | computed from the shapes |
| FP32 master + grad + Adam m,v (16 B/param) | 6.1 GB | 6.0 GB | 6.8 GB | EfficientQAT trains in FP32 |
| BF16 weights + grads with 8-bit Adam | 2.3 GB | 2.2 GB | 2.6 GB | alternative |
| Activations, 2 x 2048 tokens | about 1.5 to 3 GB (*estimate*) | same | same | gate_up output 2 x 2048 x 34816 BF16 = 285 MB; GDN chunked states |
| Fits in 14.6 GiB usable | yes | yes | yes | desktop uses 0.75 to 0.95 GiB |
| Forward+backward FLOPs at 4096 x 2048 x 2 epochs | 3.85e16 | 3.75e16 | | 6 x params x tokens |
| Time per block at 40 to 60 TFLOPS achieved (*estimate*) | 11 to 16 min | 10 to 16 min | | 5070 Ti BF16 dense peak 102 TFLOPS |
| Total, 64 blocks + MTP (*estimate*) | 12 to 17 h compute; 15 to 20 h with cache I/O | | | |
| Same at 1024 x 2048 samples (*estimate*) | 3 to 4 min per block, about 5 h total | | | |

The caches, not VRAM, are the constraint: 4096 x 2048 x 5120 BF16 is 85.9 GB per cache, 172 GB for the two, read twice per
block; at 1024 samples it is 43 GB. The host RAM on the maintainer's machine already swaps at `--jobs 16`, so the caches go to
NVMe (about 500 GB of traffic per block at the default budget, 3 minutes at 3 GB/s). For scale: EfficientQAT measured Block-AP on
Llama-2-70B at 26.6 hours and 29.9 GB on one A100-80GB [1]; scaled by parameter count a 27B model is about 10 A100-hours, 3 to 6
hours on an H100 ($10 to $25, *estimate*). The BF16 teacher never has to fit on the GPU: each block's BF16 weights (at most 766 MB)
are loaded once from the 54.7 GB checkpoint. The GDN block needs a PyTorch implementation with a backward pass (the chunked gated
delta rule of `flash-linear-attention`, which the Hugging Face `Qwen3_5` modeling code uses).

GSQ is the same loop with a different parameterization: a logit vector per weight over five integer levels around the GPTQ
initialization and a learned scale per group, annealed Gumbel-softmax, Lion optimizer, 10 epochs over 4096 x 4096 tokens per layer
[4]. Its budget is 10x EfficientQAT's per layer: 3.9e17 FLOPs per block here, 2.5e19 for the model, about 17 hours on one H100 or a
week on the 5070 Ti (*estimate*; the repository reports about 12 hours on 4 GPUs for the 40 layers of Qwen3.5-35B-A3B). It is the
method behind ISTA's Qwen3.8-27B GSQ-RCO GGUFs at 2.50 to 3.50 bpw [4], and it supports the Qwen3.5/3.6 hybrid wrappers.

### 2.3 End-to-end: scales and norms only, all parameters, and the objective

**Scales and norms only.** The Q3/Q4/Q5 formats carry one binary16 scale per 64 weights: 400 M scales over the 25.6 B streamed
parameters (267 M in the MLP alone), plus about 0.66 M RMSNorm weights (`(1 + w)` form, stored BF16). Training only these with the
codes frozen needs no straight-through estimator for the weights: the dequantized weight is `code x s`, so the gradient to `s` is
exact (`sum over the group of dL/dW_hat x code`), and only the binary16 rounding of `s` needs an STE (EfficientQAT's E2E-QP trains
step sizes this way; NVIDIA ships a `qad_scale_only.yaml` recipe that trains only LSQ `amax` scales at lr 1e-4 against the BF16
teacher [17]). The engine's own history says scales matter at 3 bits: the 39-candidate weighted scale search took RTN Q3 from
+0.275 to +0.170 nats with no other change (`00-baseline.md` section 3). End-to-end scale training on the 16 GB card is marginal:
packed codes (about 10 GiB) plus a BF16 scale copy and its gradient (1.6 GB) plus 1.3 GB of activation checkpoints for 2048 tokens,
with the FP32 master scales and Adam states (4.8 GB) offloaded to the host (*estimate*). It is simpler to train scales block-wise
(6.25 M scales per block) or to rent one H100 for the end-to-end pass: 6 x 27 B x 16.8 M tokens = 2.7e18 FLOPs, about 2 hours at
400 TFLOPS achieved, $5 to $8 (*estimate*). On the 5070 Ti the same pass is about 19 hours of compute.

**All parameters, end to end.** Memory, not FLOPs, moves this to rented hardware: 27 B parameters in FP32 master plus Adam is 432
GB, so full-parameter QAT needs 8x80 GB with FSDP/ZeRO-3 regardless of token count, and a resident BF16 teacher adds 54.7 GB. ModelOpt
reports 124 GB per GPU for a 35B-A3B student plus teacher at 32K sequence length [17]. A LoRA-QAD variant (ModelOpt `--lora_rank`,
backbone frozen, `quantize(W + (alpha/r) BA)` per linear) and the scale-only variant fit on 1 to 2 H100s or one H200/B200.

**Objective.** The evidence favours the teacher's logits over hard labels at low bits: LLM-QAT found logit distillation superior to
label training [6]; BitDistiller's CAKLD mixes forward and reverse KL by teacher confidence [5]; ModelOpt's QAD uses KL over the
teacher's top-4096 logits [17]; OnlineQAT's baseline is forward KL over the top-20 plus a 0.2-weighted cross-entropy on gold tokens
[16f]. Two 2026 results go further: on-policy distillation (student generates, teacher scores) gained 4.32 points in 20 steps where
800 teacher-forced QAD steps gained 4.09 on Qwen3-4B at 2.79 bits, and a larger teacher did not help from 1.7B up [16e]; OnlineQAT
initializes with block-wise QAT and then distills on policy [16f]. The practical consequence for a 248,320-row vocabulary: store the
teacher's top-k log-probabilities offline (64 entries as int32 id + fp16 value is 384 B per token, 3.8 GB for 10 M tokens); the
repository's own `bf16.kld` format (16-bit log-probabilities over the 16 nats below each position's maximum) is already such a
store.

### 2.4 Simulating the engine's formats in PyTorch

The fake-quant forward must round where `grouped.py` and `nvfp4.py` round; the matmul order does not matter for training (the
relaxed-kernel drift floor is KL 3.4e-3 to 4.7e-3, 30 to 85 times below the quantization gap).

- **Q3G64_F16S, Q4G64_F16S, Q5G64_F16S.** Groups of 64 along K, K padded to a multiple of 128 with zeros. `W_hat = clamp(round(W /
  s), qmin, qmax) x s` with (qmin, qmax) = (-4, 3), (-8, 7), (-16, 15). `s` is a trainable FP32 scalar per group rounded to binary16
  in the forward (`s16 = s + (fp16(s) - s).detach()`), initialized from the converter's own search (so training starts at today's
  artifact, never worse). Weights use the straight-through estimator (ParetoQ, EfficientQAT); scales use the LSQ gradient. For a
  P2E2-compatible result, clamp to (-3, 3): the compressed layout cannot store -4 (`docs/P2E2_FULL_RANGE.md`). Export writes codes
  and scales straight into the row-split planes with `pack_codes_torch` and the scale writer; do not round-trip through BF16 (`code x
  fp16 scale` has up to 14 significant bits and is not BF16-exact). The mixed recipe of `recipes.py` (Q4 `query_key`, Q5 `value_z`
  and `gate_value`, Q3 elsewhere, Q3 embedding and LM head) must be mirrored tensor for tensor.
- **NVFP4.** Per tensor `g = amax / (6 x 448)` (FP32, fixed), per 16-block `s_b` an E4M3 code, `W_hat = sign x e2m1(|W| / (s_b g)) x
  s_b g`. Two training options: (a) weights only, with `s_b` recomputed from the current weights each step as the converter does
  (round-to-nearest E4M3, with the imatrix search applied once at export); (b) a learnable continuous block scale snapped to E4M3
  with an STE, which is what ModelOpt's LSQ-NVFP4 recipe does (`learnable_amax: pre, post`, MSE init, FP8 scale sweep) [17]. The
  `INT_vs_FP` repository has a compact reference fake-quant for NVFP4's E4M3 scale quantization (`s_of_s = amax(s)/448; (s /
  s_of_s).to(float8_e4m3fn)`) [13] to cross-check against `nvfp4.py` byte for byte. For SD4, keep the 16-entry table fixed during
  training and re-run the 6-minute table DP at export, or train against the fixed table.
- **Stochastic rounding.** In the NVFP4 pretraining recipe stochastic rounding is applied to gradients in the backward GEMMs; weights
  and activations use round-to-nearest-even, weights carry 2D 16x16 scales, and the random Hadamard transform is applied only to the
  weight-gradient GEMM inputs [14]. None of that is a weight-only fine-tune. For our purposes the deterministic choice is RTN plus STE
  with learnable scales; stochastic rounding of weights is an option against STE dead zones but must be annealed back to RTN before
  export, because the artifact stores rounded codes.
- **Not simulated.** KV-cache NVFP4 (Hadamard-rotated), the two-term activation quantization of the prefill routes, and FP32
  accumulation order. The teacher-forced harness measures their effect at the 4e-3 KL level; standard practice is to leave them out
  of the weight fine-tune.

### 2.5 Low-rank branches: the arithmetic for this model

A branch stores `W ~ Q + B A` with `A` (r x K) and `B` (N x r); LoRC takes the top-r SVD of the rounding error [12a], LQER/L2QER
scale the error by activation statistics before the SVD [12b], QERA derives the factors analytically from the layer-output error
[12c], CALDERA alternates between re-quantizing Q and refitting LR and optionally LoRA-trains the factors [12d], SVDQuant moves
outliers into a rank-32 16-bit branch before 4-bit quantization [12e]. Over the 320 linears of the 64 blocks, sum(N + K) is 6.39 M:

| Rank | Branch parameters | BF16 bytes | bpw-equivalent over 25.6 B | FLOPs per token | Decode time at 855 GB/s | NVFP4-stored branch |
|---|---|---|---|---|---|---|
| 16 | 102 M | 195 MiB (+8 MiB LM head) | 0.064 | 0.20 GFLOP | +0.24 ms (+1.5% of the 16 ms Q3 step) | 55 MiB, +0.07 ms |
| 32 | 204 M | 390 MiB (+15 MiB) | 0.128 | 0.41 GFLOP | +0.48 ms (+3%) | 110 MiB, +0.13 ms |
| 64 | 409 M | 780 MiB (+31 MiB) | 0.256 | 0.82 GFLOP | +0.96 ms (+6%) | 220 MiB, +0.27 ms |

Bytes are the only real cost at one row; FLOPs are 0.8% of the base model's at rank 32. Launches are not free: two extra GEMVs per
linear would be 640 launches per token, 1 to 2 ms even from captured graphs, so the branch must be fused: compute `y = A x` once per
distinct block input (four per block: the normed residual, the attention/GDN output, the post-attention norm, the SwiGLU output,
stacking the A's of linears that share an input) in the kernel that already produces that input, and add `B[row, :] . y` in the
GEMV epilogue (r FMAs per output row against K = 5120 to 17408 for the main dot product). "Q3 plus rank-32 BF16 branch" is 3.25 +
0.13 = 3.38 bpw on the MLP, 11.1 GiB streamed instead of 10.7. Note that the branch has half the trainable parameters of the scale
plane (204 M against 400 M) and the scale plane costs nothing extra at inference, so a branch is only justified by its expressivity
(dense low-rank) after the scales have been trained.

### 2.6 Training the drafter

EAGLE-1 trains a one-layer autoregression head on the target's second-to-top features with `1.0 x feature regression + 0.1 x token
loss`, 20 epochs, lr 3e-5, "within 1 to 2 days on 8x RTX 3090" [18a]; EAGLE-3 drops the feature loss and unrolls the head over its
own outputs during training ("training-time test"; SpecForge runs `ttt_length: 7`, 10 epochs, lr 1e-4 on ShareGPT) [18b][18c]; HASS
harmonizes the draft model's objective and context between training and decoding [19]; Medusa-1 trains only the new heads at lr
1e-3 for 2 epochs on ShareGPT and Medusa-2 adds self-distillation for fine-tuned targets [20]; DFlash trains a 5-layer block drafter
against a frozen target on responses regenerated by the target itself [21]. The SpecForge recipe for DFlash2 on Qwen3.8-27B is the
closest datapoint to this engine: a 1.9 B parameter draft reading the hidden states after layers 5, 19, 33, 47 and 61 plus the last
hidden state, trained on 1.07 M ShareGPT-style conversations whose assistant turns were regenerated by Qwen3.8-27B itself with
reasoning kept (mean 4.5K tokens, about 4.8 B tokens), hard-target cross-entropy with a strict top-k selector objective, constant
lr 5e-4, 512 samples per step for full-convergence runs, at 9 to 22.65 samples/s on 8x H200, i.e. about 13 hours per epoch
(*computed from the recipe's figures*); the recipe states that acceptance on off-policy answers "plateaus far below" what the same
draft reaches on regenerated data [18c]. DeepSeek-V3 reports 85 to 90% acceptance of its MTP's second token and notes that the module
was trained for one step and reused autoregressively [22a]; GLM-5 shares the parameters of 3 MTP layers during training to remove
that train-inference discrepancy [22b].

For Euhedral the drafter is the base model's MTP block (0.42 B parameters: `fc` 52 M, packed q/k/gate/v 73 M, `o` 31 M, gate_up
178 M, down 89 M), stored NVFP4 in every artifact and applied recursively (depth 2 on Q3, 4 on NVFP4). Fine-tuning it is ordinary
EAGLE-style training with three specifics: the input hidden `h_p` must come from the **quantized** target (the engine's row-exact
verifier produces exactly the hidden the drafter will see in deployment; a PyTorch fake-quant replica of the artifact produces the
same values up to kernel-order noise), the recursion must be unrolled for depth 2 to 4 with the drafted (or gold) token fed back, and
the MTP weights should be fake-quantized to NVFP4 during training so that the exported NVFP4 layer is the trained one. The loss that
matches a greedy verifier is cross-entropy against the target's argmax (the verifier checks nothing else); adding a KL term to the
target's distribution gives a denser signal. Cost on the 16 GB card: optimizer state 6.8 GB (FP32) or 2.6 GB (8-bit Adam), the
frozen LM head in BF16 (2.5 GB) with chunked logits, about 7.6 GFLOP per token at depth 1 and 23 GFLOP with three unrolled drafts
(the LM head's forward and input gradient dominate), so 10 M tokens take 0.5 to 1.6 hours and 100 M tokens 5 to 16 hours
(*estimate*). Hidden states are 10 KB per token in BF16 (102 GB per 10 M tokens) if dumped; computing them on the fly from the
resident Q3 codes costs 2 x 27 B FLOPs per token, about 4 hours per 10 M tokens. The DFlash2 drafter (1.9 B parameters) does not fit
full-parameter training on 16 GB; LoRA on its 25 projections would, but `q3` artifacts have no DFlash2 drafter yet (`docs/DFLASH2.md`
section 9), so the MTP layer is the local target.

## 3. Published results

All perplexities are WikiText-2 unless stated. "Nats" is `ln(ppl_quant / ppl_fp16)`, the same quantity as the repository's dNLL.

**Uniform scalar QAT and reconstruction** (EfficientQAT model zoo [1]):

| Model | FP16 | W4g128 | W3g128 | W2g64 | W2g128 | Nats at W3g128 / W2g64 | Cost (paper) |
|---|---|---|---|---|---|---|---|
| Llama-2-7B | 5.47 | 5.53 | 5.81 | 6.86 | 7.17 | 0.060 / 0.226 | 4.8 h, one A100-80GB |
| Llama-2-13B | 4.88 | 4.93 | 5.12 | 5.96 | 6.08 | 0.048 / 0.200 | |
| Llama-2-70B | 3.32 | 3.39 | 3.61 | 4.52 | 4.61 | 0.084 / 0.309 | 40.9 h, one A100-80GB: Block-AP 26.6 h / 29.9 GB, E2E-QP 14.3 h / 34.2 GB at 2 bits |
| Llama-3-8B | 6.14 | 6.47 | 7.09 | 9.41 | 9.80 | 0.144 / 0.427 | |
| Llama-3-70B | 2.85 | 3.17 | 4.19 | 6.08 | 6.38 | 0.385 / 0.758 | |
| Llama-3-70B-Instruct | 5.33 | 5.35 | 5.65 | 7.86 | 8.14 | 0.058 / 0.389 | |

Llama-3 loses 2 to 4 times more than Llama-2 at the same width, consistent with "Low-Bit Quantization Favors Undertrained LLMs"
[11]: models trained on more tokens degrade more. Qwen3.8-27B is in the Llama-3 class, and the engine's `q3` (0.171 nats at an
average 3.59 bpw over the streamed tensors, MLP at 3.25) already sits between EfficientQAT's Llama-3-8B and Llama-3-70B W3g128
results without any error feedback. OmniQuant's own tables were not retrievable this session beyond a snippet (LLaMA-13B W2A16: 13.21
against GPTQ 3832; W2A16g64: 7.34 against GPTQ 10.06, model column unverified) [7]; BitDistiller's per-model tables and its GPU-hours
were not retrievable either (its abstract claims +3.54 points over LLM-QAT and +12.43 over the best PTQ at 2 bits on general tasks)
[5].

**Codebook fine-tuning and learned assignments, 2 bits** ([3], [4], [12d]):

| Model | Method | bpw | Metric | Result | Baseline | Cost |
|---|---|---|---|---|---|---|
| Llama-2-7B | AQLM + PV-Tuning 1x16g8 | about 2 | ppl | 5.68 (0.038 nats) | FP16 5.47 | PV-Tuning GPU-hours not retrieved; AQLM calibration alone is about 1 day per 7B on one A100, 10 to 14 days per 70B |
| Llama-2-13B | AQLM + PV-Tuning | about 2 | ppl | 5.05 (0.034) | 4.88 | |
| Llama-2-70B | AQLM + PV-Tuning | about 2 | ppl | 3.78 (0.130) | 3.32 | |
| Llama-3.1-8B-Instruct | GSQ | 2.13 | zero-shot avg (5 tasks) | 68.55 | FP16 73.71; GPTQ 37.53; QuIP 39.20; EfficientQAT 63.79; QTIP 69.88; PV-Tuning 69.83 | 1xH100 |
| Llama-3.1-70B-Instruct | GSQ | 2.13 | zero-shot avg | 75.57 | FP16 78.99; GPTQ 57.38; EfficientQAT 71.43; QTIP 77.25; PV-Tuning 76.27 | 4xH100 |
| Qwen3-8B Q2_K GGUF | GSQ in-format refinement | Q2_K | avg score | 56.28 | Unsloth init 50.03 | |
| Llama-2-70B | CALDERA, rank 128, 4-bit factors | about 2.1 | ppl WikiText-2 / C4 | 4.11 / 5.95 (0.213 nats) | QuIP# 2-bit without fine-tuning 5.37 / 7.51 (earlier paper version; later version re-ran QuIP#) | |
| Llama-2-70B | CALDERA, rank 256, 4-bit factors | about 2.2 | ppl | 3.98 / 5.76 (0.181) | | |

GSQ's two numbers are the most useful calibration of expectations: a learned *scalar* assignment closes most of the distance from
GPTQ/EfficientQAT to the vector-quantized frontier, and it is what ISTA used for the public Qwen3.8-27B GGUFs (IQ2_XS 2.50 bpw 8.4
GB, IQ2_S 2.75, IQ3_XXS 3.00, IQ3_S 3.50 "recommended"; a third-party summary reports IQ3_S matching the base on AIME25 and
LiveCodeBench v6 and trailing by 0.51 on GPQA-Diamond) [4][24]. Those files use llama.cpp codebook types (other chapter); the
learned-rounding part applies to uniform grids.

**Industrial QAD on a Qwen hybrid-attention MoE** (NVIDIA ModelOpt tutorial, Qwen3.6-35B-A3B, W4A4 NVFP4 with FP8 KV) [17]:

| Model | MMMU-Pro | GPQA-D | SciCode | AA-LCR | IFBench | tau2 | Avg | Cost |
|---|---|---|---|---|---|---|---|---|
| BF16 teacher | 74.6 | 84.7 | 39.9 | 69.1 | 60.0 | 94.2 | 70.4 | |
| W4A4 NVFP4 PTQ | 73.4 | 84.7 | 39.1 | 70.0 | 57.9 | 94.2 | 69.9 | 1.2 GPU-h |
| + QAD, 500 iterations (8.4 B tokens, KL top-4096, lr 1e-5 to 1e-6) | 73.9 | 84.2 | 40.2 | 69.4 | 59.6 | 93.4 | 70.1 | 735 GB200 GPU-h, 5.7 h on 128 GPUs |

Only IFBench (-2.6 to -0.3 pp) and MMMU-Pro (-1.2 to -0.7) had real losses to recover; the side effect matters for Swift 1.5: QAD
raised SciCode output length by 90.9% (non-terminating `<think>` loops rose from 0.7% to 3.6% of sub-steps) while the score was
unchanged, and 4-bit weights alone lengthened outputs 23 to 25%; a decode-time `presence_penalty` 1.5 removed it [17].

**Reasoning models and calibration data** ([16a] to [16f]):

| Finding | Setting | Source |
|---|---|---|
| Self-generated reasoning traces (NuminaMath-1.5, 128 x 2048) used as the GPTQ calibration set because of the domain gap to WikiText2 | DeepSeek-R1-Distill 1.5B to 70B, QwQ-32B, W3/W4 | [16a] |
| The model's own traces beat DeepSeek-R1-671B's traces as calibration data by +25.54 pp mean accuracy; C4/WikiText2 calibration collapsed | Qwen3-4B, 1.58-bit PTQ | [16b] |
| Reasoning calibration gains up to +38.45 pp AIME25 pass@1; largest at small scale and low bits | Qwen3-4B, W3G128 GPTQ | [16c] |
| Aggressive PTQ lowers accuracy and lengthens chains of thought; in up to 52% of failures the right answer appears mid-trace; hesitation tokens at high-KL positions; a logit penalty on 50 markers shortens CoT 12 to 23% | 1.5B to 32B, 3 PTQ methods, 5 benchmarks | [16d] |
| On-policy distillation: 20 steps gain 4.32 points against 4.09 from 800 teacher-forced QAD steps; larger teacher does not help from 1.7B up | Qwen3-4B at 2.79 bits | [16e] |
| Block-wise QAT init, then on-policy distillation: Qwen3-1.7B 57.28 at W3A16, 32.52 at W2A16 (+2.90 / +0.44 over ReasoningQAT) | Qwen3-1.7B | [16f] |

**Drafter training** ([18] to [22]):

| Result | Source |
|---|---|
| EAGLE-1 head: trainable within 1 to 2 days on 8x RTX 3090 | [18a] |
| EAGLE-3 SFT: 20 epochs, roughly 80 hours in the original configuration; 2 epochs of 8x-augmented UltraChat-200K on 8x H800 in about 8 hours (from "Scaling Laws for Speculative Decoding") | [18d] |
| About 48 H200-hours per 8 to 9 B draft on 1.4 M samples, 2 epochs, answers regenerated by the target ("Attention Drift") | [18e] |
| HASS continual training of DeepSeek-R1's official MTP, 2 epochs of ShareGPT: mean throughput 62.46 to 67.75 tok/s at T=0 (1.76x to 1.91x, +8.47%), with data the authors call "extremely less than the official MTP" and off-distribution | [19] |
| HASS on DeepSeek-R1-Distill-Qwen-32B: acceptance length 4.06 (T=0, mean over 6 sets); +8 to 20% over EAGLE-2 on LLaMA | [19] |
| DFlash2 for Qwen3.8-27B: 4.8 B tokens of self-regenerated conversations, 8x H200 at up to 22.65 samples/s | [18c] |
| DeepSeek-V3 MTP second-token acceptance 85 to 90% | [22a] |

**Hybrid SSM/attention models.** None of EfficientQAT, PV-Tuning, ParetoQ, BitDistiller, OmniQuant or QuEST reports a hybrid. The
hybrid evidence is: GSQ supports the Qwen3.5/3.6 hybrid wrappers and produced the Qwen3.8-27B GGUFs [4]; "Why Gated DeltaNet
Survives 4-Bit Quantization" finds calibration-only PTQ reaches BF16-level task accuracy for NVFP4 W4A4 on the recurrent half of a
27 B hybrid and mentions a concurrent QUASAR checkpoint of the same model that quantizes all 496 projections by QAT [15]; NVIDIA
pretrained a 12 B hybrid Mamba-Transformer in NVFP4 over 10 T tokens [14]. No paper reports W2/W3 QAT recovery for a GDN hybrid.

## 4. Fit with Euhedral

### 4.1 The eight constraints

| Constraint | Fine-tuned codes and scales | Low-rank branch | Fine-tuned MTP / drafter |
|---|---|---|---|
| 1. Row-exact 1 to 8 row decode | Unchanged: same formats, same kernels, same twins. | The fused epilogue `B[row,:] . y` is per row and deterministic; the `A x` pre-pass is shared by all rows but each row's result is independent of the row count. Row-exact by construction, like the MMA-column argument. A BF16 branch needs no new exactness proof beyond `test_q3_kernels`-style twin tests. | Not involved: the verifier never reads the drafter's arithmetic. |
| 2. Prefill at 9+ rows off the BF16 path | Unchanged. | Adds a BF16 GEMM pair of 2 x T x r x (N + K) FLOPs, 0.8% of the main GEMM at r = 32, plus an add over the output tile; about 2% of prefill time (*estimate*). | Unchanged. |
| 3. Decode instruction budget | Unchanged. | +r FMAs per output row (0.2 to 0.6%); the `A x` pre-pass is r x K per block input. Within budget. | Unchanged. |
| 4. Determinism | Weights are data. | Deterministic kernels. | Deterministic. |
| 5. Offline conversion only | The fine-tune is an offline step that emits codes and scales; the converter needs an entry that writes pre-rounded codes into the existing planes instead of rounding BF16 (new recipe, same format and layout ordinals). | New EDRL object class and layout ordinal for `A`/`B`; old artifacts keep loading. | Replaces the `mtp/*` NVFP4 objects; nothing else changes. |
| 6. The artifact decides everything | A fine-tuned `q3` is a new recipe (`q3-ft`) that the engine cannot distinguish from `q3`. | `ArtifactProfile` must detect branch objects and select the fused kernels. | `ArtifactProfile.speculativeDepth` was screened for the shipped head; a better head may move the best depth (MTP3 on Q3, MTP5 on NVFP4) and needs re-screening. |
| 7. Memory is the product | Zero bytes. A fine-tune restricted to codes -3..3 makes the 2.65-bpw P2E2 form carry calibrated quality: 1.56 GiB less than `q3`, about 90K tokens of KV at 18 KiB. | +195 to +780 MiB resident; on the NVFP4 forms this pushes weights to the host at 2 to 2.5% decode per 128 MiB, so a branch is net negative there; `q3` has the room (2.4 GiB free at 48K). | Zero bytes (same objects). |
| 8. Drafting precision is a separate knob | n/a | n/a | Exactly this knob: speed only, never correctness. |

### 4.2 Which forms benefit, and expected quality

Reasoning: at 3 bits the model is in ParetoQ's compensation regime, EfficientQAT's W3 results stay within 0.06 to 0.14 nats on Llama-2
and Llama-3-8B, GSQ reports that learned scalar assignments recover most of the GPTQ-to-VQ gap, and the engine's rounding has no
error feedback at all today, so there is room. The numbers below are estimates in the repository's metric (dNLL against the BF16 of
the same checkpoint, on the current single document; its s.e. is 0.021 for Q3-class artifacts, so only the `q3` rows are resolvable
without the larger holdout of section 4.5).

| Form | Today | Tier 0: scales + norms (block-wise, 16 GB) | Tier 1: block-wise all parameters (16 GB or 1xH100) | Tier 2: end-to-end QAD, 100 M to 1 B tokens (rented) | Confidence |
|---|---|---|---|---|---|
| `q3` (3.25 bpw MLP) | +0.170, KL 0.293 | +0.13 to +0.15 | +0.08 to +0.11, KL about 0.15 to 0.18 | +0.07 to +0.09 | medium: W3 literature is consistent |
| `q3-compressed` (2.65 bpw, codes -3..3) | +0.275 (RTN) | +0.19 to +0.22 | +0.10 to +0.15 | +0.09 to +0.13 | medium: the 7-level grid is a mild restriction of the 8-level one |
| Uniform 2.5 bpw (Q2 MLP) | not built | | +0.4 to +0.6 | +0.3 to +0.5 | high that it is not viable: every W2 scalar result on a Llama-3-class model loses 0.3 to 0.8 nats |
| `nvfp4`, `nvfp4-compressed` (4.5 / 4.25 bpw) | +0.029 / +0.028 | within noise | +0.015 to +0.02 | +0.01 to +0.02 | low: below the harness resolution; ModelOpt's W4A4 QAD recovered 1 to 2 pp on two of six benchmarks |
| `q3` + rank-32 BF16 branch (3.38 bpw), factors trained | | | about the Tier 1 figure, perhaps 0.01 to 0.02 better | | low: CALDERA's gains are at 2 bits; at 3 bits the trained scale plane already carries more parameters |

Verdict on "Q3 plus rank-32 branch against NVFP4": not competitive in quality. The branch cannot be expected to take a 3.25-bpw
backbone from +0.17 to +0.03; the published low-rank gains are largest when the backbone is very coarse (CALDERA at 2.1 bpw, 0.21
nats against about 0.48 for QuIP# 2-bit without fine-tuning in the earlier paper version), and at W3 LQER needed rank about 600 for
plain SVD-of-error on OPT-1.3B. The byte-efficient lever at 3.4 bpw is the trained scale plane (free) and, for context, the P2E2 form.

On which checkpoint to run: the deployed artifact is `nvfp4-compressed` from Swift-1.5 with UkisAI's `imatrix-swift15-v1mix.gguf`
(a reasoning, multilingual and code mix); `q3` and `nvfp4` are still converted from the base model with ISTA's matrix. Every
fine-tuning experiment should run on the checkpoint that will be served, with that checkpoint's BF16 as teacher and reference
(Swift-1.5 BF16: perplexity 8.51, NLL 2.141 on the quality document). Converting Swift-1.5 to `q3` with the v1mix matrix and
measuring it is the zero-training baseline every experiment below compares against.

### 4.3 Cost table

Assumptions (*estimates*): 40 to 60 TFLOPS achieved on the 5070 Ti, 400 on an H100, 900 on a B200; 2026 on-demand prices from
public comparisons: H100 $2.49 to $3.99 per GPU-hour (RunPod community $2.69, secure $3.29 to $3.49; Lambda $2.49 to $3.99), B200
$6.69 to $6.79 [25]; spot and community tiers are cheaper but interruptible.

| Tier | What is trained | Hardware | Time | Dollars | Data |
|---|---|---|---|---|---|
| 0a | scales + norms, block-wise | 5070 Ti, caches on NVMe | 5 h (1024 x 2048) to 15 to 20 h (4096 x 2048) | 0 | 2 to 8 M tokens, teacher activations |
| 0b | scales only, end to end, KL to teacher | 5070 Ti with host-offloaded optimizer (marginal fit) or 1xH100 | 19 h / 2 h | 0 / $5 to $8 | 16.8 M tokens; teacher top-k logits precomputed (6 to 7 h on the 5070 Ti with layer streaming, or llama.cpp on the CPU) |
| 1 | all parameters of each block (EfficientQAT Block-AP) | 5070 Ti / 1xH100 | 15 to 20 h / 3 to 6 h | 0 / $10 to $25 | 8.4 M tokens, 172 GB of caches (43 GB at 1024 samples) |
| 1b | GSQ-style learned assignments, 10 epochs per layer | 1xH100 / 5070 Ti | about 17 h / about a week | $45 to $70 / 0 | 16.8 M tokens per layer |
| 2a | all parameters or LoRA, QAD, 100 M tokens | 8xH100 (memory-bound for full parameters) | about 2 h with teacher online | $50 to $70 | 100 M tokens, half own generations |
| 2b | QAD, 1 B tokens | 8xH100 / 8xB200 | 15 to 20 h / 6 to 8 h | $400 to $650 / $340 to $450 | |
| 2c | QAD, 10 B tokens (ParetoQ-scale) | 8xH100 / 8xB200 | 150 to 190 h / 65 to 85 h | $3,000 to $6,000 / $3,500 to $4,600 | reference: ModelOpt's 8.4 B-token W4A4 QAD took 735 GB200 GPU-hours |
| 3 | MTP block, loop-unrolled, NVFP4 fake-quant | 5070 Ti | 0.5 to 1.6 h per 10 M tokens of training; 4 h per 10 M tokens of hidden states, or dump them during generation | 0 | 5 to 50 M tokens of own greedy generations (thinking on) |

### 4.4 Data: which text, how many tokens, own generations or not

- **Token counts in the literature**: Block-AP 8.4 M (4096 x 2048), E2E-QP 16.8 M, GSQ 16.8 M per layer, OmniQuant 0.26 M (128 x
  2048, 20 to 40 epochs), BitDistiller about 8 M (8,000 x 1024, teacher-generated), LLM-QAT 100K self-generated samples, ModelOpt HF
  QAD 20K x 8192 (up to 164 M), ModelOpt Megatron QAD 8.4 B for W4A4, ParetoQ about 10% of pretraining. For weight-only 3-bit
  recovery the evidence says 10 to 50 M tokens is the regime of diminishing returns; 16 to 32 M is a sensible budget here.
- **Own generations.** The evidence for calibrating and distilling reasoning models on their own traces is consistent across PTQ
  (Liu et al., AYOT, LAQuant), QAT (BitDistiller, LLM-QAT, OnlineQAT, on-policy distillation) and drafters (SpecForge's regenerated
  data, DFlash) [16][5][6][18c][21]. For Swift 1.5 the teacher data must be Swift 1.5's own generations with thinking, not the
  base's: the base thinks about twice as long (UkisAI reports a 41.9% mean and 58.5% median token reduction on GPQA-Diamond [23]),
  and distilling toward base-model traces would train the quantized model back toward the behaviour the fine-tune removed.
  Generating 10 M tokens with Swift BF16 on this machine is impractical (CPU llama.cpp), so generate with the deployed
  `nvfp4-compressed` artifact at 100 to 146 tok/s (10 M tokens is 19 to 28 hours single-stream; the prompt sets in
  `tools/mtp_draft_prompts.py` plus public prompt collections give the prompts) and score the sequences teacher-forced with Swift
  BF16 top-k logits. This is the on-policy recipe that [16e] and [16f] found more efficient than teacher-forced data, and the
  mismatch between the artifact's and BF16's greedy paths (top-1 agreement 89.4%) is small.
- **Mix.** Half own generations (chat, code, agentic with tool calls and tool results, math, thinking on; the repository's
  `MtpDraftConfidenceCudaIntegrationTest` already defines such sets), half general text of the v1mix kind (reasoning, multilingual,
  code) to keep the broad distribution and to detect overfitting to self-generated style. Sequence length 2048 to 4096; longer
  sequences cost activation memory per block for no quantization benefit.
- **The imatrix pipeline is already Swift-specific** (`llama-imatrix` on a BF16 GGUF of Swift-1.5, and UkisAI's v1mix). The same text
  should seed Tier 0/1 so that the calibration and the fine-tune agree on what matters.

### 4.5 Evaluation holdout

The current single 1279-token document gives s.e. 0.021 nats on `q3` (per-token s.d. of the paired dNLL about 0.75 nats) and 0.008 on
`nvfp4` (s.d. 0.29). To resolve 0.01-nat effects at two standard errors (s.e. 0.005) with independent tokens one needs about 22,500
tokens for Q3-class and 3,300 for NVFP4-class artifacts; tokens within a document are correlated, so take 8 to 16 held-out documents
of 2K to 4K tokens (32K to 64K tokens), report the paired dNLL with a document-level (cluster) standard error, plus KL and top-1 as
`compare_reference.py` does. Hold out two kinds of text: own generations never used in training (thinking traces, code, tool
transcripts) and external text (news, multilingual prose, code from other repositories); a fine-tune that improves the first and
worsens the second has overfitted. Exclude the imatrix text and all training text. Add the behavioural metric Swift 1.5 is sold on:
mean thinking tokens and accuracy on a fixed set of 100 to 200 reasoning prompts under greedy decoding, for BF16, the artifact and
the fine-tuned artifact (Lotfi et al. show quantization inflates chain-of-thought length; ModelOpt's QAD run doubled output length on
one benchmark at unchanged score) [16d][17]. The `bf16.kld` resolution bounds KL at about 1e-4; fine.

## 5. Experiment plan

Ordered by information per hour. Each step is one artifact through the existing harness (`TeacherForcedQualityCudaIntegrationTest`
prefix 1281, steps 1279; `compare_reference.py`; the MTP acceptance pass; the speed benchmarks), on Swift-1.5 with its BF16 as
reference unless stated.

1. **Baselines and holdout (1 day, no training).** Build the holdout of section 4.5 (8 to 16 documents, two kinds) and produce
   Swift-1.5 BF16 reference logits for it with `llama-perplexity --kl-divergence-base`. Convert Swift-1.5 to `q3` and `q3-compressed`
   with the v1mix matrix. Record dNLL, KL, top-1 and thinking length for all four forms. Go: the holdout's s.e. is at or below 0.005
   nats for `q3`.
2. **Fake-quant parity (1 to 2 days).** A PyTorch module that reproduces `grouped.py` and `nvfp4.py` byte for byte on the `q3` and
   `nvfp4-compressed` recipes (export, load, teacher-forced report equal to the artifact's within kernel drift). Without this every
   later number is uninterpretable. Go: KL between the fake-quant model and the artifact below 1e-2 on the quality document.
3. **Tier 0a, one layer (hours).** Block-wise scale-and-norm training of block 10 (a GDN block) and block 11 (attention) against cached
   BF16 outputs, 1024 x 2048 own-generation tokens, 2 epochs; measure the block output MSE before and after. Go: MSE down by at least
   30% on held-out activations (a smaller drop means the scales are already near their optimum and Tier 1 is the first real step).
4. **Tier 0a, all blocks, then Tier 1 (1 to 2 days each on the 5070 Ti).** Produce `q3-ft-scales` and `q3-ft-blocks` artifacts.
   Measure on the holdout. Go for Tier 1 as the new `q3` recipe: dNLL improves by at least 0.03 nats with no thinking-length
   increase beyond 10% and no external-text regression. The literature expectation is a 0.06 to 0.09 nat gain; below 0.02 nats the
   method is not paying on this model and the chapter's expectation was wrong.
5. **P2E2-compatible fine-tune (same cost as step 4).** Repeat Tier 1 with codes clamped to -3..3, transcode to P2E2, measure size and
   quality. Go: `q3-compressed` at or below today's `q3` dNLL (+0.170) with the P2E2 ratio still at or below 0.85 of row-split bytes.
   This is the result that changes the product (1.56 GiB of KV at `q3` quality).
6. **Drafter fine-tune (1 to 2 days).** Dump post-final-norm hiddens and greedy tokens from the engine on 5 to 20 M tokens of own
   generations (chat, code, agentic, thinking on). Fine-tune the MTP block (NVFP4 fake-quant, unrolled to depth 4, cross-entropy on
   the target argmax plus a KL term), export, re-screen depth with `MtpDraftLengthScreenCudaIntegrationTest`. Measure tokens per
   verification on the chat, code and agentic sets of `MtpDraftConfidenceCudaIntegrationTest` (today 3.11 / 3.81 / 3.82 at MTP4) and
   the per-position conditional acceptance (today 0.85 at d1 falling to 0.70 to 0.78 on chat). Go: +0.2 tokens per verification at
   the shipped depth, or the deeper depth now winning at 32K; output hashes equal greedy on every prompt (they must; if not, the
   artifact writer is wrong, not the drafter). Repeat once with the Q3 base's hiddens to see whether a target-specific head is
   needed per form (MTP_VERIFIER found the base's precision did not move acceptance; the trained head may differ).
7. **Rented Tier 2 only if step 4 saturates.** One H100 for E2E scale tuning (2 h, under $10) over the Tier 1 artifact; if that moves
   dNLL by more than the holdout's s.e., consider 8xH100 QAD at 100 M to 1 B tokens ($50 to $650). Go/no-go: more than 0.02 nats
   over Tier 1 on the holdout with the thinking-length guard intact.
8. **Low-rank branch: last.** Only after steps 4 and 7, train rank-32 factors on the frozen Tier 1 backbone (block-wise, KL to the
   teacher), measure quality in fake-quant first; implement kernels only if the gain exceeds 0.03 nats, which is what 390 MiB buys
   as KV (about 22K tokens).

## 6. Risks and open questions

- **Eroding Swift 1.5's behaviour.** The fine-tune's property is fewer thinking tokens; a teacher-forced loss on the wrong text can
  undo it, and quantization itself lengthens reasoning [16d]; ModelOpt's QAD doubled output length on SciCode at unchanged score
  through non-terminating `<think>` loops [17]. Mitigations: Swift's own generations as data, the thinking-length guard in every gate,
  on-policy sequences generated by the deployed artifact, and the low learning rates the recipes use (1e-5 for weights, 1e-4 for
  scales: "adapt weights to quantization, not learn the task" [17]).
- **Overfitting the calibration text.** 8 to 17 M tokens through 65 blocks of 0.4 B parameters each is more than enough capacity to fit
  the cache; EfficientQAT uses a 64-sample validation split and early stopping per block; the two-kind holdout of section 4.5 is the
  external check. Scale-only training has far less capacity (400 M parameters, multiplicative) and is the safer first step.
- **The P2E2 ratio may move.** The compressed form's 0.815 ratio depends on most codes being -1, 0, +1. A fine-tune that spreads codes
  toward +-2, +-3 grows the file; one that sharpens them shrinks it. Measure the histogram after step 5 before claiming the savings.
- **Base against Swift checkpoints.** `q3` is a base-model artifact today; the recommendation to fine-tune Swift means the first `q3`
  number for Swift (step 1) may differ from +0.170. The two-checkpoint situation also means a trained MTP head belongs to one target.
- **The MTP head is the base's** (official card: "the published weights include the base model's MTP head"; third-party cards state it
  was not fine-tuned by UkisAI [23]). That is an argument for step 6, and an open question: how much of the 0.70 to 0.78 conditional
  acceptance beyond d1 is recursion mismatch (fixable by unrolled training) against Swift-versus-base drift (fixable by own data)
  against the drafter's capacity (not fixable without a larger draft). The repository's own measurement that the head is calibrated
  at d1 and overconfident by 0.05 to 0.10 from d2 on points at the first.
- **Hybrid-model evidence is thin.** No W2/W3 QAT paper covers a GDN hybrid; the 48 GDN blocks hold 22% of the `q3` bytes at 4.25 to
  5.25 bpw, and whether block-wise training lets them drop to Q4/Q3 is untested. The NVFP4 W4A4 PTQ result on a 27 B hybrid's recurrent
  half and GSQ's Qwen3.5/3.6 support are encouraging but not the same question [15][4].
- **Unretrieved numbers.** OmniQuant's and BitDistiller's per-model tables, PV-Tuning's GPU-hours, Medusa's training time, the QUASAR
  checkpoint's details and Qwen3-Next's MTP training notes could not be fetched in this session (search budget); the cost table does not
  depend on them.
- **Teacher logits on this machine.** The 54.7 GB BF16 teacher runs either on the CPU (llama.cpp; throughput unmeasured here) or by
  layer streaming over PCIe in PyTorch (about 6 to 7 hours per 16.8 M tokens, *estimate*). Both are one-time costs per data set.
- **Export path.** The converter has no entry for pre-rounded codes; `pack_codes_torch` and the scale writer exist, but a `--codes-from`
  recipe, manifest fields recording the training data, and a byte-equality test against the fake-quant model are new work.

## 7. References

1. Chen et al., *EfficientQAT: Efficient Quantization-Aware Training for Large Language Models*, ACL 2025, arXiv:2407.11062;
   repository https://github.com/OpenGVLab/EfficientQAT (model zoo, `examples/`, `quantize/block_ap.py`, `main_e2e_qp.py`). Training
   times from the paper's Table 7 as quoted in search results.
2. Liu et al. (Meta), *ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization*, arXiv:2502.02631; repository
   https://github.com/facebookresearch/ParetoQ; Graphcore Research summary https://graphcore-research.github.io/paretoq/.
3. Malinovskii et al., *PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression*, NeurIPS 2024, arXiv:2405.14852;
   Egiazarian et al., *AQLM*, arXiv:2401.06118; repository https://github.com/Vahe1994/AQLM (PV-tuned model table, `finetune.py`
   recipe, quantization-time notes).
4. Dadgarnia et al., *GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling*, arXiv:2604.18556;
   repository https://github.com/IST-DASLab/GSQ; RCO allocation arXiv:2605.00649 (ID from a search snippet, unverified);
   https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF.
5. Du et al., *BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation*, ACL 2024, arXiv:2402.10631; repository
   https://github.com/DD-DuDa/BitDistiller (`train/train.sh`, `data/generation`).
6. Liu et al., *LLM-QAT: Data-Free Quantization Aware Training for Large Language Models*, ACL Findings 2024, arXiv:2305.17888.
7. Shao et al., *OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models*, ICLR 2024, arXiv:2308.13137;
   repository https://github.com/OpenGVLab/OmniQuant (`scripts/Llama-2/*/w2a16g64.sh`).
8. Panferov et al., *QuEST: Stable Training of LLMs with 1-Bit Weights and Activations*, ICML 2025, arXiv:2502.05003; repository
   https://github.com/IST-DASLab/QuEST.
9. Li et al., *Norm Tweaking: High-performance Low-bit Quantization of Large Language Models*, AAAI 2024, arXiv:2309.02784.
10. Chen et al., *Scaling Law for Quantization-Aware Training*, arXiv:2505.14302; Dremov et al., *Compute-Optimal Quantization-Aware
    Training*, arXiv:2509.22935.
11. *Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens*, arXiv:2411.17691
    (title and ID from search results; contents not read).
12. Low-rank correction: (a) Yao et al., *ZeroQuant-V2* (LoRC), arXiv:2303.08302; (b) Zhang et al., *LQER: Low-Rank Quantization
    Error Reconstruction for LLMs*, ICML 2024, arXiv:2402.02446; (c) *QERA: an Analytical Framework for Quantization Error
    Reconstruction*, arXiv:2410.06040; (d) Saha et al., *Compressing Large Language Models using Low Rank and Low Precision
    Decomposition* (CALDERA), NeurIPS 2024, arXiv:2405.18886, repository https://github.com/pilancilab/caldera; (e) Li et al.,
    *SVDQuant*, ICLR 2025, arXiv:2411.05007 (Nunchaku repository).
13. Chen et al., *INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats*, ICML 2026, arXiv:2510.25602;
    repository https://github.com/ChenMnZ/INT_vs_FP (`quant/quant_func.py`).
14. NVIDIA, *Pretraining Large Language Models with NVFP4*, arXiv:2509.25149; *Dissecting Outlier Dynamics in LLM NVFP4 Pretraining*,
    arXiv:2602.02047.
15. *Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM*, arXiv:2609.04098
    (abstract-level findings from search results).
16. Reasoning models and data: (a) Liu et al., *Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models*, COLM
    2025, arXiv:2504.04823, repository https://github.com/ruikangliu/Quantized-Reasoning-Models; (b) *Attend to Your Own Thoughts*
    (AYOT), arXiv:2608.01078; (c) *LAQuant*, arXiv:2605.08755; (d) Lotfi et al., *Quantized Reasoning Models Think They Need to Think
    Longer, but They Do Not*, arXiv:2606.00206; (e) *Train Where the Quantized Model Goes: On-Policy Distillation for Low-Bit
    Reasoning*, arXiv:2609.26708; (f) *OnlineQAT: On-Policy Distillation for Ultra-Low-Bit Large Language Models*, arXiv:2610.09346.
    Also *Efficient Knowledge Distillation for LLMs: Offline Top-K Logits and a Fused Chunked KL Loss*, arXiv:2608.03796, and
    *Unifying Block-wise PTQ and Distillation-based QAT for Progressive Quantization toward 2-bit Instruction-Tuned LLMs*,
    arXiv:2506.09104.
17. NVIDIA TensorRT Model Optimizer, repository https://github.com/NVIDIA/TensorRT-Model-Optimizer: `examples/llm_qat/README.md` and
    `configs/train/{qat_nvfp4,qad_nvfp4,qad_scale_only,qad_with_learnt_amax}.yaml`, `modelopt_recipes/general/qad/*.yaml`,
    `examples/megatron_bridge/README.md` (QAD, LoRA QAD) and `examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/README.md`.
18. Drafters: (a) Li et al., *EAGLE*, arXiv:2401.15077, *EAGLE-2*, arXiv:2406.16858, repository https://github.com/SafeAILab/EAGLE
    (`eagle/train/main.py`, `eagle/traineagle3/`); (b) Li et al., *EAGLE-3*, NeurIPS 2025, arXiv:2503.01840; (c) SpecForge,
    https://github.com/sgl-project/SpecForge (`docs/recipes/qwen3.8-27b-dflash2-disaggregated.md`,
    `examples/configs/offline/colocated/qwen3-8b-eagle3-offline.yaml`, `examples/data_regeneration/`); LK loss arXiv:2602.23881;
    (d) *Scaling Laws for Speculative Decoding*, arXiv:2505.07858 (figures from search results); (e) *Attention Drift: What
    Autoregressive Speculative Decoding Models Learn*, arXiv:2605.09992 (figures from search results).
19. Zhang et al., *Learning Harmonized Representations for Speculative Sampling* (HASS), ICLR 2025, arXiv:2408.15766; repository
    https://github.com/HArmonizedSS/HASS (DeepSeek-R1 MTP table).
20. Cai et al., *Medusa*, arXiv:2401.10774; repository https://github.com/FasterDecoding/Medusa.
21. Chen, Liang, Liu, *DFlash: Block Diffusion for Flash Speculative Decoding*, arXiv:2602.06036; repository https://github.com/z-lab/dflash;
    *DFlash 2: Keep Drafting Parallel*, Inco AI blog, August 2026, https://inco.ai/blog/dflash2/ (not fetched).
22. (a) DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437, repository https://github.com/deepseek-ai/DeepSeek-V3; (b)
    *GLM-5*, arXiv:2602.15763 (MTP parameter sharing, from search results); *Nemotron 3 Super*, arXiv:2604.12374; DLoop,
    arXiv:2610.07659 (cited by `docs/MTP_DRAFT_CONFIDENCE.md`; not retrieved).
23. Swift 1.5: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b (official card), https://ukisai.com/swift-1-5-27b,
    https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF (source of `imatrix-swift15-v1mix.gguf`),
    https://huggingface.co/BroLaurens/Swift-1.5-Qwen3.8-27B-NVFP4-GGUF (third-party statement that the MTP head was not fine-tuned).
24. Third-party summaries of the ISTA GGUF results: https://interfaze.ai/models/ista-daslabqwen38-27b-gsq-rco-gguf,
    https://openclawdc.com/blog/qwen-3-8-27b-gsq-rco-quants/.
25. 2026 GPU prices: https://dev.to/moksh/h100-pricing-in-2026-coreweave-vs-lambda-vs-runpod-gpu-hour-by-gpu-hour-ei0,
    https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/,
    https://effectstory.com/en/articles/cloud-gpu-h100-b200-pricing-comparison-2026.
26. Euhedral repository at `perf/mtp-gated-depth`: `docs/MTP_CONTRACT.md`, `docs/MTP_VERIFIER.md`, `docs/MTP_DRAFT_CONFIDENCE.md`,
    `docs/DFLASH2.md`, `docs/QUALITY.md`, `docs/P2E2_FULL_RANGE.md`, `tools/euhedral_artifacts/grouped.py`, `nvfp4.py`, `recipes.py`.
