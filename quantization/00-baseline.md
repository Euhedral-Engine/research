# Baseline: what the four forms are today, and what any new technique must satisfy

This is the reference every other document in this folder measures against. Everything here is taken from the
Euhedral repository at branch `perf/mtp-gated-depth` (commit 269a890, PR 119, 2026-10-10): `README.md`, `tools/README.md`,
`docs/QUALITY.md`, `docs/COMPACT_Q3_REFERENCE.md`, `docs/COMPRESSED_Q3.md`, `docs/P2E2_FULL_RANGE.md`, `docs/NVFP4_NATIVE.md`,
`docs/NVFP4_COMPRESSED.md`, `docs/NVFP4_RESIDENCY.md`, `docs/PREFILL_MX.md`, `docs/ATTENTION_DECODE.md`, `docs/MTP_CONTRACT.md`,
`docs/MTP_VERIFIER.md`, `docs/nvidia/tensor-cores.md` and the converter in `tools/euhedral_artifacts/`.

## 1. The model

**Qwen3.8-27B** (Alibaba, released 2026-08-14, Apache-2.0; `config.json` architecture `Qwen3_5ForConditionalGeneration`). The
checkpoint actually served is **`ukisai/Swift-1.5-Qwen3.8-27B`**, a reasoning-efficiency fine-tune by UkisAI (reported: about 58%
fewer thinking tokens than the base at similar or slightly higher scores). It has the same architecture and tensor inventory, so it
goes through the same converter. Two notes for quantization: (a) two source checkpoints are in use (PR 119 at commit 269a890): the `q3` and `nvfp4`
artifacts are converted from the base Qwen3.8-27B with ISTA-DASLab's importance matrix, while `nvfp4-compressed` (the artifact
deployed by default) is converted from Swift-1.5 with UkisAI's own published importance matrix `imatrix-swift15-v1mix.gguf`;
each is measured against the BF16 of the checkpoint it came from; (b) whether Swift 1.5 re-trained or merely copied the MTP
layer is not documented in the repo.

The converter keeps the text tower and the MTP draft layer (785 objects) and drops the vision tower.

| Item | Value |
|---|---|
| Layers | 64: 16 full-attention layers (3, 7, 11, ..., 63) and 48 Gated DeltaNet (GDN) layers |
| Hidden / FFN intermediate | 5120 / 17408, SwiGLU |
| Vocabulary | 248,320 rows (tokenizer uses 248,077) |
| Attention | 24 query heads x 256, 4 KV heads x 256 (GQA 6:1), q/k RMSNorm, partial RoPE (64 rotary dims, theta 1e7), sigmoid output gate ("gate" rows packed with q in `q_proj`) |
| GDN | `in_proj_qkv` 10240 rows = q 2048 + k 2048 (16 key heads x 128) + v 6144 (48 value heads x 128); `in_proj_z` 6144; `in_proj_a`, `in_proj_b` 48 rows each; conv1d kernel 4; `A_log`, `dt_bias` 48; `out_proj` 5120 x 6144; recurrent state 48 x 128 x 128 FP32 per layer |
| MTP | `mtp.fc` 5120 x 10240 plus one full-attention decoder block; no own embedding or LM head; one layer applied recursively (depth 2 on Q3 artifacts, 4 on NVFP4 artifacts) |
| Norms | all `(1 + w)` RMSNorm; stored BF16 |

### Parameter inventory of what decode streams every token

Every decode token reads every weight of every layer plus the LM head (single sequence, no batching). The token embedding is a gather
and the draft head is read only by drafts, so they are listed separately.

| Family | Tensors | Shape (N x K) | Params total | q3 artifact format | GiB in `q3` | GiB in `nvfp4` (4.5 bpw) |
|---|---|---|---|---|---|---|
| `mlp/gate_up` | 64 | 34816 x 5120 | 11.41 B | Q3G64 (3.25 bpw) | 4.32 | 5.98 |
| `mlp/down` | 64 | 5120 x 17408 | 5.70 B | Q3G64 | 2.16 | 2.99 |
| `gdn/value_z` | 48 | 12288 x 5120 | 3.02 B | Q5G64 (5.25 bpw) | 1.85 | 1.58 |
| `gdn/output` | 48 | 5120 x 6144 | 1.51 B | Q3G64 | 0.57 | 0.79 |
| `gdn/query_key` | 48 | 4096 x 5120 | 1.01 B | Q4G64 (4.25 bpw) | 0.50 | 0.53 |
| `attention/query_key` | 16 | 7168 x 5120 | 0.59 B | Q4G64 | 0.29 | 0.31 |
| `attention/gate_value` | 16 | 7168 x 5120 | 0.59 B | Q5G64 | 0.36 | 0.31 |
| `attention/output` | 16 | 5120 x 6144 | 0.50 B | Q3G64 | 0.19 | 0.26 |
| `text/output_head` (LM head) | 1 | 248320 x 5120 | 1.27 B | Q3G64 | 0.48 | 0.67 |
| **Streamed per token** | | | **25.6 B** | | **10.7** | **13.4** |
| `text/token_embedding` (gather) | 1 | 248320 x 5120 | 1.27 B | Q3G64 in every artifact | 0.48 | 0.48 |
| `text/draft_head` (drafts only) | 1 | 131072 x 5120 | 0.67 B | Q3 in q3 artifacts, NVFP4 in nvfp4 artifacts | 0.25 | 0.35 |
| `mtp/*` projections (drafts only) | 5 | see MTP_CONTRACT | 0.42 B | NVFP4 in every artifact | 0.22 | 0.22 |
| BF16/FP32 small objects (norms, conv, a/b projections) | 456 | | about 0.03 B | BF16 / FP32 | 0.05 | 0.05 |

Rule of thumb: **the 25.6 B streamed parameters cost 2.98 GiB per bit per weight.** A uniform format at 2.0 / 2.5 / 3.0 / 3.5 / 4.0 /
4.5 bpw streams 5.96 / 7.45 / 8.94 / 10.4 / 11.9 / 13.4 GiB per token. Q3 one-row decode moves its 10.7 GiB in 16.0 ms, about
720 GB/s against a measured DRAM rate of 855 GB/s (896 GB/s theoretical), so at one row the token time is almost exactly bytes
over bandwidth. The MLP is 67% of the streamed bytes in the `q3` artifact and 67% in `nvfp4`; the GDN input projections (`value_z`,
`query_key`) are 22% of the `q3` artifact because they sit at 5.25 and 4.25 bpw.

## 2. The four forms

An artifact is one `.edrl` file (EDRL container version 2: each tensor descriptor carries source dtype, storage format and layout).
The engine reads the format and layout of every tensor and chooses kernels, speculative depth and residency itself; there are no
user options for precision or kernels.

| Artifact | Source checkpoint and importance matrix | Weights | Layout | File | bpw of the main tensors | Mean NLL | NLL vs its BF16 (s.e.) | KL(BF16 ‖ artifact) | Top-1 agreement | Decode tok/s 4K / 32K / 64K / 128K |
|---|---|---|---|---|---|---|---|---|---|---|
| BF16 reference (llama.cpp, CPU) | base Qwen3.8-27B: perplexity 8.39, NLL 2.128; Swift-1.5: perplexity 8.51, NLL 2.142 | | | 54.7 GB | 16 | | | | | |
| `q3` | base; ISTA-DASLab `imatrix-qwen3.8-27b.gguf` | Q3G64_F16S (+Q4/Q5 inputs), calibrated | `row-split-k128-v1` | 11.72 GiB | 3.25 | 2.298 | +0.170 (0.021) | 0.293 | 76.2% | 109 / 105 / 93 / 67 |
| `q3-compressed` | base; none (round-to-nearest) | the same Q3 tensors, **round-to-nearest**, transcoded losslessly | `row-split-p2e2-v1` | 10.16 GiB | about 2.65 (Q3 planes at 81.5%) | 2.403 | +0.275 (0.025) | 0.400 | 70.5% | 95 / 92 / 82 / 64 |
| `nvfp4` | base; ISTA-DASLab `imatrix-qwen3.8-27b.gguf` | NVFP4 (E2M1 codes, E4M3 scale per 16, FP32 per tensor), calibrated | `row-split-k128-v1` | 14.52 GiB | 4.5 | 2.156 | +0.029 (0.008) | 0.147 | 88.2% | 116 / 101 / 72 / 30 |
| `nvfp4-compressed` | **Swift-1.5**; UkisAI `imatrix-swift15-v1mix.gguf` | NVFP4-SD4 (scales from a 16-entry E4M3 table per tensor), calibrated | `row-split-k128-sd4-v1` | 13.77 GiB | 4.25 | 2.170 | +0.028 (0.009) | 0.155 | 89.4% | 146 / 102 / 88 / 52 |

Quality is measured on 1279 teacher-forced tokens after a 1281-token prefix of one fixed document (`chat-corpus-v1-document.md`),
against the BF16 of the checkpoint each artifact was converted from, run in llama.cpp on the CPU (`docs/QUALITY.md`). Standard
errors are of the paired NLL difference. SD4 against plain NVFP4, both converted from the base model with the same matrix: NLL
+0.005 (0.009), KL 0.037, top-1 agreement 89.8%. The relaxed-kernel drift floor on the same artifacts is KL 3.4e-3 to 4.7e-3 with
top-1 agreement 97.3 to 97.9%: **quantization error is 30 to 85 times the kernel noise floor.** Quantization, not kernels, is
where the quality headroom is.

### Formats in detail

| Format | Codes | Group | Scale | bpw | Where used |
|---|---|---|---|---|---|
| Q3G64_F16S | signed 3-bit, -4..3 (RTN never produces -4; calibrated search does) | 64 along K | binary16 per group | 3.25 | MLP, attention/GDN output, embedding, LM head, draft head (q3 artifacts) |
| Q4G64_F16S | -8..7 | 64 | binary16 | 4.25 | `attention/query_key`, `gdn/query_key` |
| Q5G64_F16S | -16..15 (low nibble plane + fifth-bit plane) | 64 | binary16 | 5.25 | `attention/gate_value`, `gdn/value_z` |
| NVFP4 | E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6, sign) | 16 along K | E4M3 (7 bits used, 0..448) per block, FP32 per tensor = amax / (6 x 448) | 4.5 | nvfp4 artifacts, MTP layer in all artifacts |
| NVFP4-SD4 | E2M1 | 16 | 4-bit index into a 16-entry E4M3 table per tensor, FP32 per tensor | 4.25 | base-model tensors of `nvfp4-compressed`; drafting stack stays plain NVFP4 (measured better acceptance) |
| P2E2 | Q3 codes entropy coded: 2-bit primary symbol (-1, 0, +1, BIG) plus a 2-bit payload unit per BIG code (-3, -2, +2, +3); scales byte for byte | 64 | binary16 | about 2.65 (Q3 planes at 81.5% of row-split bytes) | `q3-compressed`; cannot represent -4, so it holds the RTN Q3 |
| KV cache | E2M1 | 16 along the 256-wide head after a normalized Sylvester H256 Hadamard rotation of K and V (queries rotated to match) | E4M3 | 4.5 (144 B per head row: 128 code bytes + 16 scales) | 16 attention layers x 4 KV heads: 18 KiB per token |

Layout `row-split-k128-v1`: rows padded to a multiple of 128 on K, planes (codes, high bits, scales) each on a 256-byte boundary; the
loader uploads bytes as they are and never repacks. All multi-byte values little endian.

## 3. How the weights are rounded today (`--imatrix`)

- **Importance** = llama.cpp's imatrix: the mean square of each input column over a calibration text (`in_sum2 / counts`), i.e. a
  diagonal proxy of the layer's input Hessian. Two matrices are in use: ISTA-DASLab's public `imatrix-qwen3.8-27b.gguf` (from the
  Qwen3.8-27B GSQ-RCO release; 1000 chunks of 4096 tokens, base model) for the `q3` and `nvfp4` conversions of the base
  checkpoint, and UkisAI's `imatrix-swift15-v1mix.gguf` (from `ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF`; 309 chunks of 4096
  tokens of web, reasoning, multilingual and code text) for the `nvfp4-compressed` conversion of Swift-1.5. Objects neither matrix covers (token embedding, LM head, draft head, MTP
  layer) are searched with uniform weights.
- **Q3/Q4/Q5 (`grouped.py:search_group_codes_torch`)**: per group of 64, candidates are the RTN scale `amax/qmax`, and for each end
  `e` in {qmax, -qmin} and each of 19 steps `s` in -0.9..+0.9: scale = `amax/(e+s)`, round, then refit the scale by weighted least
  squares over the codes it produced, round the scale to binary16, requantize. The group keeps the candidate with the least weighted
  squared error (39 candidates). No cross-column error feedback, no clipping search beyond this, no per-channel pre-scaling.
- **NVFP4 (`nvfp4.py:search_nvfp4_rows_torch`)**: per-tensor global scale `amax/(6 x 448)`; each 16-block's E4M3 scale code is
  searched from 6 codes below to 2 above the round-to-nearest code under weighted squared error; codes round to nearest under the
  chosen scale. Going below the RTN scale clips the block's largest values.
- **SD4**: a per-tensor table of 16 E4M3 codes chosen by dynamic programming over the per-block cost of every candidate scale (blocks
  grouped by their RTN code); each block then takes its least-error table entry; with `--imatrix` a block may take an entry up to 6
  codes below its RTN scale. Measured against plain NVFP4: NLL +0.001 (within noise), KL 0.058 for the round-to-nearest pair; +0.005 (0.009), KL 0.037
  for the calibrated pair, both from the base checkpoint.
- **Drafting stack**: MTP layer is plain NVFP4 in every artifact (Q3 drafts accepted fewer tokens per verification); draft head follows
  the artifact's quantization.
- **No fine-tuning, no rotations, no GPTQ-style error feedback, no activation-aware channel scaling, no mixed per-layer bit
  allocation beyond the fixed Q4/Q5 choice for the attention and GDN input projections.**

The converter runs in PyTorch on the GPU (default 4 worker processes), a minute or two for `q3` and `nvfp4`, about 6 minutes for the
calibrated SD4 pass, 14 minutes for RTN SD4. Calibrated passes keep a whole tensor resident on the device (the largest, `gate_up`, is
356 MB in BF16). The machine is an RTX 5070 Ti (16 GB) with an i9-14900K; host RAM is enough that 2+ GiB of weights are pinned,
but `--jobs 16` already fills swap. A llama.cpp tree (commit 67a17c17c), a BF16 GGUF, `llama-imatrix` and `llama-perplexity
--kl-divergence` are part of the user's toolchain.

## 4. Where the bytes go: memory budget

| Item | Size |
|---|---|
| Device total | 15.56 GiB |
| Desktop (Xorg, browser) and the CUDA context before loading | 0.75 to 0.95 GiB, drifting by hundreds of MiB |
| Fixed per-sequence state and workspaces (GDN state 0.15 GiB, prefill, verify, scratch) | 0.51 GiB; the planner reserves 1280 MiB |
| KV cache (NVFP4 pages, 16 layers) | 18 KiB per token: 0.56 GiB at 32K, 1.1 GiB at 64K, 2.25 GiB at 128K |
| Speculative GDN checkpoint/records (ReplaySSM) | about 157 MiB |
| MTP layer + draft head | 0.57 GiB (NVFP4), never host-backed |
| `q3` payload | 11.72 GiB: runs 128K with nothing host-backed |
| `nvfp4` payload | 14.52 GiB: host-backs 1.0 / 1.6 / 2.8 GiB at 32K / 64K / 128K |
| `nvfp4-compressed` payload | 13.77 GiB: host-backs 0.27 / 0.84 / 1.97 GiB at 32K / 64K / 128K |

Host-backed weights are staged over PCIe 5.0 x16 at 44 GB/s (huge-page pinned memory; 25 GB/s with 4 KiB pages under the IOMMU).
Decode has no reuse window: every staged byte is read once per verification, so each 128 MiB host-backed costs about 2 to 2.5% of
decode speed, and anything beyond what the GPU's own step time hides runs at the copy rate. Prefill hides staging entirely.

## 5. Execution routes the format has to serve

| Rows | Q3 / Q4 / Q5 tensors | NVFP4 / SD4 tensors |
|---|---|---|
| 1 (decode) | contiguous FP32 FMA GEMV: a lane owns 32 codes (3 words), 7 instructions per code pair, FP32 accumulate, binary16 scale applied once per half group | `decode_rows<1>` on BF16 tensor cores (`mma.m16n8k16`, FP32 accumulate): a 256-entry shared table turns a code byte into two BF16 values, scale applied with `__hmul2`; weight x scale has at most 6 significant bits so every product is exact |
| 2 to 8 (verification, drafts, catch-up) | multi-row twins repeating the one-row FMA sequence per token row while streaming the weights once; ALU- and register-bound from 4 rows (about 600 GB/s, 223-register cliff) | the same decode kernels with the token rows as MMA columns; 8 rows cost at most 9% more than 1 |
| 9 and more (prefill) | block-scaled MXFP8 (`kind::mxf8f6f4`, E4M3 x E4M3, UE8M0 activation scale per 32, unit weight scale): Q codes expanded exactly to E4M3 in producer warps; BF16 activations as two exact E4M3 terms (hi + lo); binary16 group scale applied in FP32 after each group; 408 TFLOPS peak, 204 effective for two terms, 126 achieved on gate/up | native NVFP4 OMMA (`kind::mxf4nvf4`, 815 TFLOPS peak): weights consumed in place, BF16 activations quantized to two NVFP4 terms (value + residual); tiles reach 600 to 700 TFLOPS; one activation term was rejected (0.88% -> 9.5% output RMS error, double the KL) |
| P2E2 tensors | fused decode for 1 to 8 rows (about 13% slower than `q3` at decode); prefill expands each tensor to row-split once per launch at DRAM rate (about 0.25 ms per layer per chunk) | |
| Exact oracle (tests only) | scalar reference kernels in FP32 | scalar reference |

Row-exact verification is a correctness contract: MTP and DFlash2 speculation produce *exactly* the tokens of ordinary greedy
decoding, and the KV and GDN state after a verification is bit for bit the one-row state (`SpeculativeVerifyCudaIntegrationTest`,
`SpeculativeDecodeCudaIntegrationTest`). A relaxed verifier on native FP4 kernels changed every generated text and was rejected.

### The hardware (RTX 5070 Ti, sm_120, 70 SMs; measured in `docs/nvidia/tensor-cores.md`)

| Kind | Dense TFLOPS | Notes |
|---|---|---|
| BF16/FP16 `mma.sync` with FP32 accumulate | 102 | FP16 accumulate: 204 |
| FP8 E4M3 (`kind::f8f6f4`, FP32 accumulate) | 205 | FP16 accumulate: 408 |
| FP6 E3M2/E2M3, FP4 E2M1 inside `kind::f8f6f4` | 204 | 8-bit containers; saves bytes, not FLOPs |
| MXFP8 block-scaled (`kind::mxf8f6f4`, UE8M0 per 32, FP32 accumulate) | 408 | the full-rate FP32-accumulate path used by the Q3 prefill route |
| MXFP4 / NVFP4 (`kind::mxf4` / `mxf4nvf4`, E2M1 x E2M1, UE8M0 per 32 or UE4M3 per 16) | 815 | 2:4 sparse: 1621 |
| INT8 IMMA | 408 | exact integer accumulate |
| INT4 / binary MMA | emulated, 42 | **do not use** |

Hard facts that bound format design: NVFP4 weights can only be multiplied natively against **FP4 activations** (`mxf8f6f4` accepts
only UE8M0 scales per 32, so E4M3-per-16 weight scales cannot meet FP8 or BF16 activations in hardware); FP32-accumulating tensor
cores truncate (24-bit significand) rather than round, so an exact twin must issue the same instruction sequence as its oracle;
a tensor-core row never depends on the other rows of the tile, which is what makes multi-row MMA kernels row-exact by construction.

## 6. Constraints any new format or rounding method must satisfy

1. **Row-exact 1 to 8 row decode.** The format needs a one-row kernel and multi-row twins whose every row is bitwise the one-row
   result. Two proven ways: repeat the one-row FMA sequence per row (costly in registers at 4+ rows), or put rows in MMA columns with
   products that are exact in BF16/FP16 (requires `code x scale` to have at most 8 significant bits in BF16, 11 in FP16). A codebook
   whose entries are arbitrary floats breaks the second way unless the entries are chosen BF16-exact.
2. **A prefill route at 9+ rows that does not fall back to the 102-TFLOPS BF16 path.** Today's routes need codes that expand exactly
   to E4M3 (small integers) or that *are* E2M1 x E4M3. A non-uniform codebook that is not E4M3-representable needs a two-term E4M3
   decomposition of the weights (halving the MXFP8 rate) or a BF16 fallback (4x slower).
3. **Decode instruction budget.** The Q3 GEMV decodes about 2.85e12 codes/s from L2, only 1.4x its DRAM-fed rate: a format may spend
   at most about 1.3x the instructions per code of Q3 before one-row decode slows, and the 4-row twins are already ALU-bound.
   Table lookups in shared memory are cheap; variable-length decoding that must place each value at its code position costs about
   11 instructions per pair against 7 (P2E2: -13% decode for -18.5% bytes).
4. **Determinism.** Every kernel is deterministic and replayed from captured CUDA graphs; benchmarks compare output hashes.
5. **Offline conversion only.** The converter (PyTorch, GPU, minutes) does all rounding; the loader uploads bytes as they are. New
   formats and layouts get their own EDRL format and layout ordinals; existing artifacts keep loading.
6. **The artifact decides everything.** No user-facing precision options: a new form is a new artifact recipe, and the engine's
   `ArtifactProfile` chooses depth and residency from the tensors it finds.
7. **Memory is the product.** Bytes per token set decode speed; resident bytes set the longest context (KV is 18 KiB per token). A
   smaller `q3`-quality artifact buys context; a `q3`-sized NVFP4-quality artifact buys quality. Prefill is not the bottleneck at
   short context but TTFT at 128K is 95 to 133 s.
8. **Drafting precision is a separate knob.** The MTP layer and draft head only propose; their precision trades acceptance rate (and
   so speed) against 0.57 GiB of residency, never correctness.

## 7. Measured and rejected already (do not re-propose without a new argument)

- SD4 with the 16 most frequent scales: clipped outlier blocks by up to 8x, NLL 4.32 against 2.23. The DP table that reaches the
  largest scales is what works.
- SD4 allowing scales 2 codes below RTN: squared error down, NLL -0.003 (noise), largest weights clipped by up to 14%.
- One activation term for native NVFP4 prefill: output RMS error 9.5% against 0.88%, KL doubled.
- Native FP4 (relaxed) verification: 4% faster, changed every output. Row exactness stays.
- GDeflate / nvCOMP ANS over Q3 codes: 0.885 of row-split (P2E2 alone 0.815), decompression 12x slower than P2E2 expansion; scale
  plane compresses to 0.57 but nothing can read it in place.
- Decoding P2E2 inside the prefill GEMM's weight producer: 0.5 to 0.6x the row-split kernels. Prefill expands first.
- Deeper host-staging rings; per-tensor pinned allocations (fall back to 4 KiB pages: 25 to 30 GB/s).
- Q3 MTP layer: fewer accepted tokens than NVFP4; `q3` artifacts carry an NVFP4 MTP layer.
- NVFP4 lossless coding: codes carry 3.90 of 4 bits, scales 3.63 of 8; only the scales compress (0.75 GiB), which SD4 takes.

## 8. Open items the repository names

- P2E2 cannot store code -4, so `q3-compressed` is stuck at RTN quality (+0.275 against +0.170 NLL); `docs/P2E2_FULL_RANGE.md`
  lists what a full-range layout needs. The fallback named there: calibrated Q3 restricted to -3..3 for both artifacts.
- "Code-bit reduction below 4 bits per value is the only route to large further savings" for NVFP4.
- Q3 scale plane (about 0.25 GiB) compresses 43% but needs a second table per tensor in every kernel; Q4/Q5 escape codes about
  0.1 GiB each.
- 4-row verification GEMVs run at about 600 GB/s, two thirds of peak.
- 0.51 GiB of fixed per-sequence workspaces (about 28K tokens of KV).
- 2:4 sparsity with FP4 would reach about 3 to 3.5 bpw with hardware MMAs at 1621 TFLOPS, but pruning a dense 27B model to 2:4
  needs retraining ("research, not a quick win").

## 9. Measurement harness to reuse

- Teacher-forced reports: `TeacherForcedQualityCudaIntegrationTest` (prefix 1281, 1279 steps) and `tools/compare_reference.py`
  against `llama-perplexity --kl-divergence-base bf16.kld` (mean NLL, paired dNLL with s.e., KL, top-1). Resolution of the stored
  reference log-probabilities bounds KL at about 1e-4.
- Pairwise artifact comparison: `tools/compare_teacher_forced.py`.
- Kernel drift: `RelaxedNumericsDriftCudaIntegrationTest`.
- Exactness: `SpeculativeVerifyCudaIntegrationTest`, `SpeculativeDecodeCudaIntegrationTest`; benchmark rows record `outputSha256`.
- Speed: `benchmark/configs/*.json` through `./gradlew :benchmark:run`, chat corpus, 4K / 32K / 64K / 128K prompts.
- Acceptance: MTP tokens per verification (deterministic for a fixed corpus; one pass settles it).

A single 1279-token document gives s.e. of 0.008 to 0.025 nats on dNLL: differences below about 0.02 nats (NVFP4) or 0.05 nats (Q3)
are not resolvable with it. Any proposal that claims gains of that size needs a larger held-out evaluation.
