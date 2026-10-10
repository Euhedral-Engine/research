# 06. KV cache and recurrent state: 3-bit and 2-bit KV, and whether the GDN state can shrink

Scope: the 16 attention layers' key/value cache (today E2M1 codes with one E4M3 scale per 16 of the 256 head dimensions after a
normalized Sylvester H256 rotation; 144 bytes per head row; 18 KiB per token) and the 48 Gated DeltaNet layers' recurrent state
(48 x 128 x 128 FP32 per layer, 0.15 GiB per sequence). Weight formats belong to other chapters. Engine facts come from
`docs/ATTENTION_DECODE.md`, `docs/PREFILL_MX.md`, `docs/MTP_CONTRACT.md`, `docs/MTP_VERIFIER.md`, `docs/PREFIX_CACHE.md`,
`docs/NVFP4_RESIDENCY.md`, `docs/DFLASH2.md`, `native/src/attention/*.cuh` and `native/src/gdn/recurrence.cuh`. Published numbers
carry their source; anything marked *estimate* is this chapter's own arithmetic or simulation.

## 1. Summary

The engine's KV format already belongs to the family the literature settled on for 4-bit caches: a per-head Hadamard rotation followed
by small-block scaled quantization (QuaRot's and SpinQuant's online key rotation; NVIDIA's NVFP4 KV cache with E4M3 scales per 16). The
rotation removes the per-channel key structure that KIVI, KVQuant and QServe scale around, so the only open question is how many code
bits the rotated coordinates need. Published long-context evidence supports 4 bits as near-lossless (NVFP4 KV under 1% on RULER 64K,
FP8 at 97 to 98% of baseline at 128K, KIVI-4 within noise on LongBench), 3 bits as a small measurable loss at 7B scale (QuaRot KV3
+0.21 WikiText perplexity, KVQuant 3-bit +0.25 to +0.32 on 128K to 1M models with 256K retrieval retained), and 2 bits per element as a
real loss on retrieval and reasoning (KIVI-2 loses 8.5 points on Mistral passage retrieval; the May 2026 vLLM study of TurboQuant's
3-bit mode drops AIME25 from 0.683 to 0.443 and 128K MRCR from 1.00 to 0.72). Verdict: build and measure keys and values at 3.5 bits per
element (signed 3-bit codes times the existing E4M3 scale per 16; 112-byte rows; 14 KiB per token), then values at 2.5 bits (80-byte
rows; 12 KiB per token with 3.5-bit keys); do not plan on 2.5-bit keys or a uniform 2-bit cache for a reasoning fine-tune at 64K to 128K.
Both grids keep every code-times-scale product exact in FP16, so the row-exact verifier, the prefix cache and the exact-product lookup
kernels survive with new tables rather than a redesign. The payoff is memory, not arithmetic: at 128K the cache falls from 2.25 GiB to
1.75 or 1.5 GiB, which for `nvfp4-compressed` cuts host-backed weights from 1.97 to 1.47 or 1.22 GiB and should lift 128K decode from
the measured 52 tok/s to roughly 70 to 80 tok/s (estimate), while `q3`, which host-backs nothing, gains only the attention bytes
(about +4 to +12% at 128K, nothing below 32K). The cost is a new codec, new tables in four attention kernels and their exact twins, page
sizing in the planner and prefix cache, a KV-format field in the artifact profile, and a long-context quality harness the repository
does not have. The GDN state should stay FP32: BF16 storage saves at most 1.2% of a decode token and 72 MiB, every reference keeps it
FP32, three engines report drift from BF16 state or gate rounding, and slow heads amplify per-step rounding by 1/sqrt(1 - alpha^2)
(estimate).

## 2. Mechanism

### 2.1 What the engine does today

`qk_norm_rope.cuh` applies the per-head (1 + w) RMSNorm and RoPE to the first 64 of 256 dimensions (pairs d and d + 32);
`kv_append.cuh` then quantizes the BF16 key and value rows: `hadamard256` (FP32 butterflies, entries plus or minus 1/16), and per 16
coordinates an E4M3 scale `rn(amax / 6)` clamped to [2^-9, 448] with E2M1 codes by round-to-nearest with saturation (`nvfp4_kv.cuh`).
Keys are quantized post-norm and post-RoPE. The quantizer depends only on its row, so appends of verified rows equal one-row appends.
Decode (`nvfp4_decode_gqa.cuh`) builds the QK A fragments straight from the raw 144-byte rows: a 256-entry shared table maps a byte of
two codes to an FP16 pair, one `__hmul2` by the hardware-converted E4M3 scale follows, and because an E2M1 x E4M3 product has at most 6
significant bits it is exact in FP16; queries and probabilities enter as hi + lo FP16 parts into `mma.m16n8k16` with FP32 accumulation.
The 16-wide scale group coincides with the k16 step of the MMA, which is what lets one scale multiply serve one fragment. Prefill
(`nvfp4_prefill_fa2.cuh`) expands 32-key tiles into FP16 shared rows through the same table and runs FP16 MMAs (60 TFLOPS at 32K). The
GDN recurrence (`gdn/recurrence.cuh`) keeps the state in FP32 registers, computes the delta rule in FP32 with alpha and beta already
FP32, and writes it back in FP32.

### 2.2 The schemes

| Scheme | Grid | Scaling axis | Rotation | Correction / extras | Bits per element |
|---|---|---|---|---|---|
| KIVI [1] | 2-bit (or 4-bit) asymmetric uniform, zero point | keys per channel over 32 tokens; values per token, groups of 32 | none | 32 to 128 recent tokens kept in FP16 | 2 + scale and zero per 32, plus the window |
| KVQuant [2] | non-uniform (k-means, sensitivity weighted), 2 to 4 bits | keys per channel, pre-RoPE; values per token | none | 1% outliers per vector in FP16; first 5 tokens FP16 | about 3.3 at "3-bit, 1%" (averaged at 128K) |
| QuaRot [3] | asymmetric uniform, 4 and 3 bits, clip 0.95 | per head or per token | online Hadamard on q and k after RoPE; V rotation fused into v_proj / o_proj | none | 4 or 3 + scale and zero per 128 |
| SpinQuant [4] | as QuaRot for KV | per head | learned weight rotations; online Hadamard R3 for keys | none | 4 |
| Atom [5], QServe [6] | INT4 asymmetric; KV4 with SmoothAttention | per token or head | none; QServe moves key outliers into queries | none | 4 + scale and zero |
| TurboQuant [7] | Lloyd-Max Gaussian codebook at b - 1 bits plus 1-bit QJL sign of the residual | the row norm (FP16 per 128 or 256) | random orthogonal or Walsh-Hadamard | QJL gives an unbiased inner product; 2.5 / 3.5-bit results split outlier and regular channels | 2.5 or 3.5 |
| PolarQuant, Han et al. [8] | angles of the rotated vector | none per block | random rotation | none | about 3.9 at the reported 4.2x |
| PolarQuant, key cache [9] | 2-D sub-vectors as radius and angle, post-RoPE | per pair | none | QK product by table lookup | 3 to 4 |
| RotateKV [10] | 2-bit uniform | per channel after rotation | outlier-aware (channel-reordered) Walsh-Hadamard, pre-RoPE, grouped heads | sink tokens found from massive activations | 2 |
| ZipCache [11], MiniKV [12] | 4-bit salient tokens, 2-bit others; 2-bit plus eviction | per token | none | saliency / heavy hitters | mixed |
| FP8 E4M3 / E5M2, vLLM [23] | 8-bit float | per tensor or per KV head, calibrated | none | none | 8 |
| NVFP4 KV, NVIDIA and SGLang [25][26] | E2M1 | E4M3 per 16 plus FP32 per tensor | none | none | 4.5 |
| Euhedral today | E2M1 | E4M3 per 16, no global scale | fixed H256 per head, online, post-RoPE | none | 4.5 |

### 2.3 Keys versus values, rotation, RoPE and QK-norm

Keys are the hard half. QServe measures fixed outlier channels per head about 10x the typical magnitude [6]; KVQuant finds them more
consistent before RoPE, and pre-RoPE per-channel scaling improves 3-bit LLaMA-7B perplexity by 0.82 [2]; KIVI's ablation shows that
per-channel keys with per-token values is the only 2-bit axis assignment that does not collapse [1]; QuaRot's KV ablation gives +0.03
perplexity for K4V3 against +0.18 for K3V4 [3]; "More for Keys, Less for Values" has K4V2 recovering about 94% of baseline where K2V4
loses up to 30 points, and RateQuant finds key distortion 3 to 4x value distortion with an optimal split near 2.85 / 2.15 bits at a
2.5-bit average [31]. Values have no channel structure; per-token scaling suffices.

A per-head Hadamard changes the problem: every rotated coordinate is a plus or minus 1/16 weighted sum of all 256 channels, so a fixed
outlier channel becomes a near-constant offset spread over the row and the per-coordinate distribution is close to Gaussian (QuaRot's
argument [3]; in the simulation of 2.4 the two-outlier-channel rows quantize slightly better than Gaussian rows). This is why QuaRot
and SpinQuant quantize rotated keys per head with one scale, and why TurboQuant uses one Gaussian codebook for every coordinate. The
engine already has this, so per-channel key scaling, pre-RoPE key statistics and dense-and-sparse outlier lists buy nothing here. What
remains after rotation is per-token structure, which the per-16 E4M3 scale handles row by row.

RoPE and the rotation do not commute, so a pre-RoPE cache needs a per-key RoPE at read time (KVQuant's fused kernels) or a
position-dependent query rotation per key; both break the one-table exact-product lookup. With 64 of 256 dimensions rotated, 75% of a
key is position-free anyway, and RoPE is orthogonal, so post-RoPE group statistics match pre-RoPE ones up to the mixing within the 32
rotated pairs. Pre-RoPE placement only matters when the rotation is fused into `W_k` (RotateKV); the engine rotates online. The
varjoranta vLLM port handles partial-rotary Qwen3.6 models with a block-diagonal Walsh-Hadamard that keeps the RoPE prefix and the
content suffix apart [34]; the engine's full H256 is simpler and consistent because queries are rotated identically.

QK-norm matters too. A key row is `x / rms(x) * (1 + w)`, so its channel magnitudes are set by the norm weights and its norm is bounded
by `16 * max(1 + w_d)`: this model's "outlier channels" are the large entries of `key_norm`, known at conversion time, and no key token
can have an anomalous norm. Community data point the same way: Qwen2.5-7B (no QK-norm) has K_max/K_med above 250x in some layers [30]
and a q4_0 key cache in llama.cpp takes its WikiText perplexity from 7.96 to about 1590 (user report) [28], while Qwen3-8B's key variance
is uniform across layers except layer 0 [28] and Qwen 3.6 and Gemma 4 keep KL below 0.04 with q8_0 and 0.087 to 0.117 with q4_0 caches
[29]. Values are not normalized, so sink tokens (first token, massive-activation tokens [32]) can carry large value rows; per-row
scaling absorbs the magnitude and the rotation flattens the within-row range. Keeping the first tokens at higher precision (KVQuant,
KVSink [32]) is cheap insurance a per-page flag could express if the long-context test asks for it.

### 2.4 Which lower-bit grids keep the products exact

Constraint 1 of `00-baseline.md` needs `code x scale` within 11 significant bits for FP16. With an E4M3 scale (4 significant bits):

- Signed integers up to 15 in magnitude: at most 8 bits, exact in FP16 and BF16. This covers INT4 (-8..7, or midrise plus or minus
  1..15), INT3 (-4..3, or midrise plus or minus 1, 3, 5, 7) and INT2 (midrise plus or minus 1, 3). Ranges stay inside FP16
  (7 x 448 = 3136; 1 x 2^-9).
- E2M1 (today): at most 6 bits.
- Lloyd-Max or other non-uniform centroids: exact only if rounded to at most 7 significant bits (a 2^-7 perturbation, negligible for
  MSE). The 2-bit Gaussian codebook is nearly the midrise grid already (0.4528 and 1.5104 sigma against 0.5 and 1.5 at the optimal
  uniform step).
- Asymmetric grids with a zero point (KIVI, QuaRot, Atom): `s * c - z` is not an exact FP16 product; the zero point has to be factored
  out as `- z_g * sum_d q_d` per group and added in FP32 outside the MMA. Possible, but it costs a zero per group and a second
  accumulation path; symmetric grids avoid it.
- TurboQuant's QJL term is a sign vector times a projected query, itself an exact-product MMA, but the community ports and the vLLM
  study find that QJL adds variance that softmax amplifies (tonbistudio: 0/27 generation tests with QJL against 18/18 without; vLLM
  shipped only the no-QJL `_nc` variants) [34].

Own simulation (estimate; `scratchpad/sim/kvsim.py`): 20,000 rows of 256 i.i.d. channels rotated by the exact H256, quantized per 16
with E4M3-rounded scales; relative RMS error per element and relative RMS error of `q . k` against Gaussian queries agree to three
digits. Laplacian, Student-t (nu = 4) and mixed-norm rows gave the same numbers within 0.01; rows with two fixed outlier channels
(20x, 8x) quantize slightly better after rotation (INT3 0.150 instead of 0.171).

| Grid (scale) | bpe | Relative RMS error | SQNR (dB) | Error relative to today |
|---|---|---|---|---|
| E2M1, E4M3 per 16 (today) | 4.5 | 0.095 | 20.4 | 1.00 |
| INT4 midrise plus or minus 1..15, E4M3 per 16 | 4.5 | 0.081 | 21.9 | 0.85 |
| INT3 midrise plus or minus 1, 3, 5, 7, E4M3 per 16 | 3.5 | 0.171 | 15.3 | 1.80 |
| INT3 midtread -4..3 (scale amax/3), E4M3 per 16 | 3.5 | 0.199 | 14.0 | 2.10 |
| INT2 midrise plus or minus 1, 3, E4M3 per 16 | 2.5 | 0.400 | 8.0 | 4.2 |
| Lloyd-Max 3-bit, FP16 norm per 256 | 3.06 | 0.185 | 14.7 | 1.95 |
| Lloyd-Max 2-bit, FP16 norm per 256 | 2.06 | 0.342 | 9.3 | 3.6 |

Three readings. Each bit removed costs about 6 dB, as rate-distortion predicts: 3.5 bpe carries 1.8x today's error, 2.5 bpe 4.2x. The
E2M1 grid is 1.5 dB worse than a symmetric integer grid at the same 4.5 bpe on rotated rows (its 0.5 and 6 levels are rarely right once
the group maximum sets the scale), so an INT4 cache at unchanged bytes is a free-quality candidate if the tables are rebuilt anyway.
And norm scaling at 3.06 bpe (TurboQuant without QJL) is worse than INT3 with E4M3 per 16 at 3.5 bpe, so the per-16 scale, which also
keeps the k16 alignment the kernels rely on, is worth its 0.44 bits.

### 2.5 Proposed row layouts (256-token pages, token-major, 4 heads per token)

| Format | Code bytes | Scale bytes | Row | Per token (16 layers, K and V) | K page (256 tokens x 4 heads) |
|---|---|---|---|---|---|
| NVFP4 (today) | 128 (byte = 2 codes) | 16 | 144 | 18 KiB | 147,456 B |
| INT3 per 16 | 96: eight 12-byte units, each two 16-groups = 32 codes x 3 bits in three 32-bit words, code pair p at bit offset 6p | 16 | 112 | 14 KiB | 114,688 B |
| INT2 per 16 | 64 (byte = 4 codes) | 16 | 80 | 10 KiB | 81,920 B |
| K INT3 / V INT2 | | | 112 / 80 | 12 KiB | |

Both rows stay multiples of 16 bytes, so `cp.async` chunks per token drop from 9 to 7 or 5. The 12-byte unit keeps every load 4-byte
aligned; the two code pairs that straddle a word boundary (bit offsets 30 and 60) take one funnel shift. Scales keep the E4M3 contract
with `s = rn(amax / 7)` (INT3) or `rn(amax / 3)` (INT2) and saturating codes, as `e2m1_encode` does today. Key and value pages have
separate page tables, so mixed K/V formats need no common row size.

## 3. Published results

### 3.1 Perplexity at short context

| Model | Size | K / V bits | Metric | Baseline | Result | Source |
|---|---|---|---|---|---|---|
| LLaMA-2 | 7B | KV4 asymmetric, group 128, clip 0.95 | WikiText ppl | 5.47 | 5.51 | QuaRot appendix [3] |
| LLaMA-2 | 7B | KV3 | WikiText ppl | 5.47 | 5.68 (+0.21) | [3] |
| LLaMA-2 | 7B | K4V3 / K3V4 / K4V2 | WikiText ppl delta | | +0.03 / +0.18 / +0.28 (5.75) | [3] |
| LLaMA-2 | 70B | KV3 | WikiText ppl delta | | +0.07 | [3] |
| LLaMA | 7B | 3-bit keys, pre-RoPE vs post-RoPE | WikiText ppl | | pre-RoPE better by 0.82 | KVQuant [2] |
| LLaMA, Llama-2, Llama-3, Mistral | 7B to 70B | nuq3 with 1% outliers | WikiText and C4 ppl delta | | under 0.1 | KVQuant abstract [2] |
| LWM-Text-Chat-128K | 7B | nuq4-1% / nuq3-1% / nuq2-1%, first 5 tokens FP16 | ppl (2K eval) | 9.63 | 9.67 / 9.94 / 11.57 | KVQuant repo [2] |
| LWM-Text-Chat-1M | 7B | same | ppl (2K eval) | 8.90 | 8.98 / 9.22 / 10.80 | [2] |
| LLaMA-2 | 7B | 2-bit, outlier-aware rotation, sink-aware | ppl delta | | under 0.3 | RotateKV [10] |

### 3.2 LongBench and downstream tasks

| Model | K / V bits | Metric | Baseline | Result | Source |
|---|---|---|---|---|---|
| LongChat-7b-v1.5-32K | KIVI-2 / KIVI-4 (group 32, residual 128) | LongBench 15-task average | 38.72 | 38.30 / 38.79 | KIVI repo [1] |
| Mistral-7B-Instruct-v0.2 (GQA) | KIVI-2 / KIVI-4 | LongBench average; PassageRetrieval | 43.54; 89.33 | 42.43 / 43.53; 80.83 / 89.42 | [1] |
| Llama-2-7B | KIVI-2, residual 128 / 32 | LongBench average | 44.52 | 44.27 / 44.08 | KIVI paper [1] |
| Llama-3.1-8B-Instruct | TurboQuant 3.5 / 2.5; PolarQuant 3.9; KIVI 3 | LongBench average | 50.06 | 50.06 / 49.44; 49.78; 48.50 | TurboQuant Table 1 as relayed [7][34] |
| Llama-3.1-8B-Instruct | same | needle-in-a-haystack | 0.997 | 0.997 / 0.997; 0.995; 0.981 | same |
| Llama-3.3-70B-Instruct | FP8; TQ k8v4; TQ 4-bit; TQ k3v4; TQ 3-bit (all without QJL) | MRCR-16 at 128K, relative | 1.000 | 0.991; 0.973; 0.957; 0.834; 0.722 | vLLM / Red Hat study, May 2026, as relayed [34] |
| Qwen3-30B-A3B | same formats | AIME25 pass@1 | 0.683 | 0.680; 0.668; 0.652; 0.532; 0.443 | same |
| Qwen3-30B-A3B, H100 | FP8; TQ 4-bit; TQ 3-bit | decode tokens/s | 4,520 | 4,510; 2,680; 2,140 | same |
| Qwen3-8B and others | KIVI-4 / KIVI-2 | reasoning tasks | | 4-bit holds; 2-bit "significant drop" | Kitty [33] |
| Qwen2.5-3B-Instruct | TQ-style K6/V4, K4/V4, K4/V2, with and without a 128-token FP16 window | hidden-fact retrieval at 2K and 4K | exact | K6/V4 exact; K4/V4 partial then miss; K4/V2 miss | tonbistudio port [34], community |

### 3.3 Long-context retrieval at 64K and beyond

| Model | Format | Test | Result | Source |
|---|---|---|---|---|
| LWM-Text-Chat-256K | KVQuant 3-bit, 1% outliers | haystack at 256K input | base model's retrieval retained (figure) | KVQuant repo [2] |
| Llama-3.3-70B | FP8 KV and FP8 attention, after an FA3 accumulation fix | 128K needle AUC | 97 to 98% of BF16; 13% before the fix | vLLM blog 2026-04-22 [24] |
| unnamed (NVIDIA) | NVFP4 KV | RULER 64K and others | under 1% loss vs BF16 and FP8; 50% smaller than FP8; about 5% better than MXFP4 KV | NVIDIA blog [25] |
| Qwen3.5-397B-A17B, Qwen3.8-27B | NVFP4 KV vs FP8 | GSM8K, GPQA-Diamond, AIME 2025, SWE-bench | "very small" differences on the 397B; the 27B table was not retrieved | LMSYS blog 2026-09-16 [26] |
| Llama-3.1-8B-Instruct | KIVI, PolarQuant at 0.25 of the cache | NIAH 4K to 104K | quantization beats eviction at equal budget | "More Tokens, Lower Precision" [33] |
| Qwen3.5-27B-AWQ (16 attention layers of 64, head_dim 256) | TQ 3-bit keys, 2-bit values | cosine of reconstructed K; of V | 1.000; 0.940 ("value quant dominates degradation") | 0xSero port [34], community |

No source gives RULER or needle tables at 128K for a 3-bit uniform or Hadamard-rotated cache on a 27B-class hybrid; the 70B MRCR and
30B AIME25 rows are the closest, and they are the strongest warning in this chapter.

### 3.4 Recurrent state

| Source | Finding |
|---|---|
| Gated DeltaNet-2 [16] | state stored FP32 across chunks and during recurrent decoding; the WY solve is the most precision-sensitive step |
| SGLang issue 38975 [18] | rounding beta through BF16 before the delta-rule update drifts the state with context; FP32 is bit-exact after the fix |
| Ollama issue 15865 [19] | a kernel writing the GDN state in BF16 corrupts Qwen3.5/3.6 output each step; the MLX reference keeps FP32 |
| exllamav3 PR 477 [20] | opt-in FP16 state storage to save VRAM; BF16 judged "not precise enough" |
| transformers PR 49291 [21] | MiniMax lightning attention: decay factors at or above about 0.998 round to 1.0 in BF16, heads stop forgetting |
| Kozyrev and Maiboroda [17] | NVFP4 W4A4 for the GDN half of a hybrid 27B matches BF16 within seed noise on MMLU-Pro, GSM8K, AIME'25, GPQA-Diamond, LiveCodeBench and RULER; injected error is forgotten by the recurrence and does not compound with context; the BF16 kernel floor is a 3.8e-3 median relative difference |
| Quamba2 [15] | 8-bit SSM state caching with per-state-group scales (`ssm_state_scale` in `qChunkScan.py`); W4A8 Mamba2-8B loses 1.6% on average |
| dasc, LeapQuant, DAMP, SSDi8, Nemotron 3 Ultra [22] | 2026 work quantizes cached or checkpointed states (INT8 per (layer, head, key) at 3.88x; 1.19 bytes per element for GDN and KDA; FP16 channels where INT8 degrades; block-scaled INT8 with stochastic rounding) and dequantizes into FP32 before continuing; DAMP finds uniform INT8 or FP8 already hurting complex reasoning |

## 4. Fit with Euhedral

### 4.1 The eight constraints of `00-baseline.md` section 6

1. **Row-exact 1 to 8 row decode.** The verifier twin (`euhedral_attention_decode_gqa_nvfp4_rows`) runs each row through `decode_block`
   with the same lookups as the one-row launch. INT3 and INT2 codes times E4M3 are exact in FP16 (2.4), so the lookup keeps its
   "equals the FP32 product rounded to FP16" property and the twin stays bitwise.
2. **A prefill route off the 102-TFLOPS path.** This constraint is about linears; prefill attention already expands tiles to FP16 and
   only its producer warps change.
3. **Decode instruction budget.** The 3-bit A-fragment build is one `bfe`, one shared load from a 64-entry table and one `hmul2` per
   code pair, today's count, plus a funnel shift for the two straddling pairs per 32 codes, and 7 instead of 9 `cp.async` chunks per
   key row. Estimate: within 10% of today's expansion instructions against 22% (INT3) or 44% (INT2) fewer bytes.
4. **Determinism.** Encode is per row in FP32 with a fixed butterfly order and round-to-nearest; decode is a table. Graph replay and
   output hashes are unaffected.
5. **Offline conversion only.** The KV format is not a tensor; it needs an artifact-level field (a KV-format ordinal in the EDRL
   profile) that `ArtifactProfile` reads, with existing artifacts defaulting to NVFP4 KV.
6. **The artifact decides everything.** Follows from 5: `q3-kv3` or `nvfp4-compressed-kv3` is a new recipe, not an option.
7. **Memory is the product.** Section 4.3.
8. **Drafting precision is a separate knob.** The MTP layer's own cache (one layer in the same page format; `PREFIX_CACHE.md` counts
   256-token MTP pages) is 1/17 of the KV and only seeds drafts; it can follow the base format or stay NVFP4, decided by acceptance.

Two more contracts hold for the same reason the verifier does. The **prefix cache** copies pages as bytes and sizes nodes by bytes per
token, so it needs the new constants and nothing else; its bit-for-bit restore survives because the quantizer is deterministic and the
chunk grid is unchanged. The **exact oracle** (`reference_decode_gqa.cuh`, the `_exact` twins, `reference_prefill_fa2.cuh`) needs the
software decode of the new codes, and the equality test that enumerates every finite E4M3 scale byte stays as is.

### 4.2 Kernels and components that change

| File | Change |
|---|---|
| `nvfp4_kv.cuh` | INT3 / INT2 encode (`rn(amax/7)`, `rn(amax/3)`, saturating codes), software decode, `element()`, `cache_row` with a row-size parameter |
| `kv_append.cuh` | row offsets 112 / 80 instead of 144; same warp-per-row structure |
| `nvfp4_pipe.cuh` | 64-entry (INT3 pair) and 16- or 256-entry (INT2 quad) FP16 pair tables beside `e2m1_pair_bits` |
| `nvfp4_decode_gqa.cuh` | `issue_keys` chunk count, A-fragment field extraction with funnel shifts, `load_values` field widths; the `_rows` twin inherits |
| `nvfp4_prefill_fa2.cuh`, `nvfp4_prefill32.cuh` | producer `load` and expansion per 12-byte unit; `stage()` |
| `reference_*.cuh`, `nvfp4_attention.cuh` (`_exact`) | software decode; merge unchanged |
| planner, `PrefixCache`, `ArtifactProfile` | bytes per token, page bytes, KV-format field |

### 4.3 Resident memory before and after

| K / V format | Row bytes K / V | KiB per token | 32K | 64K | 128K |
|---|---|---|---|---|---|
| NVFP4 / NVFP4 (today) | 144 / 144 | 18 | 0.56 GiB | 1.13 GiB | 2.25 GiB |
| INT3 / INT3 | 112 / 112 | 14 | 0.44 | 0.88 | 1.75 |
| INT3 / INT2 | 112 / 80 | 12 | 0.38 | 0.75 | 1.50 |
| INT2 / INT2 | 80 / 80 | 10 | 0.31 | 0.63 | 1.25 |

Host-backed weights per artifact (today from `00-baseline.md` section 4; after = today minus the KV saving, estimate; the planner
rounds in 64 MiB steps):

| Artifact | Context | Today | INT3 / INT3 | INT3 / INT2 | INT2 / INT2 |
|---|---|---|---|---|---|
| `q3` (11.72 GiB) | 32K to 128K | 0 | 0 | 0 | 0 |
| `nvfp4-compressed` (13.77 GiB) | 32K | 0.27 GiB | 0.15 | 0.08 | about 0 (fully resident) |
| | 64K | 0.84 | 0.59 | 0.47 | 0.34 |
| | 128K | 1.97 | 1.47 | 1.22 | 0.97 |
| `nvfp4` (14.52 GiB) | 32K | 1.0 | 0.88 | 0.81 | 0.75 |
| | 64K | 1.6 | 1.35 | 1.23 | 1.1 |
| | 128K | 2.8 | 2.3 | 2.05 | 1.8 |

No NVFP4 artifact becomes fully resident at 64K or 128K; `nvfp4-compressed` does at 32K with a 2.5-bit cache and nearly does (0.08 GiB)
with INT3 / INT2. The GDN state (0.15 GiB), the 157 MiB of ReplaySSM buffers and the 0.51 GiB of per-sequence workspaces are untouched.

### 4.4 Long-context decode speed (estimates)

Two effects. First, host-backed weights. `NVFP4_RESIDENCY.md` measures one-row decode at 48.1 tok/s resident, 45.4 with 512 MiB
host-backed, 35.0 with 1 GiB and 19.2 with 2 GiB: about the first 0.7 GiB hides under the GPU's own step time, the rest streams at the
44 GB/s copy rate, so a step is roughly max(GPU time, host-backed bytes / 44 GB/s). The baseline's `nvfp4-compressed` figures fit:
88 tok/s at 64K with 0.84 GiB (20 ms of copy per step, GPU-bound) and 52 at 128K with 1.97 GiB (about 48 ms of copy per step,
copy-bound). Cutting the 128K host-backed set to 1.47 GiB (INT3 / INT3) makes the copy about 36 ms, to 1.22 GiB (INT3 / INT2) about
30 ms, near the GPU's own 128K step time: expected 128K decode about 70 and about 80 tok/s respectively, with INT2 / INT2 gaining little
more because the step is then GPU-bound. For plain `nvfp4`, 2.8 to 2.05 GiB takes the copy from 68 to 50 ms per step and decode from
the measured 30 tok/s to perhaps 40.

Second, attention bytes, the only effect for `q3`. From `ATTENTION_DECODE.md` the one-row kernel streams 72 MiB per layer at 64K in
107 us (82% of DRAM rate), the 3-row verifier takes 308 us per layer, and a fused read of all three rows was bounded at a 19% saving,
so the twin is partly bandwidth-bound. Scaled linearly to 128K the twin costs about 9.9 ms per step over 16 layers, and the `q3`
slowdown from 109 tok/s at 4K to 67 at 128K (+5.75 ms per token) is almost entirely this attention. With 22% (INT3 / INT3) or 33%
(INT3 / INT2) fewer bytes and a kernel between latency-bound and bandwidth-bound, the step saves 1 to 2.5 ms of about 24 ms: `q3` at
128K goes from 67 to roughly 70 to 75 tok/s (+4 to +12%); at 32K the attention share is a few percent and the gain under 3%. The 3-bit
field extraction gives some of this back in the values warp unless tuned.

### 4.5 MTP, DFlash2 and the GDN state

The MTP cache follows the base format (4.1, point 8). DFlash2 keeps a BF16 ring of the last 2048 positions, 5 layers x 2 x 4 MiB =
40 MiB whatever the context (`DFLASH2.md`), plus 40 MiB per prefix-cache node; no KV format touches it, and its drafts are verified
against the base cache, so acceptance is the only coupling to measure.

The GDN state stays FP32. Arithmetic: 48 layers x 3 MiB read and written per token is about 0.3 GiB, 2.5% of a 16 ms token, so BF16
storage saves at most 1.2% of decode time and 72 MiB of device memory (plus 72 MiB of ReplaySSM checkpoint and 73 MiB per prefix-cache
checkpoint, which would double the 27 checkpoints the 4 GiB arena holds). Against that: the delta rule
`S <- alpha S + beta (v - alpha S k) k^T` forgets a per-step storage rounding error of relative size 2^-9 (BF16) at rate alpha, so the
steady-state error is about 2^-9 / sqrt(1 - alpha^2): 7x the single rounding at alpha = 0.99 and 22x at alpha = 0.999, i.e. 1.4% to 4%
of the state (estimate), well above the 3.8e-3 BF16 kernel floor measured in [17] for this architecture class. FP16 storage (2^-12)
would sit at 0.2 to 0.5%, which is why exllamav3 offers FP16 and not BF16, but it still changes every output against the FP32 reference,
and the row-exact replay would then reproduce a worse state exactly. The 2026 state-compression papers quantize checkpoints and
dequantize into FP32 before continuing; the prefix cache's contract (a restored state equals a cold run buffer for buffer) rules that
out here. The engine computes alpha and beta in FP32 already, so the SGLang and transformers failure modes do not apply.

### 4.6 FP4 attention for prefill

SageAttention3 reports 1038 TOPS on an RTX 5090 (5x FlashAttention) with Q and K in NVFP4 and P and V under two-level FP4 scaling, on
the same `kind::mxf4nvf4` m16n8k64 MMA this hardware has [13]; its README limits the claim to video and image diffusion models and
recommends SageAttention2++ where precision matters, and Attn-QAT measures at most 1.3x over BF16 FlashAttention-4 on B200 for
block-scaled QK and a slowdown when PV is also quantized [14]. The lever is small here anyway: from `PREFILL_MX.md`, one 512-row
chunk's attention costs 0.66 ms at a 2K cache and 6.96 ms at 32K, so a 32K prompt spends about 0.23 s of its 20 s TTFT in attention
(estimate, about 1%) and a 128K prompt about 3.7 s of 95 to 133 s (about 3%). A 4x faster kernel that also quantizes P and Q buys at
most 2% of TTFT and costs the prefill path's agreement with the oracle. Not recommended.

## 5. Experiment plan

1. **Numerics-only prototype in today's container** (about 50 lines, one day). Keep the 144-byte rows and swap only the codec: a
   test-build switch replaces `e2m1_encode` / `e2m1_magnitude` / `e2m1_pair_bits` and the scale divisor with the INT4-midrise, INT3-midrise
   and INT2-midrise grids (magnitude in the low nibble bits, sign in bit 3). Storage does not shrink, but every kernel then computes
   exactly the numerics of the candidate cache, exact twins included. Run `TeacherForcedQualityCudaIntegrationTest` and
   `tools/compare_reference.py` for each grid and for E2M1 on `q3` and `nvfp4-compressed`. This also measures for the first time what
   the cache costs at all, since the baseline's artifact and kernel comparisons both run through the NVFP4 cache. Go if INT3 / INT3 and
   INT3 / INT2 stay within KL 0.01 and 1 top-1 point of E2M1 (the kernel drift floor is KL 3.4e-3 to 4.7e-3); no-go for a grid that
   doubles the E2M1 cache's KL against the BF16 reference.
2. **Row statistics** (Python, from a dump of the BF16 K and V rows of the quality document): relative error per layer and per K / V
   under each grid, group maxima, and whether layer 3 (the first attention layer) or sink positions need a higher-precision page.
   Decide INT4-midrise versus E2M1 at 4.5 bpe here.
3. **Long-context harness** (new): passkey and multi-needle retrieval at 32K, 64K and 128K through the HTTP API and a RULER subset at
   64K, run on the E2M1 cache first for the floor, then on the step-1 prototypes. Go if 128K retrieval stays within 1 point of E2M1.
4. **Kernels and layout.** Codec and exact reference first, bitwise tests over all scale bytes and lengths 1 to 16385 as in
   `test_gqa_decode_warp_specialized_equals_single_warp_reference_bitwise`; then decode GQA and its `_rows` twin; then the two prefill
   kernels; then planner, prefix cache and the profile field. Time the decode kernel cold at 4K to 64K against today's 21 / 38 / 67 /
   116 us one-row and 37 / 84 / 151 / 308 us three-row figures.
5. **Benchmarks.** `benchmark/configs/*.json` at 4K / 32K / 64K / 128K for `q3` and `nvfp4-compressed`, MTP tokens per verification,
   the prefix-cache tests, and the teacher-forced report once more (output hashes will differ from the E2M1 run because logits change).
   Go if `nvfp4-compressed` at 128K gains at least 15% with the step-1 and step-3 quality intact.

## 6. Risks and open questions

- **Reasoning sensitivity.** The only 3-bit results on 2025-26 reasoning models (TurboQuant 3-bit on Qwen3-30B-A3B: AIME25 0.683 to
  0.443) are bad, and Swift-1.5 is a reasoning fine-tune judged on a 1279-token document. The teacher-forced harness cannot see a
  long-generation failure; step 3 plus a chain-of-thought task must precede any artifact shipping a 3-bit cache.
- **Head dimension 256 and 16 layers.** Nearly every published number is for head_dim 128 and 32 layers. The rotation argument gets
  stronger at 256 and the cache is already a quarter of a dense model's, but the two reports on this architecture family (0xSero on
  Qwen3.5-27B, varjoranta on Qwen3.6-27B) are community reconstruction cosines, not task accuracy.
- **Values at 2 bits.** The simulation puts INT2 at 4.2x today's error and the community ports put values as the bottleneck at 2 bits;
  the published support for 2-bit values is K4V2 on 7B models at 2K to 4K. INT3 / INT2 is the stretch target; INT3 / INT3 the plan.
- **Sink and first-token rows.** The engine quantizes every row alike. KVQuant and KVSink keep the first tokens in FP16; a per-page
  precision flag can express that but complicates the exact-product path. Decide from step 2.
- **Instruction cost of 3-bit fields.** The scores warp runs at about 0.1 instructions per cycle and the FA2 producers already lag the
  compute warps; funnel shifts could eat the byte saving at 16K and below. Measure before committing to 12-byte units.
- **Which grid at 4.5 bpe.** INT4-midrise beats E2M1 by 1.5 dB on rotated rows in simulation; confirming it on real rows (step 2) would
  improve today's artifacts at zero memory cost, but it changes the exact oracle and every stored prefix-cache node's meaning.
- **Prefix cache arena.** Each checkpoint's 147 MiB GDN state dominates the arena; a smaller KV format does not move the 27-checkpoint limit.
- **Unverified sources.** The TurboQuant table and the vLLM / Red Hat May 2026 study are cited as relayed by community READMEs; the
  LMSYS Qwen3.8-27B NVFP4-KV table was not retrieved; no RULER-128K table for a 3-bit cache was found.

## 7. References

1. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. arXiv:2402.02750 (ICML 2024). https://github.com/jy-yuan/KIVI (`docs/long_bench.md`).
2. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. arXiv:2401.18079 (NeurIPS 2024). https://github.com/SqueezeAILab/KVQuant (`README.md`, `quant/README.md`, `lwm/README.md`).
3. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. arXiv:2404.00456. https://github.com/spcl/QuaRot (`fake_quant/rotation_utils.py`, `quarot/transformers/kv_cache.py`); KV ablation and clip sweep from the paper's appendix as returned by search.
4. SpinQuant: LLM Quantization with Learned Rotations. arXiv:2405.16406 (ICLR 2025).
5. Atom: Low-bit Quantization for Efficient and Accurate LLM Serving. arXiv:2310.19102 (MLSys 2024).
6. QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving. arXiv:2405.04532 (MLSys 2025).
7. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate. arXiv:2504.19874 (ICLR 2026).
8. PolarQuant: Quantizing KV Caches with Polar Transformation. arXiv:2502.02617.
9. PolarQuant: Leveraging Polar Transformation for Efficient Key Cache Quantization and Decoding Acceleration. arXiv:2502.00527.
10. RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations. arXiv:2501.16383 (IJCAI 2025, https://www.ijcai.org/proceedings/2025/0690.pdf).
11. ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification. NeurIPS 2024, https://proceedings.neurips.cc/paper_files/paper/2024/file/7e57131fdeb815764434b65162c88895-Paper-Conference.pdf.
12. MiniKV: Pushing the Limits of 2-Bit KV Cache via Compression and System Co-Design. arXiv:2411.18077 (Findings of ACL 2025).
13. SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training. arXiv:2505.11594 (NeurIPS 2025). https://github.com/thu-ml/SageAttention (`sageattention3_blackwell/`).
14. Attn-QAT: 4-Bit Attention With Quantization-Aware Training. arXiv:2603.00040.
15. Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models. arXiv:2503.22879 (ICML 2025); Quamba, arXiv:2410.13229. https://github.com/enyac-group/Quamba (`quamba/qChunkScan.py`).
16. Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention. arXiv:2605.22791.
17. Kozyrev, Maiboroda. Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM. arXiv:2609.04098.
18. SGLang issue 38975. https://github.com/sgl-project/sglang/issues/38975
19. Ollama issue 15865. https://github.com/ollama/ollama/issues/15865
20. exllamav3 pull request 477. https://github.com/turboderp-org/exllamav3/pull/477
21. transformers pull request 49291. https://github.com/huggingface/transformers/pull/49291
22. dasc, arXiv:2608.30386; LeapQuant, arXiv:2609.38166; DAMP, arXiv:2608.27513; Low-Discrepancy Dither for Quantized Recurrent State Caches, arXiv:2609.39185; SSDi8, arXiv:2608.21952; Nemotron 3 Ultra, arXiv:2606.15007.
23. vLLM, Quantized KV Cache documentation (raw file read). https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/quantized_kvcache.md
24. vLLM blog, The State of FP8 KV-Cache and Attention Quantization in vLLM, 2026-04-22. https://vllm.ai/blog/2026-04-22-fp8-kvcache
25. NVIDIA, Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache. https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/ ; NVIDIA Research, Pushing Intelligence to 4-bit. https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/
26. LMSYS, Accelerating Long-Context and Agentic Inference with NVFP4 KV Cache, 2026-09-16. https://www.lmsys.org/blog/2026-09-16-nvfp4-kv-cache/
27. vLLM PR 46329 (NVFP4 KV on sm_120/sm_121). https://github.com/vllm-project/vllm/pull/46329 ; SGLang issue 42369 (NVFP4 KV reusing FP8-calibrated scales). https://github.com/sgl-project/sglang/issues/42369 ; vLLM RFC 28538 (4-bit KV through Hadamard transforms). https://github.com/vllm-project/vllm/issues/28538
28. llama.cpp server README (raw file read; cache types f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1). https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md ; discussion 23470 (asymmetric q8/q4 cache, Qwen2.5-7B). https://github.com/ggml-org/llama.cpp/discussions/23470 ; discussion 21297 (Qwen3-8B key statistics); issue 26989 (NVFP4 KV request).
29. localbench, Gemma 4 and Qwen 3.6 with q8_0 and q4_0 KV cache: KL divergence results. https://localbench.substack.com/p/kv-cache-quantization-benchmark
30. Hurwitz Quaternion Multiplicative Quantization for KV Cache Compression. arXiv:2605.27646.
31. More for Keys, Less for Values: Adaptive KV Cache Quantization, arXiv:2502.15075; KVTuner, arXiv:2502.04420 (ICML 2025); RateQuant, arXiv:2605.06675.
32. KVSink, arXiv:2508.04257; Prefixing Attention Sinks can Mitigate Activation Outliers (CushionCache), arXiv:2406.12016; Massive Activations in Large Language Models, Sun et al. 2024 (arXiv ID not verified in this search).
33. More Tokens, Lower Precision, arXiv:2412.12706; Kitty, arXiv:2511.18643; KVLinC, arXiv:2510.05373; D-Quant, arXiv:2609.19880; OScaR, arXiv:2605.19660; Accurate KV Cache Quantization with Outlier Tokens Tracing, arXiv:2505.10938 (ACL 2025).
34. TurboQuant community ports (cloned; untrusted secondary sources): https://github.com/OnlyTerp/turboquant (`BENCHMARKS.md` relays the vLLM / Red Hat study at https://blog.vllm.ai/2026/05/11/turboquant.html, not fetched); https://github.com/0xSero/turboquant ; https://github.com/tonbistudio/turboquant-pytorch ; https://github.com/varjoranta/turboquant-vllm ; https://github.com/AmesianX/TurboQuant.
35. RULER: What's the Real Context Size of Your Long-Context Language Models? Hsieh et al. 2024 (arXiv ID not verified in this search).
36. Euhedral, branch `perf/mtp-gated-depth`: the docs named in the scope note plus `docs/MTP_SPECULATIVE.md`; `native/src/attention/nvfp4_kv.cuh`, `nvfp4_pipe.cuh`, `kv_append.cuh`, `nvfp4_decode_gqa.cuh`, `nvfp4_prefill_fa2.cuh`, `qk_norm_rope.cuh`; `native/src/gdn/recurrence.cuh`; `ArtifactProfile.java`.
