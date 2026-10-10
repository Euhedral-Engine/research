# Public baselines and evaluation: where the four forms stand, and how to measure

This chapter is the external yardstick for the folder: every public quantization of Qwen3.8-27B found on
2026-10-10, the recipes behind the best ones, what is known about the Swift 1.5 checkpoint the engine serves, and
the evaluation practice the engine should adopt. Sources are of three kinds and are labelled: repositories cloned
and read (llama.cpp at commit 23b0202a, ik_llama.cpp with its bundled discussion archive, IST-DASLab/GSQ and /RCO,
UkisAI/Swift-Qwen3.8-27B-evals, five community llama.cpp forks); Hugging Face cards and forum posts seen only
through search-result summaries (huggingface.co could not be fetched; marked "via summary"); and papers identified
by arXiv ID from search results (abstract-level claims). Nothing was recalled from memory.

Footing. The engine's numbers in `00-baseline.md` (PR 119 at commit 269a890) rest on two checkpoints: `q3`,
`q3-compressed` and `nvfp4` are conversions of the base Qwen3.8-27B with ISTA-DASLab's `imatrix-qwen3.8-27b.gguf`,
scored against the base BF16 (PPL 8.39 on the quality document); the deployed `nvfp4-compressed` is a conversion of
Swift 1.5 with UkisAI's published `imatrix-swift15-v1mix.gguf`, scored against Swift 1.5's BF16 (PPL 8.51). Every
public number in section 2 is for the base model unless the row says otherwise, on wikitext-class text, with an F16
KV cache and unquantized activations. For the three base forms the checkpoint footing matches the public one; what
differs is the text and what the number bundles (the engine's includes its NVFP4 KV cache and, for the NVFP4 forms,
two-term FP4 activation quantization in prefill). Section 6 says how to remove those differences.

## 1. Summary

At equal file size the engine's forms sit far below the public frontier for this model on the one metric both report,
KL divergence to BF16, and for the three base forms the comparison is on the same checkpoint: `nvfp4` (14.52 GiB,
4.64 bpw over 26.9 B parameters) measures KL 0.147 and top-1 88%, where public 4.4 to 5.3 bpw GGUFs measure KL 0.010
to 0.019 and top-1 95 to 96%; `q3` (11.72 GiB, 3.74 bpw) measures KL 0.293 and top-1 76%, where public 3.0 to 4.1 bpw
GGUFs measure KL 0.05 to 0.08 and top-1 about 92%, and only 2.5 to 2.7 bpw files reach KL 0.15 to 0.30 with top-1
about 83%. The deployed `nvfp4-compressed`, converted from Swift 1.5 with UkisAI's matrix and scored against Swift's
BF16, lands at KL 0.155 and 89.4%, so the checkpoint is not what separates the engine from the public files. The gap
is 5 to 10x; the single short document contributes sampling error of about 0.01 to 0.025 in KL, the text genre
plausibly a factor of 2 (estimate), and the NVFP4 KV cache about 0.015 (one same-GPU measurement). The best public
recipes share three features the engine lacks: the LM head and the GDN output projection stay at 5 to 8 bits, the
per-tensor allocation is searched rather than fixed (ISTA's GSQ-RCO reports AIME25 and LiveCodeBench v6 equal to
BF16 at 3.5 bpw), and the quantizer uses error feedback or learned assignments rather than per-group scale search.
Swift 1.5 is a LoRA-derived reasoning-efficiency tune (its 1.0 evals repo serves it as an adapter over the BF16 base)
claiming 58.5% fewer thinking tokens and +0.35% average score under a "Swift Open License v1.0"; whether its MTP
block was touched is undocumented. UkisAI's own GSQ-RCO GGUF of Swift 1.5, with the 1.27 M-token matrix the deployed
form already uses, is the most comparable public quantization of the served checkpoint, but its bit widths and
quality could not be retrieved in this session. The recommended protocol: evaluate every form on the deployed
checkpoint; keep the 1279-token document for continuity and add wikitext-2 test, a code set and Swift 1.5's own
thinking generations at 32K scored tokens each; report mean KL with standard error, median and 99th percentile,
same-top-1 and RMS delta-p; isolate weight error from KV error by dequantizing artifacts back to a GGUF; add
teacher-forced KL at 64K and 128K depth, a needle set, MTP acceptance with the base and the Swift head, and
thinking-token counts on GSM8K and LiveCodeBench, because quantization inflates reasoning length and that is the
property Swift 1.5 was trained for.

## 2. Public quantizations of Qwen3.8-27B

### 2.1 Table

Bits per weight are file size over 26.9 B parameters (text tower plus MTP block; ISTA's card implies the same base:
53.8 GB BF16 = 16.00 bpw); GiB converted at 2^30, GB at 10^9. "KLD" is mean KL(BF16 || quant) unless noted; rows use
different texts and references and compare only within a source.

| Release (checkpoint) | Method, allocation | File, bpw | Calibration | Reported quality | Source |
|---|---|---|---|---|---|
| bartowski/Qwen3.8-27B-GGUF (base) | llama.cpp k/i-quants, imatrix; "standard" mix and a "computed" per-tensor layout | Q4_K_M 16.55 GiB = 5.28; Q3_K_M about 13 GiB (estimate) | bartowski calibration set | wiki.test.raw, 100 chunks, vs BF16, base PPL 6.744. Mean KLD: Q6_K 0.0030; Q4_K_M 0.0129 standard / 0.0139 computed; Q3_K_M 0.0466 / 0.0564 (standard file 8.2% smaller). Percentiles: Q8_0 median 0.00033, 99% 0.0071; Q5_K_M 0.0023, 0.051; Q4_K_M 0.0058, 0.142. IQ2_XXS: KLD 0.301, PPL 9.12 vs Q8_0 (independent) | [1][2] via summary |
| unsloth/Qwen3.8-27B-GGUF (base) | Dynamic 2.0/3.0 layer-selective mixes; blk.64 MTP block embedded (272 MiB in UD-Q4_K_XL), output Q6_K | UD-IQ2_XXS 8.39 GiB = 2.68; UD-Q2_K_XL 9.83 GB = 2.92; UD-Q4_K_XL 17,559,178,144 B = 5.22 | 300K to 1.5M tokens of chat, long-context and tool-calling text | UD-IQ2_XXS KLD 0.146 mean, 0.076 median, top-1 82.98% (independent, vs Q8_0); UD-Q4_K_XL KL 0.00955 vs BF16, top-1 96.02% (third party); Q6_K 0.0029, Q4_0 0.028 (akopytko, vs BF16, base PPL 6.7945); unsloth v3 raw runs 0.018772 +- 0.000201 and 0.013652 +- 0.000229 (quants unnamed); "about 8% better top-1 than the next option" for UD-Q2_K_XL (publisher) | [2][3][4][5][6] |
| AtomicChat/Qwen3.8-27B-GGUF (base) | llama.cpp types | Q8_0 to IQ1_M | | KLD about 0.0006 (Q8_0) to 0.34 (IQ1_M) | [7] via summary |
| Qwen discussion #65 poster, "AD-IQ3_S" (base) | own per-tensor allocation, 20 community GGUFs in one harness | 13.8 GB = 4.10 | | eval_neutral, ctx 4096, 87 chunks, BF16 ref: top-1 92.4%, 33% less drift than unsloth Q3_K_M at that size; unsloth UD 2-bit wins below 11 GB; Q6/Q8 equivalent above 25 GB; everything degrades fast below 10 GB | [8] |
| ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF (base) | GSQ codes, RCO per-tensor GGUF type; IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S; "-mtp" variants; mmproj | 8.4 GB = 2.50; 9.3 GB = 2.77; 10.1 GB = 3.00; 11.8 GB = 3.51 | GSQ: OpenThoughts for Qwen3.5-class configs; imatrix published (1000 x 4096 tokens per the engine docs) | AIME25 / GPQA-D / LCB v6 / zero-shot avg: BF16 100.00 / 89.90 / 85.71 / 74.34; IQ2_XS 96.67 / 84.85 / 76.57 / 74.54; IQ2_S 100.00 / 86.36 / 82.29 / 75.70; IQ3_XXS 100.00 / 88.89 / 84.57 / 74.81; IQ3_S 100.00 / 89.39 / 85.71 / 74.47, wikitext2 PPL 7.07, C4 11.76, FineWeb-Edu 8.34, called "task-lossless" | [9][10] via summary |
| ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF (Swift 1.5, deployed checkpoint) | GSQ-RCO GGUF with UkisAI's own `imatrix-swift15-v1mix.gguf` | not retrieved | 309 x 4096 tokens, about 1.27 M tokens of web, reasoning, multilingual and code text | not retrieved: the repository appeared in one search result and the search budget ran out before its card could be queried; whether GSQ was rerun on Swift or ISTA's allocation reused is unknown | [46][74] |
| RentedNoodle IQ3_XXS + Q6_K MTP head (uncensored base) | ISTA's published per-tensor RCO allocation, custom imatrix | 9.75 GiB = 3.11 | custom "OrcaRouter-native" imatrix | WikiText-2 PPL 6.1745 +- 0.149 (40 x 512; no same-harness BF16); needle 6/6 on a 15K-word haystack; MTP n-max 2 acceptance 0.67 at 32K, 0.70 to 0.77 at 65K; RTX 5070 Ti 16 GB | [11] repo |
| cdiamond iMatrix-NVFP4-MTP GGUF (base) | NVFP4 bulk, sensitive tensors Q5_K/Q6_K, MTP embedded | 17.1 GB = 5.09 | imatrix | short WikiText-2 control: 6.1197 vs Q4_1 6.1127 (gap below the check's error) vs stock NVFP4 build 6.4949 | [12] via summary |
| NVFP4 checkpoints (base): nvidia; unsloth; Inferact; RadixArk; huginnfork NVFP4A16; TelperionAI NVFP4-AWQ-GPTQ | nvidia: NVFP4 on MLP and lm_head, FP8 attention; huginnfork: MLPs only, attention and vision BF16; TelperionAI: AWQ scaling + GPTQ | nvidia about 24.6 GiB in vLLM; RadixArk about 22 GiB | | unsloth "92 to 97% top-1", vLLM MTP acceptance 0.788; Inferact 0.897; huginnfork "+3% PPL over bf16"; RadixArk + TurboQuant 4-bit KV + MTP-3 about 160 tok/s on an RTX 5090 | [13][14][15][16][17][18] |
| Qwen official | only FP8 (block 128) is official; no official GGUF, AWQ, GPTQ or MLX at launch | about 28 GB | | | [19] |
| MLX (mlx-community 4-bit and MTP-4bit, lmstudio-community 4-bit, PocketAiHub 4/6/8-bit) | mlx_lm affine | 4-bit 14.98 GiB = 4.78; 8-bit 27.50 GiB | | none | [20] |
| Not found | doth4580/Qwen3.8-27B-NVFP4-FULL: a "veloGB10" engine model, "equivalent to `--quantize all`", no card, author retracted its speed claims; bernhardbrieger, RedHatAI/Neural Magic, mradermacher: no Qwen3.8-27B release found | | | | [21] |

Engine forms on the same footing: `q3` 11.72 GiB = 3.74 bpw, `q3-compressed` 10.16 GiB = 3.24, `nvfp4` 14.52 GiB
= 4.64 (base, ISTA matrix); `nvfp4-compressed` 13.77 GiB = 4.40 (Swift 1.5, UkisAI matrix). The engine files carry a
0.25 to 0.35 GiB vocabulary-trimmed draft head no GGUF has; without it `q3` is 3.66 and `nvfp4` 4.52 bpw. Both the
engine files and the GGUFs with embedded MTP carry the 0.42 B MTP block.

### 2.2 What the best recipes do

**llama.cpp mixes** (`src/llama-quant.cpp`): `output.weight` is Q6_K for every k-quant and IQ3_S/IQ4_XS mix, Q5_K
for IQ2 and IQ3_XXS; `attn_v` is Q4_K or Q5_K for IQ3_S, IQ3_M, IQ4_XS and Q3_K_M when GQA is at least 4 (Qwen3.8:
6); `ffn_down` gets more bits in the first and last eighth of the layers and every third layer between; norms and
`ssm_conv1d` are never quantized. MTP tensors (`blk.64.*`) follow the trunk's rules; the only special case found is
GLM5-Next's `nextn.eh_proj` floor at Q8_0 [34]. The engine's `q3` puts the LM head, token embedding, `attention/output`
and `gdn/output` at 3.25 bpw.

**unsloth Dynamic 2.0/3.0**: layer-selective widths; chat and tool-call calibration; KLD plus a "Divergence-300"
metric (300 held-out prompts, 32 greedy tokens against BF16) [22][3]. For the Qwen3.5 hybrid family unsloth retired
MXFP4, advises never pushing `ssm_out` down, and keeps SSM output at Q5_K and SSM alpha/beta at Q8_0 even in
UD-Q2_K_XL [23][24].

**ISTA GSQ-RCO.** GSQ (Gumbel-Softmax Quantization, arXiv:2604.18556; Dadgarnia, Tabesh, Nikdan, Helcig, Kurtic,
Kleinegger, Alistarh) is post-training scalar quantization on a symmetric group-wise integer grid, so INT kernels and
GGUF k-quants can run it: GPTQ initialization per layer, then one learnable logit vector per weight over the grid,
sampled with Gumbel-Softmax at an annealed temperature (2.0 to 0.05) and logit scale (100 to 500), trained 10 epochs
per layer with Lion against the layer's reconstruction error on calibration activations, then hard-rounded. Group
size 128; GPTQ with 512 samples, 10% damping; Qwen3.5 configs calibrate on 4096 OpenThoughts samples of up to 4096
tokens, Llama configs on FineWeb-Edu. Reported: 2.13 bpw Llama-3.1-70B zero-shot 75.57 vs 78.99 FP16 and 76.3 to
77.3 for vector quantizers; unsloth Qwen3-8B Q2_K refined in-format from 50.03 to 56.28 [25][26]. RCO (Riemannian
Constrained Optimization, arXiv:2605.00649; Helcig, Alistarh) picks one of K options per tensor under an exact total
bit budget: the budget is a smooth manifold in softmax-logit space, Adam steps are projected onto it, drift is
retracted along the cost vector, the discrete forward pass uses Gumbel-STE with an exact knapsack solver; claimed
parity with EvoPress at 3 to 16x less wall-clock [27][28]. The GGUF release is GSQ-refined weights with an RCO-chosen
GGUF type per tensor; RentedNoodle reuses ISTA's published allocation file `REF-IQ3_XXS-mtp.rco-allocation.txt` [11].
ISTA reports benchmarks, not KLD; the one independent test (60 questions, 58/60 for IQ2_S MTP) notes the
"task-lossless" label is the card's [29].

**cdiamond's hybrid** keeps sensitive tensors at Q5_K/Q6_K inside an NVFP4 file and matches Q4_1 where a stock
all-NVFP4 build was 6% worse in PPL [12]. With huginnfork (MLP-only NVFP4) and NVIDIA (FP8 attention), every public
NVFP4 release of this model keeps something above 4 bits.

### 2.3 Sensitivity findings for the hybrid architecture

- `linear_attn.out_proj` (`ssm_out`, the engine's `gdn/output`, Q3 today) is the most KLD-sensitive tensor in a
  per-tensor sweep on Qwen3.5-9B; `in_proj_z` tolerates MXFP4 badly; FFN gate/up tolerate about 3 bits [24].
- Low-bit SSM tensors produce "endless reasoning traces" unless protected at Q5 to Q8 [23].
- The GDN recurrent path is fragile at depth in llama.cpp: instant end-of-sequence beyond about 130K on CUDA and CPU
  [30], a 25x decode slowdown past about 80K [31], and a BF16 chunked GDN prefill kernel that alone produced KLD about
  0.1 and top-1 92% on a Q8_0 model [32]. A kernel-precision regression looks like quantization loss and must be
  separated by the drift test.
- Vision: GGUF makers ship the tower as a separate mmproj; no source reports its sensitivity. The MTP block does
  not run in the ordinary forward pass, so no KLD sees its quantization [8]; only acceptance does.

### 2.4 MTP in llama.cpp and the drafting-stack precedent

Mainline llama.cpp has MTP speculation since PR #22673 (merged 2026-05-16): `--spec-type draft-mtp`,
`convert_hf_to_gguf.py --mtp` exports the head as a sidecar, `--no-nextn` drops it from the trunk, `llama-imatrix
--nextn` collects importance for the block; `src/models/qwen35.cpp` implements the Qwen3.5/3.8 head as `graph_mtp`
[33][34]. Reported acceptance (llama.cpp and vLLM count it differently):

| Setup | Acceptance | Source |
|---|---|---|
| PR #22673 author, Qwen3.6-27B and 35B-A3B, 3 drafts | about 75% steady state, "more than 2x" speedup | [33] |
| Qwen3.6-35B-A3B UD-Q4_K_XL, RTX 3090, n=2, corrected counters | 78.4%, +22.7% throughput (CI +22.1 to +23.3) | [35] repo |
| Qwen3.8-27B NVFP4-MTP GGUF, RTX 5090, n-max 4, 192K ctx | 44.3 to 49.5% across head-precision tiers; Qwen3.6-27B 69.7% | [36] |
| Qwen3.8-27B GSQ-RCO IQ3_XXS + Q6_K head, RTX 5070 Ti, n-max 2 | 0.67 at 32K; 0.70 to 0.77 at 65K | [11] |
| Qwen3.8-27B NVFP4 in vLLM | 0.897 Inferact, 0.788 unsloth | [15] |
| Qwen3.5-27B Q4_K_M, early fork | 47.5%, net slower (recurrent-state checkpoints) | [37] |

Three experiments bear on constraint 8 (drafting precision is a separate knob). Head tiers LOW (Q5_0 LM head, IQ4_XS
embeddings and MTP head), MEDIUM (Q8_0/Q6_K/IQ4_XS) and VERY-HIGH (all BF16) gave 49.0%, 44.3% and 49.5% acceptance
on Qwen3.8-27B [36]. On an L4, replacing UD-Q4_K_XL's embedded Q6_K draft output with a Q4_0 sidecar and then a Q2_K
output tensor raised mean accepted length 2.88 to 2.99 to 3.01 (cheaper reads), while requantizing the MTP block to
Q3_K lowered first-position acceptance from 0.779 (Q4_0) to 0.746 and was slower [4]. RentedNoodle found the V-cache
precision effect on acceptance flips sign with context state (q4_0 V 0.756 vs q8_0 V 0.742 short; 0.742 vs 0.819
after a long ingest) [11]. The engine's own result (Q3 MTP accepted fewer tokens than NVFP4) matches: the draft head
can go very low, the MTP block cannot, and KLD sees neither.

### 2.5 Where the forms fall

The `q3` artifact (11.72 GiB) is larger than ISTA's IQ3_S (10.99 GiB); `nvfp4` (14.52 GiB) is 1.8 GiB below
UD-Q4_K_XL. On KL the base forms and the public files describe the same checkpoint against the same kind of BF16
reference, so only text, KV cache and prefill activations differ: public 4.4 to 5.3 bpw files report 0.010 to 0.019
and top-1 95 to 96%, the 4.64 bpw `nvfp4` 0.147 and 88%, which public data associates with 2.5 to 2.7 bpw (UD-IQ2_XXS
0.146, 83%); public 3.0 to 4.1 bpw files report about 0.05 to 0.08 (Q3_K_M 0.047 to 0.056, Qwen3.5-class IQ3_S 0.061)
and top-1 about 92% (AD-IQ3_S), the 3.74 bpw `q3` 0.293 and 76%, close to bartowski's IQ2_XXS (0.301) at 2.5 bpw. The
deployed `nvfp4-compressed` (Swift 1.5, UkisAI's 1.27 M-token matrix, Swift BF16 reference) reports 0.155 and 89.4%,
within noise of the base `nvfp4` (SD4 against plain NVFP4 on the base: +0.005 +- 0.009 nats, KL 0.037), so neither the
checkpoint nor the choice between the two public matrices moves the number. The drift floor is KL 0.003 to 0.005,
so kernels are not it; NVFP4 KV adds about 0.015 on this model [11], a tenth of the `nvfp4` gap; text and sample
effects are quantified in section 4. The remainder is the rounding method and the allocation.

## 3. Swift 1.5

### 3.1 Stated by the publisher or seen in primary artifacts

- **Lineage.** `ukisai/Swift-Qwen3.8-27b` (1.0) is served in its own evaluation logs as a vLLM LoRA adapter over the
  Qwen3.8-27B BF16 base; the evals README defines the `swift` arm as "the same base with the Swift LoRA" and the card
  carries the `lora` tag [38][39]. Swift 1.5 "was made from Swift 1.0", scaling up its post-training methods with a
  focus on long-horizon, agentic and coding tasks (card, via summary) [40]. The Show HN post for 1.0 names RL (GSPO)
  and on-policy distillation, penalizing tokens associated with overthinking, plus a transfer component derived from
  BottleCap AI's ThinkingCap-Qwen3.6-27B [41]. No training dataset is named anywhere found.
- **Claims on the 1.5 card** (via summary): 58.5% fewer thinking tokens, +0.35% average score, 1.95x speed-up; the
  GGUF and NVFP4 cards and ukisai.com say 9.18x, unresolved [40][42][43]. LiveCodeBench v6 81.71 vs 76.76 base
  (1.0: 81.55), GPQA-Diamond 88.59 vs 88.28, IFBench 72.07 vs 73.53; it trails the base on IFBench, ERQA, AIME 2026
  and HMMT Nov 2025 [40][44].
- **Swift 1.0 primary logs** (cloned, 159,726 files): nine benchmarks, five seeds; MMLU-Pro 5-shot CoT base 85.47%
  (per-seed s.d. 0.12 pp) vs Swift about 84.95%; base completion mean 3,147 tokens, median 712, p90 8,432; a
  2026-09-04 LiveCodeBench "fine-grained v6" run at xhigh on 175 problems x 5 seeds: base 546/875 = 62.40%, Swift
  611/875 = 69.83%, mean completion -17.2%, paired thinking reduction 24.1%; the card's 76.76 vs 81.55 comes from a
  different grading set in the same repository [38].
- **License.** UkisAI's contribution under a "Swift Open License v1.0" with a NOTICE file; the base stays Apache
  2.0; Featherless and LLM Explorer list "other" [40][45].
- **Distribution.** BF16, GGUF, INT4 for vLLM, NVFP4, Quark FP8 for AMD, MLX, and the GSQ-RCO GGUF; bartowski
  publishes an imatrix set [46][47]. Featherless serves it at 32K with tool calling [45].
- **UkisAI's GSQ-RCO GGUF and matrix.** `ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF` exists (one search result) and
  ships `imatrix-swift15-v1mix.gguf`, 309 chunks of 4096 tokens, about 1.27 M tokens of web, reasoning, multilingual
  and code text (per the amended Euhedral docs, which use it for the deployed form) [46][74]. It is the most directly
  comparable public quantization of the served checkpoint; its bit widths, allocation and quality were not retrieved
  (search budget exhausted before the coordinator's note), a one-query follow-up.
- **Timing.** 1.0 logs dated 2026-09-04, launch coverage 2026-09-16; 1.5 pages were 2 to 4 weeks old on 2026-10-10:
  late September 2026 (approximate).

### 3.2 Unverified or absent

- Whether the LoRA touched `blk.64` (MTP) or only trunk projections, and whether 1.5 ships merged weights. One review
  says the "MTP head kept for self-speculative decoding" [48]; the launch post says 1.0 "retains text, image, and
  video support", implying the vision tower is present [41]. If the trunk moved and the head did not, the head drafts
  for shifted hidden states; acceptance on Swift vs base is the only test (section 6).
- The 58.5% figure is a median; a critic computes 41.0% mean reduction at xhigh [49]; one coding test found Swift
  emitting more tokens than the base (60,572 vs 11,744) [50].
- Sampling settings, training data, LoRA rank, MTP loss, and the GSQ-RCO GGUF's bits and quality: not found.

For quantization: the served checkpoint differs from the base by a low-rank update, so base KLD tables transfer
approximately and the engine's Swift form lands where its base forms do; the deployed form already uses UkisAI's
Swift matrix, known at genre level and a third the size of ISTA's (1.27 M vs 4.1 M tokens); and the model's selling
point is a token budget that quantization inflates (section 4.5).

## 4. Evaluation methodology

### 4.1 KL divergence versus perplexity

ikawrakow defines quantization error as PPL(Q)/PPL(fp16) - 1: the aim is distance to the same model, benchmarks are
less sensitive and slower, KLD and PPL are "closely related" and PPL is convenient; elsewhere he calls them
equivalent on an infinite corpus with PPL superior on a limited one [51][52]. His KLD-precision fix notes that the
16-bit stored base log-probabilities, floored 16 nats below the maximum, distort the stored-base PPL for large
vocabularies but not KLD; `compare_reference.py` reads exactly this format [53]. The opposing view, "Accuracy is Not
All You Need" (arXiv:2407.09141): accuracy and PPL average out symmetric per-token changes, KLD and top-1 flips do not
(Spearman 0.981 between KLD and flips on MMLU) [54]; smcleod's 2026 write-up argues KLD isolates quantization damage
from the model's own uncertainty [55]. llama.cpp's Llama-3-8B scoreboard shows why both are reported: q4_K_M raises
PPL 2.8% at KLD 0.031 and same-top-p 91.9%; q2_K raises PPL 56% at KLD 0.445 and same-top-p 71.1% [56].

### 4.2 What `llama-perplexity --kl-divergence` reports

`tools/perplexity/perplexity.cpp` scores only the second half of each `-c` chunk (`first = n_ctx/2`: 1279 positions
per 2560-token chunk, hence the engine's prefix 1281 and 1279 steps). It prints Mean PPL(Q) and PPL(base), their log
correlation, mean and ratio of PPL; Mean KLD with a Gaussian-approximation uncertainty; Maximum, 99.9%, 99.0%, 95.0%,
90.0%, Median, 10%, 5%, 1%, 0.1% and Minimum KLD; Mean delta-p with percentiles, RMS delta-p, and "Same top p" with a
binomial standard error [56]. The README's reading: symmetric delta-p percentiles mean added noise, negative skew
means the model got worse, RMS delta-p is the noise's standard deviation. `compare_reference.py` prints only mean NLL,
paired dNLL with s.e., mean KL and top-1; the percentiles and delta-p are cheap to add.

### 4.3 Typical values that put the engine's numbers in context

| Model (reference, text) | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | IQ4_XS | Q3_K_M | IQ3_S | IQ3_XXS | IQ2_XXS | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B base (BF16, wiki.test) | median 0.0003 | 0.0030 | median 0.0023 | 0.0129 to 0.0139 | | 0.0466 to 0.0564 | | | 0.30 (vs Q8_0) | [1][2] |
| Qwen3.8-27B base, unsloth UD | | 0.0029 | | UD-Q4_K_XL 0.0096, top-1 96.0% | | | | | UD-IQ2_XXS 0.146, top-1 83.0% | [2][5][6] |
| Qwen3.6-27B finetune (DavidAU) | | | | 0.0147 | 0.0172 | | IQ3_M 0.0569 | | | [57] |
| Qwen3.5 dense (unsloth, size unstated) | | | | 0.0192 | 0.0235 | | 0.0613 | | | [23] |
| Qwen3.8-Flash-Next MoE (AesSedai) | | | | 0.034 | 0.076 | | 0.160 | | | [58] |
| Llama-3-8B (f16, wikitext; imatrix rows) | 0.0014 | 0.0055 | 0.0108 | 0.028 | 0.036 | 0.084 | 0.111 | 0.184 | 0.812 | [56] |
| Llama-3-8B same-top-p | 97.7% | 96.0% | | 91.9% | | | | | q2_K 71.1% | [56] |

The Gemma-3-27B QAT study in ik_llama.cpp reports PPL only (q4_0 8.2500, iq4_xs 8.2290, q4_K_M 8.2303 vs BF16 8.2021:
0.3 to 0.6% at 4.5 to 4.7 bpw on a QAT model) [59]. A 27B-class dense model at 4.4 to 5.3 bpw sits at KL 0.01 to 0.02
and top-1 95 to 96%; at 3.5 to 4.1 bpw at KL 0.05 to 0.08 and top-1 about 92%; KL 0.15 to 0.30 with top-1 76 to 83% is
2.5 to 2.7 bpw territory. The engine's 0.147 at 4.64 bpw and 0.293 at 3.74 bpw are unusual by about one full bit of
effective precision, after allowing roughly 0.015 for the KV cache.

### 4.4 Texts and sample size

Can the single 1279-token document explain part of it? Genre first: the document is the engine's own architecture
documentation (Markdown with Java and CUDA identifiers), text the model never saw, against wikitext for the public
numbers. KLD does depend on text (unsloth's chat-calibrated quants score worse on wikitext at 512 context than on
chat [23]; imatrix sets that overlap wikitext lower wiki.test PPL by 1 to 2% [52]); a factor of 2 between genres is
plausible, 5 to 10x is not (estimate). Sampling second: the repository's s.e. of 0.008 (NVFP4) and 0.025 (Q3) on dNLL
over 1279 tokens imply per-token standard deviations of 0.29 and 0.89 nats; KL per token is heavy-tailed, and if its
standard deviation is 2 to 3x its mean, the s.e. of the engine's mean KL is about 0.008 to 0.012 (NVFP4) and 0.016 to
0.025 (Q3) (heuristic). Same-top-1 on 1279 tokens has a binomial s.e. of 0.9 to 1.2 pp. Third, the engine's number
bundles the NVFP4 KV cache and two-term FP4 prefill activations, which no GGUF number does. Tokens for a target
resolution, extrapolated from the measured s.e. by 1/sqrt(N):

| Target s.e. on dNLL | NVFP4-class (sd 0.29) | Q3-class (sd 0.89) |
|---|---|---|
| 0.01 nats | about 0.8K tokens | about 8K tokens |
| 0.005 nats | about 3.3K | about 32K |
| 0.002 nats | about 20K | about 200K |

Public practice: bartowski scores 100 chunks of wiki.test.raw; the llama.cpp scoreboard uses the whole wikitext-2
test (about 300K tokens, KLD s.e. 2e-4 to 3e-3); ubergarm uses a 1.6 MiB transcript corpus and reports median and 99%
KLD [60]; discussion #65 uses 87 chunks at 4096 context [8]. Differences of 0.01 in mean KL between two engine forms
are not resolvable on 1279 tokens; 32K scored tokens (25 chunks of 2560) brings the s.e. to about 0.002 to 0.005.

### 4.5 Downstream deltas versus KLD

"Statistically-Lossless Quantization" (arXiv:2605.02404) reports baseline MMLU and GSM8K accuracy whenever KL stays
below 0.01 [61]. "Displacement Is Not Direction" (arXiv:2606.19558) finds KLD and PPL correlate with benchmarks only
across badly degraded models and are uncorrelated near baseline: they filter damaged candidates, they do not rank
near-lossless ones [62]. Flips track KLD tightly [54]. ISTA's card fits: IQ3_S equal to BF16 on AIME25 and LCB at 3.5
bpw, while 2.5 bpw drops AIME25 3.3 points and LCB 9.1 [9]. Kaitchup finds every standard Qwen3.8-27B GGUF from Q4 down
clears a 95% recovery bar and that differences appear in token efficiency [63].

For a reasoning-efficiency tune that second axis is the important one: "Quantization Inflates Reasoning"
(arXiv:2606.25519) and "Quantized Reasoning Models Think They Need to Think Longer" (arXiv:2606.00206) report longer
traces under aggressive quantization and a strong dependence on the calibration set [64][65]. Benchmark variance is
large relative to the deltas of interest (binomial s.e., one seed, computed): AIME25 (30) 3.1 pp, GPQA-Diamond (198)
2.2 pp, LiveCodeBench v6 (1,055) 1.2 pp, GSM8K (1,319) 0.6 pp, MMLU-Pro (12,032) 0.3 pp. UkisAI runs five seeds
everywhere and still sees 0.1 to 0.2 pp per-seed spread on MMLU-Pro [38]. KLD is the instrument; benchmarks are the
gate.

### 4.6 Long context under quantization

The EMNLP 2025 study (arXiv:2505.20276) finds 4-bit weight quantization loses up to 23% on average at 128K and up to
16% on RULER/OneRuler regardless of method (AWQ 1.8%, GPTQ-int4 2.7%, BNB-nf4 6.9% average), with 8-bit the safe
choice [66]. For Qwen3.8-27B there is only RentedNoodle's needle set (6/6 at about 20K tokens, IQ3_XXS with q8_0/q4_0
KV; NVFP4 KV "validated to 96K") [11] and hanxiao's accepted-length-at-depth curve to 152K [4]. Given the GDN issues of
section 2.3, long-context failures here can come from the recurrent path rather than weights or KV. Teacher-forced KL
at depth (the last 1279 tokens after a 64K or 128K prefix) exercises weights, the NVFP4 KV at depth and 48 layers of
GDN state together; a needle set separates retrieval.

## 5. Calibration data

| Maker | Set | Size, context | Notes |
|---|---|---|---|
| bartowski | calibration_datav3 (Dampf on Kalomaze); later v5, v6 | 2,481 lines, 279,515 bytes; described by a third party as scientific papers, code, dialogue, multilingual, Wikipedia, legal, fiction | gist host blocked here; contents known only by description [67][68] |
| unsloth Dynamic 2.0/3.0 | hand-curated chat, long-context and tool-calling text | 300K to 1.5M tokens | argues wikitext calibration overfits wikitext scores; evaluates KLD on wikitext-class sets for comparability [22][23] |
| ubergarm | calibration_data_v5_rc at ctx 512 | | "fancy calibration corpus nor extra context length ... not clearly proven" [60] |
| ikawrakow | wiki.train.raw, many batches | | unsloth's and bartowski's imatrix lowered wiki.test PPL more than his (8.88 and 8.97 vs 9.07 on one recipe), read as test overlap [52] |
| llama.cpp scoreboard | wikitext 1K to 10M tokens | | "no consistent improvement from using more Wikitext tokens" [56] |
| ISTA GSQ | OpenThoughts (reasoning traces) for Qwen3.5-class and Kimi, FineWeb-Edu for Llama | 4096 samples x up to 4096 tokens; GPTQ 512 samples | repo configs [26] |
| ISTA `imatrix-qwen3.8-27b.gguf` (base) | text not found | 1000 x 4096 tokens, about 4.1 M | used for the engine's `q3` and `nvfp4` [9] |
| UkisAI `imatrix-swift15-v1mix.gguf` (Swift 1.5) | web, reasoning, multilingual, code | 309 x 4096 tokens, about 1.27 M | used for the deployed `nvfp4-compressed`; from the Swift GSQ-RCO GGUF repo [46][74] |

Research on reasoning models points one way. "Quantization Hurts Reasoning?" (arXiv:2504.04823) calibrated on the
model's own reasoning outputs and found their activations far from pre-training text [69]; "Think Before You Prune"
(arXiv:2511.18864) finds self-generated reasoning data better than pre-training text, with a selection step [70];
both 2026 inflation papers adopt the target model's own math traces after finding they beat WikiText on accuracy and
length [64][65]; AYOT (arXiv:2608.01078) feeds the model's own traces as context and keeps improving with more
calibration tokens [71]; LAQuant (arXiv:2605.08755) selects reasoning-domain calibration by Hessian-subspace
alignment [72]. GSQ's OpenThoughts choice is the same idea applied by a GGUF maker.

Recommendation for the deployed checkpoint: keep UkisAI's `imatrix-swift15-v1mix.gguf` as the documented default
(public, Swift-specific, contains reasoning text) and test one controlled variant: text generated by BF16 Swift 1.5
itself (its chat template, thinking on, `--parse-special` so template tokens count) on several hundred prompts of 1K
to 4K tokens across math, code, tool use and long documents, disjoint from every evaluation prompt, with a minority
of plain prose and code; `llama-imatrix --in-file` can merge it with UkisAI's file. Judge the variant on T4 and T2 of
section 6, not on T1 alone. Convert `q3` and `nvfp4` from Swift 1.5 before comparing them with the deployed form: a
base conversion scored against the base BF16 says nothing about what ships. Record the matrix file's hash, chunk
count and `--ctx-size`, the text hash and token count when the text is the engine's own, the llama.cpp commit and the
checkpoint hash in the artifact metadata.

## 6. Recommended protocol for the engine

Reuse the harness of `00-baseline.md` section 9; change the inputs and the report.

**Texts** (Qwen tokenizer; all held out from calibration):

| Text | Purpose | Size |
|---|---|---|
| T1 `chat-corpus-v1-document.md`, prefix 1281, 1279 steps | continuity with every number so far | 1279 scored tokens |
| T2 wikitext-2 `wiki.test.raw` | the only text with public Qwen3.8-27B KLD at every width (bartowski, 100 chunks) | 25 chunks of 2560 = 32K scored tokens minimum, 100 chunks preferred |
| T3 code: permissively licensed source in 4 languages, not from the engine repository | code behaviour | 32K scored tokens |
| T4 Swift 1.5 generations: BF16 Swift 1.5 answers with full thinking traces on 100 to 300 prompts (GSM8K-style, LiveCodeBench-style, tool use), template tokens included | deployment distribution; thinking-specific damage | 32K scored tokens; a disjoint set of the same kind is the calibration text |

**Reference and checkpoint.** Every form is converted from and scored against the deployed checkpoint, Swift 1.5
(`convert_hf_to_gguf.py --outtype bf16`, then `llama-perplexity --kl-divergence-base` at `-c 2560` on the CPU, as
`docs/QUALITY.md` now describes); base conversions are kept only as the bridge to public numbers on T2. Each number
names its checkpoint and matrix. Where CPU BF16 is impractical at depth, a Q8_0 GGUF may stand in (public KLD of Q8_0
to BF16 on this model is 0.0006 to 0.0014), stated in the report.

**Metrics per text.** Mean KL with s.e.; median, 95%, 99% and maximum KL; mean dNLL with paired s.e.; same-top-1 with
binomial s.e.; mean and RMS delta-p and the 1% and 99% delta-p percentiles. Extend `compare_reference.py`; all are
available from the stored reference and the report.

**Attribution runs, once per artifact.** (a) Weight-only KL: dequantize the artifact's tensors to BF16, write a GGUF,
run `llama-perplexity --kl-divergence` on T2 with an F16 KV cache; this is the number comparable to bartowski's column
and separates weight error from KV and prefill-activation error. (b) The drift floor (existing test). (c) The engine's
full number. (c) minus (a) minus (b) is the cost of the KV cache and FP4 prefill, which no other chapter can see.

**Decision thresholds (external yardsticks).** KL below 0.01 is the lossless class in the literature and Q5_K_M to
Q6_K on this model; 0.010 to 0.020 is Q4_K_M/UD-Q4_K_XL class, top-1 95 to 96%; 0.05 to 0.08 is Q3_K_M/IQ3_S class,
top-1 about 92%; above 0.15 is 2-bit class. A change between two forms counts only if it exceeds two standard errors on
T2 and T4 and has the same sign on all four texts.

**Long context.** Teacher-forced KL on the last 1279 tokens after 32K, 64K and 128K prefixes built from T2 and T4
material (three prefixes each), against the reference at the same depth; a needle set of at least 9 depths at 32K,
64K and 128K, greedy; before and after any KV or GDN-state change. A depth where the engine's KL rises faster than
the reference's own NLL is a GDN or KV defect, not quantization.

**Downstream.** GSM8K (1,319, s.e. 0.6 pp) and a fixed 2,000-question MMLU-Pro subset (s.e. 0.8 pp) as gates;
LiveCodeBench v6 (1,055) when code changes are claimed; AIME25 only with five seeds. Record completion and thinking
tokens (mean, median, p90) against the BF16 Swift 1.5 arm on every run: a form that keeps accuracy but adds 20%
thinking tokens has lost most of what Swift 1.5 bought.

**Drafting stack.** MTP accepted tokens per verification on the fixed corpus, with the Swift 1.5 MTP block and with the
base model's block grafted in. Equal acceptance means the block was not retrained and the base head is a valid
fallback; higher acceptance with the Swift block answers section 3.2.

**Record with every imatrix and artifact.** Checkpoint hash, llama.cpp commit, matrix file name and SHA-256 (ISTA's
base file or UkisAI's Swift file today), calibration text identity, SHA-256 and token count when the text is the
engine's own, `--ctx-size`, chunk count, `--parse-special`, `--nextn` and `--process-output` flags, and the output
of `llama-imatrix --show-statistics` for the file actually used.

## 7. Risks and open questions

- Public KLD numbers are for the base model on wikitext; the engine's base forms match the checkpoint but differ in
  text and bundle the KV cache. The weight-only run on T2 is the only exact comparison; until then the "one bit of
  effective precision" gap carries a plausible 2x uncertainty from text.
- Only `nvfp4-compressed` describes what ships; `q3` and `nvfp4` are base conversions. Nothing in `00-baseline.md`
  yet says how the Q3 route behaves on Swift 1.5.
- Every Hugging Face card here was read through search summaries; values could be transcription errors in those
  summaries. bartowski's IQ3_S and IQ4_XS rows for this model were not visible at all.
- ISTA's "task-lossless" claim and table are publisher numbers; the one independent test used 60 questions.
- Swift 1.5's MTP training, vision tower, license terms and sampling guidance are undocumented in what was found;
  read the license before redistributing artifacts.
- Both importance matrices are public files whose texts are known at genre level (UkisAI) or not at all (ISTA); they
  can be fixed by hash but not reproduced, and UkisAI's is a third the size of ISTA's, a confound when compared.
- UkisAI's GSQ-RCO GGUF of Swift 1.5 has unknown bit widths and quality here; its card is the first thing to read.
- MTP acceptance on Qwen3.8-27B is 44 to 50% at n-max 4 in llama.cpp, 67 to 77% at n-max 2, 79 to 90% in vLLM;
  the metrics differ and none is the engine's "tokens per verification".
- The GDN path has open llama.cpp correctness issues beyond 80K to 130K tokens; validate the reference's own NLL
  curve against depth before trusting long-context KL.
- The search budget ran out before the UkisAI GSQ-RCO card, bartowski's full percentile table, the unsloth Dynamic
  3.0 page, the ISTA imatrix text, the Swift Open License text and Swift 1.5 sampling guidance could be queried.

## 8. References

1. bartowski/Qwen3.8-27B-GGUF. https://huggingface.co/bartowski/Qwen3.8-27B-GGUF (via summary)
2. unsloth/Qwen3.8-27B-GGUF discussion 49, "Independent KLD benchmark". https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/49
3. unsloth discussion 74, "Introducing Unsloth Dynamic v3 Qwen3.8". https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/74 ; https://unsloth.ai/docs/basics/dynamic-3.0-ggufs
4. hanxiao, Qwen3.8-27B UD-Q4_K_XL on an NVIDIA L4 (README, cloned). https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4
5. kingy.ai, "Best Qwen3.8-27B GGUF". https://kingy.ai/blog/qwen3-8-27b-best-quantization-gguf/
6. akopytko/Qwen3.8-27B-NVFP4-GGUF. https://huggingface.co/akopytko/Qwen3.8-27B-NVFP4-GGUF (via summary)
7. AtomicChat/Qwen3.8-27B-GGUF. https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF (via summary)
8. Qwen/Qwen3.8-27B discussion 65. https://huggingface.co/Qwen/Qwen3.8-27B/discussions/65
9. ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF. https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF (via summary)
10. ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ. https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ
11. RentedNoodle/den_llama.cpp (README, cloned). https://github.com/RentedNoodle/den_llama.cpp
12. cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF https://huggingface.co/cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF ; https://forums.developer.nvidia.com/t/qwen3-8-27b-at-256k-on-a-24-gb-blackwell-target-gpu-imatrix-nvfp4-mtp-55-4-tok-s/380456
13. nvidia/Qwen3.8-27B-NVFP4. https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4 (via summary)
14. unsloth/Qwen3.8-27B-NVFP4. https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4 (via summary)
15. vLLM recipes, Qwen3.8-27B. https://recipes.vllm.ai/Qwen/Qwen3.8-27B
16. MiaAI-Lab/Qwen3.8-27B-NVFP4-RTX-5090 (README, cloned). https://github.com/MiaAI-Lab/Qwen3.8-27B-NVFP4-RTX-5090 ; https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4 ; https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4
17. huginnfork/Qwen3.8-27B-NVFP4A16. https://huggingface.co/huginnfork/Qwen3.8-27B-NVFP4A16 (via summary)
18. TelperionAI/Qwen3.8-27B-NVFP4-AWQ-GPTQ. https://huggingface.co/TelperionAI/Qwen3.8-27B-NVFP4-AWQ-GPTQ
19. Qwen/Qwen3.8-27B-FP8 https://huggingface.co/Qwen/Qwen3.8-27B-FP8 ; no official GGUF/AWQ/GPTQ/MLX per https://kingy.ai/blog/qwen3-8-27b-specs-benchmarks-local-hardware/
20. https://huggingface.co/mlx-community/Qwen3.8-27B-4bit ; https://huggingface.co/mlx-community/qwen3.8-27b-mtp-4bit ; https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit ; https://huggingface.co/PocketAiHub/Qwen3.8-27B-MLX
21. NVIDIA forum, "Qwen3.8-27B-NVFP4 @ TP=4" (doth4580 model). https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-tp-4-running-at-80-tok-s-sustained-peak-at-92-tok-s/380904
22. Unsloth Dynamic 2.0 GGUFs. https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs
23. Unsloth, Qwen3.5 GGUF Benchmarks https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks ; Kaitchup, "Qwen3.5 9B MoQ" https://kaitchup.substack.com/p/qwen35-9b-moq-inside-a-strong-36
24. kaushall13, Qwen3.5-9B quantization plan. https://github.com/kaushall13/qwen3.5-9b-quantization/blob/main/docs/qwen35_9b_q4_quantization_plan.md
25. GSQ, arXiv:2604.18556. https://arxiv.org/abs/2604.18556
26. IST-DASLab/GSQ (README and configs, cloned). https://github.com/IST-DASLab/GSQ
27. RCO, arXiv:2605.00649. https://arxiv.org/abs/2605.00649
28. IST-DASLab/RCO (README, cloned). https://github.com/IST-DASLab/RCO
29. zephel01, GSQ-RCO test with and without MTP. https://note.com/zephel01/n/nb849cea8359d
30. llama.cpp issue 27756. https://github.com/ggml-org/llama.cpp/issues/27756
31. llama.cpp issue 27623. https://github.com/ggml-org/llama.cpp/issues/27623
32. stew675/llama-cpp-rdna-boosts issue 113. https://github.com/stew675/llama-cpp-rdna-boosts/issues/113
33. llama.cpp PR 22673, "llama + spec: MTP Support". https://github.com/ggml-org/llama.cpp/pull/22673
34. llama.cpp at commit 23b0202a (cloned): docs/speculative.md, tools/imatrix/README.md, convert_hf_to_gguf.py, src/models/qwen35.cpp, src/llama-quant.cpp. https://github.com/ggml-org/llama.cpp
35. thc1006/qwen3.6-speculative-decoding-rtx3090 (README, cloned). https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090
36. PierpaoloPernici, Qwen3.8-27B NVFP4-MTP benchmark on RTX 5090 (gist, via summary). https://gist.github.com/PierpaoloPernici/2e6f6f42965d531b364f39ab0e8a52ad
37. NodeNestor/qwen3.5-27b-mtp-llamacpp (README, cloned). https://github.com/NodeNestor/qwen3.5-27b-mtp-llamacpp
38. UkisAI/Swift-Qwen3.8-27B-evals (cloned). https://github.com/UkisAI/Swift-Qwen3.8-27B-evals
39. ukisai/Swift-Qwen3.8-27b. https://huggingface.co/ukisai/Swift-Qwen3.8-27b (via summary)
40. ukisai/Swift-1.5-Qwen3.8-27b. https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b (via summary)
41. Show HN: Swift-Qwen3.8-27B. https://news.ycombinator.com/item?id=49727511
42. https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF ; https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-NVFP4
43. https://ukisai.com/swift-1-5-27b ; https://ukisai.com/news/introducing-swift
44. HackerNoon, "Swift-Qwen3.8-27B Cuts Reasoning Tokens Without Sacrificing Much Accuracy". https://hackernoon.com/swift-qwen38-27b-cuts-reasoning-tokens-without-sacrificing-much-accuracy
45. https://featherless.ai/models/ukisai/Swift-1.5-Qwen3.8-27b ; https://llm-explorer.com/model/ukisai%2FSwift-1.5-Qwen3.8-27b,3jHOVbzg1EPZp5GUvwUGrh
46. ukisai Swift 1.5 collection https://huggingface.co/collections/ukisai/swift-15-27b ; https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/tree/main ; https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-INT4 ; https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-Quark-FP8-dynamic-AMD
47. bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF. https://huggingface.co/bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF
48. MindStudio, "Qwen 3.8 27B Swift vs Original". https://www.mindstudio.ai/blog/qwen3-8-27b-swift-vs-original-thinking
49. traictory, "UkisAI's Swift-Qwen3.8-27B sells a median, not a mean". https://traictory.com/news/2026-09-16-ukisai-swift-qwen38-27b
50. HackerNoon, "The Qwen3.8-27B Variant Built to Stop Overthinking". https://hackernoon.com/the-qwen38-27b-variant-built-to-stop-overthinking
51. ik_llama.cpp discussion 8. https://github.com/ikawrakow/ik_llama.cpp/discussions/8
52. ik_llama.cpp discussion 359, "Qwen3 quantization experiments". https://github.com/ikawrakow/ik_llama.cpp/discussions/359
53. ik_llama.cpp PR 325, "Fix KLD precision". https://github.com/ikawrakow/ik_llama.cpp/pull/325
54. "Accuracy is Not All You Need", arXiv:2407.09141. https://arxiv.org/abs/2407.09141
55. smcleod, "Measuring Model Quantisation Quality with KL Divergence" (2026-04). https://smcleod.net/2026/04/measuring-model-quantisation-quality-with-kl-divergence/
56. llama.cpp tools/perplexity/README.md and perplexity.cpp (cloned). https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md
57. DavidAU Qwen3.6-27B imatrix GGUF card. https://huggingface.co/DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF (via summary)
58. AesSedai/Qwen3.8-Flash-Next-GGUF. https://huggingface.co/AesSedai/Qwen3.8-Flash-Next-GGUF (via summary)
59. ik_llama.cpp discussion 334, iq4_ks on gemma-3-27b-it-qat. https://github.com/ikawrakow/ik_llama.cpp/discussions/334
60. ik_llama.cpp discussion 434, "Quant Cookers Basic Guide". https://github.com/ikawrakow/ik_llama.cpp/discussions/434
61. "Statistically-Lossless Quantization of Large Language Models", arXiv:2605.02404. https://arxiv.org/abs/2605.02404
62. "Displacement Is Not Direction", arXiv:2606.19558. https://arxiv.org/abs/2606.19558
63. Kaitchup, "Qwen3.8 27B GGUF Benchmark: Q4 to Q1". https://kaitchup.substack.com/p/qwen38-27b-gguf-benchmark-q4-to-q1 (seen via its Flash-Next companion post)
64. "Quantization Inflates Reasoning", arXiv:2606.25519. https://arxiv.org/abs/2606.25519
65. "Quantized Reasoning Models Think They Need to Think Longer, but They Do Not", arXiv:2606.00206. https://arxiv.org/abs/2606.00206
66. "Does quantization affect models' performance on long-context tasks?", arXiv:2505.20276, EMNLP 2025. https://arxiv.org/abs/2505.20276
67. bartowski calibration gists https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8 ; https://gist.github.com/bartowski1182/82ae9b520227f57d79ba04add13d0d0d ; https://huggingface.co/blog/bartowski/imatrix-dataset
68. lemon07r/bartowski-imatrix-v3-semantic. https://huggingface.co/datasets/lemon07r/bartowski-imatrix-v3-semantic
69. "Quantization Hurts Reasoning?", arXiv:2504.04823. https://arxiv.org/abs/2504.04823
70. "Think Before You Prune", arXiv:2511.18864. https://arxiv.org/abs/2511.18864
71. "Attend to Your Own Thoughts", arXiv:2608.01078. https://arxiv.org/abs/2608.01078
72. "LAQuant", arXiv:2605.08755. https://arxiv.org/abs/2605.08755
73. "A KL Lens on Quantization", arXiv:2604.13440 (KL-guided sensitivity for hybrid SSM-Transformer models; abstract only). https://arxiv.org/abs/2604.13440
74. Euhedral repository, `docs/QUALITY.md` and PR 119 at commit 269a890 (source checkpoints, the two importance matrices with their chunk counts, and the measured numbers), as summarized in `00-baseline.md`.
