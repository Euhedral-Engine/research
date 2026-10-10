# Public baselines and evaluation: where the four forms stand, and how to measure

This chapter is the external yardstick for the rest of the folder. It collects every public quantization of
Qwen3.8-27B found on 2026-10-10, the recipes behind the best ones, what is known about the Swift 1.5 checkpoint
the engine serves, and the evaluation practice the engine should adopt. Sources are of three kinds and are
labelled as such: repositories cloned and read (llama.cpp at commit 23b0202a, ik_llama.cpp with its bundled
discussion archive, IST-DASLab/GSQ and /RCO, UkisAI/Swift-Qwen3.8-27B-evals, five community llama.cpp forks),
Hugging Face model cards and forum posts seen only through search-result summaries (huggingface.co itself could
not be fetched; such numbers are marked "card, via summary"), and papers identified by arXiv ID from search
results (abstract-level claims only). Nothing below was recalled from memory.

Footing. The engine's quality numbers in `00-baseline.md` (PR 119 at commit 269a890) rest on two checkpoints:
`q3`, `q3-compressed` and `nvfp4` are conversions of the base Qwen3.8-27B with ISTA-DASLab's
`imatrix-qwen3.8-27b.gguf`, measured against the base BF16 (PPL 8.39 on the quality document); the deployed
`nvfp4-compressed` is a conversion of Swift 1.5 with UkisAI's published `imatrix-swift15-v1mix.gguf`, measured
against Swift 1.5's BF16 (PPL 8.51). Every public number in section 2 is for the base model unless the row says
otherwise, measured on wikitext-class text with an F16 KV cache and unquantized activations. For the three base
forms the model footing therefore matches the public one; what still differs is the text and what the number
bundles (the engine's includes its NVFP4 KV cache and, for the NVFP4 forms, two-term FP4 activation quantization in
prefill). Section 6 says how to remove those differences.

## 1. Summary

At equal file size the engine's forms sit far below the public Pareto frontier for this model on the only metric
both report, KL divergence to BF16, and for the three base-model forms the comparison is on the same checkpoint:
`nvfp4` (14.52 GiB, 4.64 bpw over 26.9 B parameters) measures KL 0.147 and top-1 agreement 88%, where public 4.4 to
5.3 bpw GGUFs of the base model measure KL 0.010 to 0.019 and top-1 95 to 96%; `q3` (11.72 GiB, 3.74 bpw) measures
KL 0.293 and top-1 76%, where public 3.0 to 4.1 bpw GGUFs measure KL 0.05 to 0.08 and top-1 about 92%, and only 2.5
to 2.7 bpw files reach KL 0.15 to 0.30 with top-1 about 83%. The deployed `nvfp4-compressed`, converted from Swift
1.5 with UkisAI's matrix and measured against Swift's own BF16, lands at KL 0.155 and 89.4%, so the checkpoint
footing is not what separates the engine from the public files. The gap is 5 to 10x and cannot be explained by the
single short document (its sampling error is about 0.01 to 0.025 in KL) or by the text genre alone (plausibly a
factor of 2, an estimate); the NVFP4 KV cache accounts for roughly 0.015 of it on this model (one same-GPU
measurement). The best public recipes share three features the engine lacks: the LM head and the GDN output
projection are kept at 5 to 8 bits, the per-tensor bit allocation is searched rather than fixed (ISTA's GSQ-RCO:
Gumbel-Softmax refinement of the codes plus a Riemannian budget search over GGUF types, reporting AIME25 and
LiveCodeBench v6 equal to BF16 at 3.5 bpw), and the quantizer uses an importance-weighted search with error
feedback or learned assignments rather than per-group scale search. Swift 1.5 is a LoRA-derived
reasoning-efficiency tune (the 1.0 evals repo serves it as an adapter over the BF16 base); its card claims 58.5%
fewer thinking tokens and +0.35% average score, with LiveCodeBench v6 81.71 against 76.76, under a "Swift Open
License v1.0"; whether the MTP block was touched is undocumented. UkisAI also published a GSQ-RCO GGUF of Swift 1.5
with its own importance matrix (309 chunks of 4096 tokens, about 1.27 M tokens of web, reasoning, multilingual and
code text), the matrix the deployed form already uses; that release is the most directly comparable public
quantization of the served checkpoint, but its bit widths and reported quality could not be retrieved in this
session. The recommended protocol is: evaluate every form on the deployed checkpoint; keep the 1279-token document
for continuity, add wikitext-2 test (to read against bartowski's numbers), a code set, and Swift 1.5's own thinking
generations; score at least 32K tokens per text; report mean KL with standard error, median and 99th percentile,
same-top-1 and RMS delta-p from `llama-perplexity --kl-divergence`; isolate weight error from KV error by
dequantizing artifacts back to a GGUF; add teacher-forced KL at 64K and 128K depth, a needle set, MTP acceptance
against the base and the Swift head, and thinking-token counts on GSM8K and LiveCodeBench, because quantization
inflates reasoning length and that is the one property Swift 1.5 was trained for.

## 2. Public quantizations of Qwen3.8-27B

### 2.1 Table

Bits per weight are computed from the published file size over 26.9 B parameters (text tower plus MTP block,
vision excluded; ISTA's own card implies the same base: 53.8 GB BF16 = 16.00 bpw). GiB figures were converted at
2^30, GB at 10^9. "KLD" is mean KL(BF16 || quant) unless noted; different rows use different texts and references
and are comparable only within a source.

| Release | Method and allocation | File, bpw | Calibration | Reported quality | Source |
|---|---|---|---|---|---|
| bartowski/Qwen3.8-27B-GGUF (base) | llama.cpp k/i-quants with imatrix; two layouts per type ("standard" llama.cpp mix, and a "computed" per-tensor layout) | Q4_K_M 16.55 GiB = 5.28 bpw; Q3_K_M about 13 GiB (estimate, 0.79x of Q4_K_M) | bartowski imatrix (calibration_datav3 family) | wiki.test.raw, 100 chunks, vs BF16, base PPL 6.744. Mean KLD: Q6_K 0.0030; Q4_K_M 0.0129 standard, 0.0139 computed; Q3_K_M 0.0466 standard, 0.0564 computed (standard file 8.2% smaller). Percentiles: Q8_0 median 0.00033, 99% 0.0071; Q5_K_M median 0.0023, 99% 0.051; Q4_K_M median 0.0058, 99% 0.142 | [1] card, via summary |
| bartowski IQ2_XXS (base) | i-quant, imatrix | about 2.5 bpw | same | KLD 0.301, PPL 9.12 against a Q8_0 reference on wikitext-2 (independent run) | [2] |
| unsloth/Qwen3.8-27B-GGUF UD-IQ2_XXS (base) | Dynamic 2.0/3.0 mixed layers | 8.39 GiB = 2.68 bpw | unsloth chat-and-tool-call calibration | KLD 0.146 mean, 0.076 median, top-1 82.98% (independent, vs Q8_0, wikitext-2) | [2] |
| unsloth UD-Q2_K_XL (base) | Dynamic 3.0 | 9.83 GB = 2.92 bpw | same | "about 8% more accurate on top-1 than the next best option" (publisher) | [3] |
| unsloth UD-Q4_K_XL (base, 2026-08-19 rebuild) | Dynamic 3.0; blk.64 MTP block embedded (272 MiB), token_embd 682 MiB, output Q6_K 994.6 MiB | 17,559,178,144 B = 5.22 bpw | same | KL 0.00955 vs BF16, top-1 96.02% (third-party common test); unsloth raw runs on unnamed quants: 0.018772 +- 0.000201 and 0.013652 +- 0.000229 | [4][3][5] |
| unsloth Q6_K and Q4_0 (base) | plain llama.cpp types | | | wikitext vs BF16, base PPL 6.7945: Q6_K 0.0029, Q4_0 0.028 | [6] card, via summary |
| AtomicChat/Qwen3.8-27B-GGUF (base) | llama.cpp types | Q8_0 to IQ1_M | | KLD about 0.0006 (Q8_0) to 0.34 (IQ1_M) | [7] card, via summary |
| Qwen discussion #65 poster, "AD-IQ3_S" (base) | own per-tensor allocation, compared with 20 community GGUFs in one harness | 13.8 GB = 4.10 bpw | | eval_neutral held-out, ctx 4096, 87 chunks, BF16 reference: top-1 92.4%; 33% less drift than unsloth Q3_K_M at the same size; unsloth UD 2-bit wins below 11 GB; Q6/Q8 from any publisher equivalent above 25 GB | [8] |
| ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF (base) | GSQ codes, RCO per-tensor type allocation; IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S; "-mtp" variants; mmproj published | 8.4 GB = 2.50; 9.3 GB = 2.77; 10.1 GB = 3.00; 11.8 GB = 3.51 bpw | GSQ calibration (OpenThoughts for Qwen3.5-class configs in the repo); imatrix file published | Card table: BF16 AIME25 100.00, GPQA-D 89.90, LCB v6 85.71, zero-shot avg 74.34. IQ2_XS 96.67 / 84.85 / 76.57 / 74.54. IQ2_S 100.00 / 86.36 / 82.29 / 75.70. IQ3_XXS 100.00 / 88.89 / 84.57 / 74.81. IQ3_S 100.00 / 89.39 / 85.71 / 74.47 (100.2% recovery), wikitext2 PPL 7.07, C4 11.76, FineWeb-Edu 8.34. Card calls IQ3_S "task-lossless" | [9][10] card, via summary |
| ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF (Swift 1.5, the deployed checkpoint) | GSQ-RCO GGUF release by UkisAI with its own importance matrix `imatrix-swift15-v1mix.gguf` | bit widths and sizes not retrieved | 309 chunks x 4096 tokens, about 1.27 M tokens of web, reasoning, multilingual and code text (per the amended Euhedral docs) | reported quality not retrieved: the repository appeared in one search result and the search budget ran out before its card could be queried; whether GSQ was rerun on Swift or ISTA's base allocation reused is unknown | [46][74] |
| RentedNoodle IQ3_XXS trunk + Q6_K MTP head (uncensored base) | packed from ISTA's published per-tensor RCO allocation with a custom imatrix | 9.75 GiB = 3.11 bpw | custom "OrcaRouter-native" imatrix | WikiText-2 PPL 6.1745 +- 0.149 (40 x 512 chunks; no same-harness BF16 given); needle 6/6 at a 15K-word haystack; MTP n-max 2 acceptance 0.67 at 32K, 0.70 to 0.77 at 65K, on an RTX 5070 Ti 16 GB | [11] repo |
| cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF (base) | hybrid GGUF: NVFP4 bulk, sensitive tensors at Q5_K/Q6_K, MTP embedded | 17.1 GB = 5.09 bpw | imatrix | short WikiText-2 control: 6.1197 vs Q4_1 6.1127 (author calls the 0.11% gap below the check's error) vs a stock NVFP4 build 6.4949 | [12] card and forum post, via summary |
| nvidia/Qwen3.8-27B-NVFP4 (base) | ModelOpt: NVFP4 on MLP and lm_head, FP8 on attention; 262K context | about 24.6 GiB in vLLM | | none in summary | [13] card, via summary |
| unsloth/Qwen3.8-27B-NVFP4 (base) | "Dynamic V3.0 (preview)" NVFP4 | | | "92 to 97% top-1 accuracy", 1.5x faster than BF16 (publisher); vLLM MTP acceptance 0.788 (recipe page) | [14][15] |
| Inferact, RadixArk NVFP4 (base) | NVFP4 for vLLM/SGLang | about 22 GiB (RadixArk) | | Inferact MTP acceptance 0.897 in vLLM (recipe page); RadixArk + TurboQuant 4-bit KV + MTP-3 at about 160 tok/s on an RTX 5090 | [15][16] |
| huginnfork/Qwen3.8-27B-NVFP4A16 (base) | NVFP4 weights, BF16 activations; only MLPs quantized, attention and vision tower BF16 | | | "3% perplexity increase over bf16" | [17] card, via summary |
| TelperionAI/Qwen3.8-27B-NVFP4-AWQ-GPTQ (base) | AWQ scaling + GPTQ into NVFP4 | | | none in summary | [18] |
| Qwen/Qwen3.8-27B-FP8 (base, official) | fine-grained FP8, block 128 | about 28 GB | | none | [19] |
| MLX: mlx-community 4-bit and MTP-4bit, lmstudio-community 4-bit, PocketAiHub 4/6/8-bit | mlx_lm affine quantization | 4-bit 14.98 GiB = 4.78 bpw; 8-bit 27.50 GiB | | none | [20] |
| Qwen official GGUF / AWQ / GPTQ / MLX | none found at launch (one review); the only official quantized release is FP8 | | | | [19] |
| doth4580/Qwen3.8-27B-NVFP4-FULL | model built for the "veloGB10" engine on DGX Spark, equivalent to `--quantize all` on the BF16 weights per its author; no card found; its speed claims were retracted by the author | | | | [21] |
| bernhardbrieger/Qwen3.8-27B-NVFP4 | not found by search | | | | |
| RedHatAI / Neural Magic, mradermacher | no Qwen3.8-27B release found by search | | | | |

The engine's forms on the same footing (file size, 26.9 B): `q3` 11.72 GiB = 3.74 bpw, `q3-compressed`
10.16 GiB = 3.24 bpw, `nvfp4` 14.52 GiB = 4.64 bpw (all three base conversions with ISTA's matrix),
`nvfp4-compressed` 13.77 GiB = 4.40 bpw (Swift 1.5 with UkisAI's matrix). The engine files carry a 0.25 to 0.35 GiB
vocabulary-trimmed draft head that no GGUF has; without it `q3` is 3.66 and `nvfp4` 4.52 bpw. GGUFs with the MTP
block embedded (unsloth, bartowski, cdiamond) and the engine files both carry the 0.42 B MTP block.

### 2.2 What the best recipes do

**llama.cpp standard mixes** (read from `src/llama-quant.cpp`): `output.weight` is Q6_K for every k-quant and
IQ3_S/IQ4_XS mix, Q5_K for IQ2 and IQ3_XXS mixes; `attn_v` is Q4_K or Q5_K for IQ3_S, IQ3_M, IQ4_XS and Q3_K_M
when GQA is 4 or more (Qwen3.8 has 6); `ffn_down` gets more bits in the first and last eighth of the layers and
every third layer in between; norms, `ssm_conv1d` and other small SSM tensors are never quantized. MTP tensors
(`blk.64.*`) are classified as repeating-layer tensors and follow the trunk's rules; the only architecture-specific
rule found is GLM5-Next's `nextn.eh_proj` floor at Q8_0. The engine's `q3` artifact instead puts the LM head, the
token embedding, `attention/output` and `gdn/output` at 3.25 bpw.

**unsloth Dynamic 2.0/3.0**: layer-selective bit widths, calibration on 300K to 1.5M tokens of hand-curated
chat, long-context and tool-calling text, evaluated with KLD and a "Divergence-300" metric (300 held-out prompts, 32
greedy tokens against BF16) [22][3]. For the Qwen3.5 hybrid family unsloth retired MXFP4, advises never
quantizing `ssm_out` down, and in UD-Q2_K_XL keeps the SSM output at Q5_K and the SSM alpha/beta at Q8_0
[23][24].

**ISTA GSQ-RCO**. GSQ (Gumbel-Softmax Quantization, arXiv:2604.18556, Dadgarnia, Tabesh, Nikdan, Helcig,
Kurtic, Kleinegger, Alistarh) is post-training scalar quantization that keeps a symmetric group-wise integer grid
compatible with INT kernels and GGUF k-quants: GPTQ initialization per layer, then a learnable logit vector per
weight over the small grid, sampled with Gumbel-Softmax at an annealed temperature (2.0 to 0.05) and logit scale
(100 to 500), trained for 10 epochs per layer with Lion against the layer-wise reconstruction error on calibration
activations, then hard-rounded. Group size 128; GPTQ with 512 samples and 10% damping; the Qwen3.5 configs
calibrate on 4096 OpenThoughts samples of up to 4096 tokens (reasoning text), the Llama configs on FineWeb-Edu.
The paper reports 2.13 bpw Llama-3.1-70B zero-shot 75.57 against 78.99 FP16 and vector-quantized 76.3 to 77.3, and
in-format refinement of unsloth Qwen3-8B Q2_K from 50.03 to 56.28 average [25][26]. RCO (Riemannian Constrained
Optimization, arXiv:2605.00649, Helcig and Alistarh) picks one of K options per tensor under an exact total-bit
budget: the budget is a smooth manifold in softmax-logit space, Adam steps are projected onto it, drift is retracted
along the cost vector, and the discrete forward pass uses Gumbel-STE with an exact multiple-choice knapsack
solver; the paper claims parity with EvoPress at 3 to 16x lower wall-clock [27][28]. The GGUF release therefore
is GSQ-refined weights with an RCO-chosen GGUF type per tensor; RentedNoodle's repo reuses ISTA's published
per-tensor allocation file (`REF-IQ3_XXS-mtp.rco-allocation.txt`) [11]. ISTA's card reports benchmarks, not KLD;
a third-party test of the four files on 60 questions found 58/60 for the IQ2_S MTP build and notes that the
"task-lossless" label is the card's, not an independent measurement [29]. The ISTA imatrix file is described in the
engine repository as 1000 chunks of 4096 tokens; its text was not found.

**cdiamond's hybrid**: NVFP4 for the bulk with Q5_K/Q6_K for sensitive tensors, 5.09 bpw, PPL equal to Q4_1 on a
short control where a stock all-NVFP4 build was 6% worse [12]. Together with huginnfork (MLP-only NVFP4, +3% PPL)
and NVIDIA (attention at FP8), every public NVFP4 release of this model keeps something above 4 bits; none
quantizes the GDN projections to FP4 without comment.

### 2.3 Sensitivity findings for the hybrid architecture

- `linear_attn.out_proj` (`ssm_out`, the engine's `gdn/output`, Q3 today) is the single most KLD-sensitive tensor
  in a per-tensor sweep on Qwen3.5-9B; `in_proj_z` tolerates MXFP4 badly; FFN gate/up tolerate about 3 bits [24].
- Low-bit SSM tensors produce "endless reasoning traces" unless protected; recipes that keep them at Q5 to Q8 do
  not show this [23].
- The GDN recurrent state is numerically fragile at depth: llama.cpp issues report instant end-of-sequence beyond
  about 130K tokens on CUDA and CPU (hypothesis: accumulated state error over 48 layers) [30], a 25x decode
  slowdown past about 80K [31], and a BF16 chunked GDN prefill kernel on RDNA3 that alone produced KLD about 0.1
  and top-1 92% on a Q8_0 model [32]. The engine keeps GDN state in FP32; the lesson for evaluation is that a
  kernel-precision regression looks exactly like quantization loss and must be separated by the drift test.
- Vision: every GGUF maker drops the tower into a separate mmproj; huginnfork keeps it BF16. No source reports
  quantization sensitivity of the tower, which the engine does not ship.
- The MTP block does not run in the ordinary forward pass, so no KLD measurement sees its quantization [8]; only
  acceptance does.

### 2.4 MTP in llama.cpp and the drafting-stack precedent

Mainline llama.cpp has MTP speculative decoding since PR #22673 (merged 2026-05-16): `--spec-type draft-mtp`,
`convert_hf_to_gguf.py --mtp` exports the head as a sidecar, `--no-nextn` drops it from the trunk file, and
`llama-imatrix --nextn` collects importance for the MTP block [33][34]. `src/models/qwen35.cpp` implements the
Qwen3.5/3.8 head as `graph_mtp` and asserts a single MTP block. Reported acceptance, with the usual caveat that
llama.cpp and vLLM count acceptance differently:

| Setup | Acceptance | Source |
|---|---|---|
| PR #22673 author, Qwen3.6-27B and 35B-A3B, 3 drafts | about 75% steady state, "more than 2x" speedup | [33] |
| Qwen3.6-35B-A3B UD-Q4_K_XL, RTX 3090, n=2, corrected counters | 78.4%, +22.7% throughput (95% CI +22.1 to +23.3) | [35] repo |
| Qwen3.8-27B NVFP4-MTP GGUF, RTX 5090, n-max 4, 192K ctx | 44.3 to 49.5% across head-precision tiers; Qwen3.6-27B reference 69.7% | [36] |
| Qwen3.8-27B GSQ-RCO IQ3_XXS + Q6_K head, RTX 5070 Ti, n-max 2 | 0.67 at 32K; 0.70 to 0.77 at 65K | [11] |
| Qwen3.8-27B NVFP4 in vLLM | 0.897 Inferact, 0.788 unsloth | [15] |
| Qwen3.5-27B Q4_K_M, early fork | 47.5%, net slower (recurrent-state checkpoint cost) | [37] |

Three public experiments bear on the engine's constraint 8 (drafting precision is a separate knob). On Qwen3.8-27B
the head-precision tiers LOW (Q5_0 LM head, IQ4_XS embeddings and MTP head), MEDIUM (Q8_0/Q6_K/IQ4_XS) and
VERY-HIGH (all BF16) gave 49.0%, 44.3% and 49.5% acceptance, that is no gain from precision [36]. On an L4 with
UD-Q4_K_XL, replacing the embedded Q6_K draft output with a Q4_0 sidecar and then retyping its output tensor to Q2_K
raised mean accepted length from 2.88 to 2.99 to 3.01 (cheaper reads), while requantizing the MTP block itself to
Q3_K lowered first-position acceptance from 0.779 (Q4_0) to 0.746 and was slower despite the smaller file [4].
RentedNoodle found the V-cache precision effect on acceptance flips sign with context state (q4_0 V 0.756 vs q8_0 V
0.742 on a short context; 0.742 vs 0.819 after a long ingest) [11]. The engine's own finding (Q3 MTP accepted fewer
tokens than NVFP4 MTP) matches the Q3_K result; the public evidence says the draft *head* can go very low, the
MTP *block* cannot, and nothing about MTP is visible in KLD.

### 2.5 Where the forms fall

On file size the `q3` artifact (11.72 GiB) is larger than ISTA's IQ3_S (10.99 GiB) and the `nvfp4` artifact
(14.52 GiB) is 1.8 GiB below unsloth's UD-Q4_K_XL. On KL the three base forms and the public files describe the same
checkpoint against the same kind of BF16 reference, so only text, KV cache and prefill activations differ: public
files between 4.4 and 5.3 bpw report KL 0.010 to 0.019 and top-1 95 to 96%; the engine's 4.64 bpw `nvfp4` reports
0.147 and 88%, which public data associates with 2.5 to 2.7 bpw (UD-IQ2_XXS 0.146, 83%). Public files at 3.0 to 4.1
bpw report KL about 0.05 to 0.08 (Q3_K_M 0.047 to 0.056; Qwen3.5-class IQ3_S 0.061) and top-1 about 92% (AD-IQ3_S
4.10 bpw); the engine's 3.74 bpw `q3` reports 0.293 and 76%, close to bartowski's IQ2_XXS (0.301) at 2.5 bpw. The
deployed `nvfp4-compressed`, converted from Swift 1.5 with UkisAI's 1.27 M-token matrix and measured against Swift's
BF16, reports 0.155 and 89.4%, within noise of the base `nvfp4` (SD4 against plain NVFP4 on the base is +0.005
+- 0.009 nats, KL 0.037 between the two), so neither the checkpoint nor the choice between the two public matrices
moves the number. Three facts bound what is footing and what is real: the drift floor on the same artifacts is KL
0.003 to 0.005, so kernels are not it; a same-GPU measurement of NVFP4 KV on this model adds KL about 0.015 [11],
so the KV cache is about a tenth of the `nvfp4` gap; and the text and sample questions are quantified in section 4.
The remainder is the rounding method and the allocation.

## 3. Swift 1.5

### 3.1 Stated by the publisher or seen in primary artifacts

- **Lineage.** `ukisai/Swift-Qwen3.8-27b` (Swift 1.0) is served in its own evaluation logs as a vLLM LoRA adapter
  (`/adapter`, parent `qwen38bf`) over the Qwen3.8-27B BF16 base; the README of the evals repository defines the
  `swift` arm as "the same base with the Swift LoRA" and the Hugging Face page carries the `lora` tag [38][39].
  Swift 1.5 "was made from Swift 1.0", scaling up "the post-training methods that previously improved Swift 1.0",
  with focus on long-horizon, agentic and coding tasks (card, via summary) [40]. The Show HN post for 1.0 names the
  methods tried as RL (GSPO) and on-policy distillation, penalizing tokens associated with overthinking, plus a
  transfer component derived from BottleCap AI's ThinkingCap-Qwen3.6-27B [41]. No training dataset is named
  anywhere found.
- **Claims on the 1.5 card** (via summary): 58.5% fewer thinking tokens, +0.35% average score, 1.95x speed-up on
  several tasks; the GGUF and NVFP4 cards and ukisai.com say 9.18x, an unresolved discrepancy [40][42][43].
  Benchmarks quoted for 1.5 vs base: LiveCodeBench v6 81.71 vs 76.76 (Swift 1.0: 81.55), GPQA-Diamond 88.59 vs
  88.28, IFBench 72.07 vs 73.53; it trails the base on IFBench, ERQA, AIME 2026 and HMMT Nov 2025 [40][44].
- **Swift 1.0 primary logs** (cloned, 159,726 files): nine benchmarks, five seeds each; MMLU-Pro 5-shot CoT base
  85.47% (per-seed s.d. 0.12 pp) vs Swift about 84.95%; base mean completion 3,147 tokens, median 712, p90 8,432;
  LiveCodeBench "fine-grained v6" run of 2026-09-04 at xhigh effort on 175 problems x 5 seeds: base 546/875 =
  62.40%, Swift 611/875 = 69.83%, mean completion -17.2%, paired thinking reduction 24.1%. The card's 76.76 vs 81.55
  comes from a different grading set in the same repository (`official_grades_v6`) [38].
- **License.** UkisAI's contribution is under a "Swift Open License v1.0" with a NOTICE file; the base remains
  Apache 2.0; Featherless and LLM Explorer list the license as "other" [40][45].
- **Distribution.** ukisai publishes BF16 weights, GGUF, INT4 for vLLM, NVFP4, Quark FP8 for AMD, MLX, and a
  `Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF`; bartowski publishes an imatrix GGUF set [46][47]. Featherless serves it at 32K
  with tool calling [45].
- **UkisAI's GSQ-RCO GGUF and importance matrix.** The repository `ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF`
  exists (seen in a search result) and ships `imatrix-swift15-v1mix.gguf`, measured over 309 chunks of 4096 tokens,
  about 1.27 M tokens of web, reasoning, multilingual and code text (per the amended Euhedral docs, which adopted
  the file for the deployed `nvfp4-compressed` conversion) [46][74]. This is the most directly comparable public
  quantization of the served checkpoint. Its bit widths, per-tensor allocation, reported quality, and whether GSQ
  refinement was rerun on Swift or ISTA's base allocation was reused could not be retrieved in this session (the
  search budget was exhausted before the coordinator's note arrived); each is a one-query follow-up.
- **Timing.** The 1.0 evaluation logs are dated 2026-09-04 and the launch coverage 2026-09-16; 1.5 pages were 2 to 4
  weeks old on 2026-10-10, so a late-September 2026 release (approximate).

### 3.2 Unverified or absent

- Whether the LoRA touched the MTP block (`blk.64`) or only trunk projections, and whether 1.5 ships merged
  weights or an adapter. One review says the "MTP head kept for self-speculative decoding" [48]; the launch post
  says 1.0 "retains text, image, and video support", implying the vision tower is present and presumably copied
  [41]. If the trunk changed and the head did not, the head drafts for a model whose hidden states moved; the
  engine's acceptance on Swift versus the base is the only way to know (section 6).
- The 58.5% figure is a median; one critic computes a 41.0% mean reduction at xhigh effort [49], and a side-by-side
  coding test found Swift emitting more tokens than the base on one task (60,572 vs 11,744) [50].
- Recommended sampling settings: not found in any summary.
- Training data, LoRA rank, and whether 1.5 was trained with the MTP loss: not found.
- Bit widths and quality of `ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF`: not retrieved (see 3.1).

For quantization the facts that matter are: the served checkpoint differs from the base by a low-rank update on
some projections, so public base-model KLD tables transfer approximately, and the engine's own Swift form lands
where its base forms do; the deployed form already uses UkisAI's Swift matrix, whose text is known at genre level and
whose size (1.27 M tokens) is a third of ISTA's base matrix (4.1 M tokens); and the model's selling point is a token
budget that quantization is known to inflate (section 4.5).

## 4. Evaluation methodology

### 4.1 KL divergence versus perplexity

ikawrakow defines quantization error as PPL(Q)/PPL(fp16) - 1 and argues that, since the aim is distance to the
same full-precision model rather than task performance, benchmarks are both less sensitive and slower; he treats KLD
and PPL as "closely related" and PPL as more convenient, and elsewhere as equivalent for an infinite corpus with PPL
superior on a limited one [51][52]. His 2025 PR on KLD precision notes that the 16-bit stored base log-probabilities
with a floor 16 nats below the maximum distort the stored-base PPL for large vocabularies but leave KLD and the
other statistics intact; the engine's `compare_reference.py` reads exactly this format [53]. The opposing view,
from "Accuracy is Not All You Need" (arXiv:2407.09141), is that accuracy and PPL average out symmetric per-token
changes while KLD and top-1 flips do not (Spearman 0.981 between KLD and flips on MMLU) [54]; the 2026 smcleod
write-up argues KLD isolates quantization damage from the model's own uncertainty [55]. The llama.cpp README's own
Llama-3-8B scoreboard shows why both are reported: q4_K_M raises PPL by 2.8% while KLD is 0.031 and same-top-p
drops to 91.9%; q2_K raises PPL 56% at KLD 0.445 and same-top-p 71.1% [56].

### 4.2 What `llama-perplexity --kl-divergence` reports

From `tools/perplexity/perplexity.cpp`: for each chunk of `-c` tokens only the second half is scored (`first =
n_ctx/2`, so 1279 positions per 2560-token chunk, which is why the engine uses prefix 1281 and 1279 steps). The
final block prints Mean PPL(Q), PPL(base), the correlation of their logs, mean and ratio of PPL, then Mean KLD with
an uncertainty from the Gaussian approximation over tokens, Maximum, 99.9%, 99.0%, 95.0%, 90.0%, Median, 10%, 5%,
1%, 0.1% and Minimum KLD, then Mean delta-p with percentiles, RMS delta-p, and "Same top p" with a binomial standard
error [56]. The README's reading guide: symmetric delta-p percentiles mean the quantization adds noise; a negative
skew means the model got worse; RMS delta-p is the standard deviation of that noise under a Gaussian model.
The engine's `compare_reference.py` prints mean NLL, paired dNLL with s.e., mean KL and top-1 only; the percentiles,
delta-p and RMS delta-p are missing and are cheap to add.

### 4.3 Typical values that put the engine's numbers in context

| Model (reference, text) | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | IQ4_XS | Q3_K_M | IQ3_S | IQ3_XXS | IQ2_XXS | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B base (BF16, wiki.test) | median 0.0003 | 0.0030 | median 0.0023 | 0.0129 to 0.0139 | | 0.0466 to 0.0564 | | | 0.30 (vs Q8_0) | [1][2] |
| Qwen3.8-27B base, unsloth UD (BF16 or Q8_0, wikitext) | | 0.0029 | | UD-Q4_K_XL 0.0096, top-1 96.0% | | | | | UD-IQ2_XXS 0.146, top-1 83.0% | [2][5][6] |
| Qwen3.6-27B finetune (DavidAU) | | | | 0.0147 | 0.0172 | | IQ3_M 0.0569 | | | [57] |
| Qwen3.5 dense (unsloth page, size unstated) | | | | 0.0192 | 0.0235 | | 0.0613 | | | [23] |
| Qwen3.8-Flash-Next MoE (AesSedai) | | | | 0.034 | 0.076 | | 0.160 | | | [58] |
| Llama-3-8B (f16, wikitext; imatrix rows) | 0.0014 | 0.0055 | 0.0108 | 0.028 | 0.036 | 0.084 | 0.111 | 0.184 | 0.812 | [56] |
| Llama-3-8B same-top-p | 97.7% | 96.0% | | 91.9% | | | | | q2_K 71.1% | [56] |

The Gemma-3-27B QAT study in ik_llama.cpp reports PPL only (q4_0 8.2500, iq4_xs 8.2290, q4_K_M 8.2303 against BF16
8.2021, i.e. 0.3 to 0.6% error at 4.5 to 4.7 bpw on a QAT model) [59]. Across these, a 27B-class dense model at 4.4 to
5.3 bpw sits at KL 0.01 to 0.02 and top-1 95 to 96%; at 3.5 to 4.1 bpw at KL 0.05 to 0.08 and top-1 about 92%;
KL 0.15 to 0.30 and top-1 76 to 83% is 2.5 to 2.7 bpw territory (unsloth's own UD-IQ2_XXS and bartowski's IQ2_XXS).
The engine's 0.147 at 4.64 bpw and 0.293 at 3.74 bpw are therefore unusual by about one full bit of effective
precision relative to the public frontier, after allowing roughly 0.015 for the KV cache.

### 4.4 Texts and sample size

Can the single 1279-token document explain part of it? Three effects, in order of size. First, genre: the document
is the engine's own architecture documentation (Markdown prose with Java and CUDA identifiers), text the model never
saw; public numbers are wikitext. KLD does depend on text (unsloth reports its chat-calibrated quants score worse on
wikitext at 512 context than on chat text [23]; ikawrakow found imatrix sets that overlap wikitext lower wiki.test PPL
by 1 to 2% [52]); a factor of 2 between genres is plausible, 5 to 10x is not (estimate). Second, sampling error:
from the repository's measured s.e. of 0.008 (NVFP4) and 0.025 (Q3) on dNLL over 1279 tokens, the per-token
standard deviation of the NLL difference is 0.29 and 0.89 nats; the KL per token is heavy-tailed, and if its standard
deviation is 2 to 3x its mean the standard error of the engine's mean KL is about 0.008 to 0.012 (NVFP4) and 0.016
to 0.025 (Q3) (heuristic, not measured). Same-top-1 on 1279 tokens has a binomial s.e. of 0.9 to 1.2 pp. Third, the
reference itself: the engine compares against BF16 llama.cpp on CPU, bartowski against BF16 llama.cpp on GPU, both
fine; but the engine's number also bundles the NVFP4 KV cache and two-term FP4 prefill activations (NVFP4 forms) in a
way no GGUF number does. Token counts for a target resolution, extrapolated from the measured s.e. (1/sqrt(N)):

| Target s.e. on dNLL | NVFP4-class (sd 0.29) | Q3-class (sd 0.89) |
|---|---|---|
| 0.01 nats | about 0.8K tokens | about 8K tokens |
| 0.005 nats | about 3.3K | about 32K |
| 0.002 nats | about 20K | about 200K |

Public practice: bartowski scores 100 chunks of wiki.test.raw; the llama.cpp scoreboard uses the full wikitext-2 test
(about 300K tokens, s.e. on KLD 2e-4 to 3e-3 in its tables); ubergarm uses a 1.6 MiB transcript corpus and reports
median and 99% KLD [60]; Qwen discussion #65 uses 87 chunks at 4096 context [8]. Differences of 0.01 in mean KL
between two engine forms are not resolvable on 1279 tokens; 32K scored tokens (25 chunks of 2560) brings the s.e. to
about 0.002 to 0.005 at Q3-class noise.

### 4.5 Downstream deltas versus KLD

Evidence from 2024 to 2026 converges on a threshold picture. "Statistically-Lossless Quantization" (arXiv:2605.02404)
reports baseline accuracy maintained on MMLU and GSM8K whenever KL stays below 0.01 and degradation beyond [61].
"Displacement Is Not Direction" (arXiv:2606.19558) finds KLD and PPL correlate with benchmark score only across
badly degraded models and are essentially uncorrelated near baseline, so they filter damaged candidates but do not
rank near-lossless ones [62]. Flips track KLD tightly (Spearman 0.981 on MMLU) [54]. ISTA's card is consistent:
IQ3_S equal to BF16 on AIME25 and LiveCodeBench at 3.5 bpw, while at 2.5 bpw AIME25 drops 3.3 points and LCB 9.1 [9].
Kaitchup's Qwen3.8-27B study finds every standard GGUF from Q4 down clears a 95% accuracy-recovery bar and that the
differences show up in token efficiency instead [63].

For a reasoning-efficiency fine-tune the second axis is the important one. "Quantization Inflates Reasoning"
(arXiv:2606.25519) and "Quantized Reasoning Models Think They Need to Think Longer" (arXiv:2606.00206) both report
longer reasoning traces under aggressive quantization, and both found the calibration set changes the inflation
[64][65]. Benchmark variance is large relative to the deltas of interest (binomial s.e. for one seed): AIME25 with
30 problems 3.1 pp, GPQA-Diamond 198 problems 2.2 pp, LiveCodeBench v6 1,055 problems 1.2 pp, GSM8K 1,319 problems
0.6 pp, MMLU-Pro 12,032 problems 0.3 pp (computed). UkisAI's own logs run five seeds everywhere and still show
per-seed spreads of 0.1 to 0.2 pp on MMLU-Pro [38]. KLD remains the sensitive instrument; benchmarks are the
acceptance gate.

### 4.6 Long context under quantization

The EMNLP 2025 study "Does quantization affect models' performance on long-context tasks?" (arXiv:2505.20276) finds
4-bit weight quantization loses up to 23% on average at 128K and up to 16% on RULER/OneRuler regardless of method,
with AWQ 1.8%, GPTQ-int4 2.7% and BNB-nf4 6.9% average drops and 8-bit the safe choice [66]. Nothing of the kind
exists for Qwen3.8-27B beyond RentedNoodle's needle set (6/6 at about 20K tokens with IQ3_XXS and q8_0/q4_0 KV;
NVFP4 KV "validated to 96K") [11] and hanxiao's accepted-length-at-depth curve to 152K [4]. The GDN state issues
of section 2.3 mean long-context failures on this architecture can come from the recurrent path rather than from
weights or KV. Teacher-forced KL at depth (the last 1279 tokens after a 64K or 128K prefix) is the cheap test that
exercises weights, the NVFP4 KV at depth, and 48 layers of GDN state together; a needle set separates retrieval.

## 5. Calibration data

| Maker | Set | Size and context | Notes |
|---|---|---|---|
| bartowski | calibration_datav3 (Dampf on Kalomaze); later v5 and a v6 described in a blog | 2,481 lines, 279,515 bytes (third-party description: scientific papers, code, dialogue, multilingual, Wikipedia, legal, fiction) | the gist host is blocked from here; contents known only by description [67][68] |
| unsloth Dynamic 2.0/3.0 | hand-curated chat, long-context and tool-calling text | 300K to 1.5M tokens | argues wikitext calibration overfits wikitext scores; KLD evaluated on wikitext-class sets (calibration_v3/v5 plus wikitext) for comparability [22][23] |
| ubergarm | calibration_data_v5_rc at ctx 512 | | "fancy calibration corpus nor extra context length ... not clearly proven" [60] |
| ikawrakow | wiki.train.raw, many batches | | unsloth's and bartowski's imatrix lowered wiki.test PPL more than his (8.88 and 8.97 vs 9.07 on one recipe), which he reads as those sets overlapping the test text [52] |
| llama.cpp scoreboard | wikitext 1K to 10M tokens | | "no consistent improvement from using more Wikitext tokens for the importance matrix" [56] |
| ISTA GSQ | OpenThoughts (reasoning traces) for Qwen3.5-class and Kimi configs, FineWeb-Edu for Llama | 4096 samples x up to 4096 tokens; GPTQ 512 samples | repo configs [26] |
| ISTA GSQ-RCO-GGUF imatrix (`imatrix-qwen3.8-27b.gguf`, base model) | unknown text | 1000 chunks x 4096 tokens, about 4.1 M tokens (as recorded in the engine repository) | used for the engine's `q3` and `nvfp4`; text not found [9] |
| UkisAI `imatrix-swift15-v1mix.gguf` (Swift 1.5) | web, reasoning, multilingual and code text | 309 chunks x 4096 tokens, about 1.27 M tokens | used for the engine's deployed `nvfp4-compressed`; published in the Swift 1.5 GSQ-RCO GGUF repository [46][74] |
| Euhedral | the two public files above | | the engine records which matrix each artifact used; the exact texts behind either file are not published at document level |

Research on reasoning models points one way. "Quantization Hurts Reasoning?" (arXiv:2504.04823) used reasoning data
self-generated by the model as calibration and found its activation distribution far from pre-training text [69].
"Think Before You Prune" (arXiv:2511.18864) finds pre-training text poorly suited and self-generated reasoning data
better, with a selection strategy [70]. The two 2026 inflation papers adopt the target model's own math traces as
the default calibration set after finding they beat WikiText on both accuracy and length [64][65]. AYOT
(arXiv:2608.01078) feeds the model's own traces and answers as context during ternarization and keeps improving with
more calibration tokens [71]; LAQuant (arXiv:2605.08755) selects reasoning-domain calibration by Hessian-subspace
alignment [72]. GSQ's choice of OpenThoughts for Qwen3.5 is the same idea applied by a GGUF maker.

For the deployed checkpoint the recommendation is therefore: keep UkisAI's `imatrix-swift15-v1mix.gguf` as the
documented default (it is public, Swift-specific, and already contains reasoning text), and test one controlled
variant against it: calibration text generated by BF16 Swift 1.5 itself (its chat template, thinking enabled,
`--parse-special` so the template tokens are counted), on prompts spanning math, code, agentic tool use and long
documents, several hundred prompts of 1K to 4K tokens, disjoint from every evaluation prompt, with a minority of
plain prose and code so wikitext-class text is not out of distribution; `llama-imatrix --in-file` can also merge it
with UkisAI's file. Judge the variant on T4 and T2 of section 6, not on T1 alone. Convert the `q3` and `nvfp4`
forms from Swift 1.5 as well before comparing them with the deployed form, since a base conversion scored against
the base BF16 says nothing about what ships. Record the matrix file's hash, chunk count, `--ctx-size`, the text
hash and token count when the text is the engine's own, the llama.cpp commit and the checkpoint hash in the
artifact metadata.

## 6. Recommended protocol for the engine

Reuse the harness of `00-baseline.md` section 9 and change only the inputs and the report.

**Texts (all tokenized with the Qwen tokenizer; all held out from calibration)**

| Text | Purpose | Size |
|---|---|---|
| T1 `chat-corpus-v1-document.md`, prefix 1281, 1279 steps | continuity with every number published so far | 1279 scored tokens |
| T2 wikitext-2 `wiki.test.raw` | the only text with public Qwen3.8-27B KLD at every bit width (bartowski, 100 chunks) | 25 chunks of 2560 = 32K scored tokens minimum; 100 chunks preferred |
| T3 code: permissively licensed source in 4 languages, not from the engine repository | code behaviour; the engine's own corpus is code-adjacent | 32K scored tokens |
| T4 Swift 1.5 generations: BF16 Swift 1.5 answers with full thinking traces on 100 to 300 prompts (GSM8K-style, LiveCodeBench-style, tool use), template tokens included | deployment distribution; catches thinking-specific damage | 32K scored tokens; a disjoint set of the same kind is the calibration text |

**Reference and checkpoint.** Every form is converted from and scored against the deployed checkpoint, Swift 1.5
(`convert_hf_to_gguf.py --outtype bf16`, then `llama-perplexity --kl-divergence-base` at `-c 2560` on the CPU, as
`docs/QUALITY.md` now describes); base-model conversions are kept only as the bridge to public base-model numbers
on T2. Each reported number names its checkpoint and its matrix. For long-context depth where CPU BF16 is
impractical, a Q8_0 GGUF may stand in (public KLD of Q8_0 to BF16 on this model is 0.0006 to 0.0014), stated in the
report.

**Metrics per text.** Mean KL with s.e., median, 95%, 99% and maximum KL, mean dNLL with paired s.e., same-top-1
with binomial s.e., mean and RMS delta-p, and the 1% and 99% delta-p percentiles (skew). Extend
`compare_reference.py` to print these; they are all available from the stored reference and the report file.

**Attribution runs, once per artifact.** (a) Weight-only KL: dequantize the artifact's tensors back to BF16, write
a GGUF, run `llama-perplexity --kl-divergence` on T2 with an F16 KV cache; this is the number directly comparable to
bartowski's column and separates weight error from KV and prefill-activation error. (b) Drift floor on the same
text (existing test). (c) The engine's full number. The difference (c) minus (a) minus (b) is the cost of the KV
cache and FP4 prefill, which no other chapter can otherwise see.

**Decision thresholds (external yardsticks, not targets set by the engine).** KL below 0.01 is the lossless class
in the literature and Q5_K_M-to-Q6_K territory on this model; 0.010 to 0.020 is Q4_K_M/UD-Q4_K_XL class with
top-1 95 to 96%; 0.05 to 0.08 is Q3_K_M/IQ3_S class with top-1 about 92%; above 0.15 is 2-bit class. A change
between two forms counts only if it exceeds two standard errors on T2 and T4 and has the same sign on all four
texts.

**Long context.** Teacher-forced KL on the last 1279 tokens after prefixes of 32K, 64K and 128K built from T2 and T4
material (one chunk each, repeated with three prefixes), compared with the reference at the same depth; a needle set
of at least 9 depths at 32K, 64K and 128K with greedy decoding; both before and after any KV or GDN-state change.
Any depth where the engine's KL rises faster than the reference's own NLL is a GDN or KV defect, not quantization.

**Downstream.** GSM8K (1,319, one seed, s.e. 0.6 pp) and a fixed 2,000-question MMLU-Pro subset (s.e. 0.8 pp) as
accuracy gates; LiveCodeBench v6 (1,055) when code changes are claimed; AIME25 only with five seeds. On every run
record completion tokens and thinking tokens (mean, median, p90) against the BF16 Swift 1.5 arm, because token
inflation is the failure mode specific to this checkpoint; a form that keeps accuracy but adds 20% thinking tokens
has lost most of what Swift 1.5 bought.

**Drafting stack.** MTP accepted tokens per verification on the fixed corpus, run twice: with the Swift 1.5 MTP
block and with the base model's MTP block grafted in. Equal acceptance means the block was not retrained and the
base head is a valid fallback; higher acceptance with the Swift block answers the open question in section 3.2.

**Record with every imatrix and artifact.** Checkpoint hash, llama.cpp commit, the matrix file's name and SHA-256
(ISTA's base file or UkisAI's Swift file today), calibration text identity and SHA-256 with token count when the
text is the engine's own, `--ctx-size`, chunk count, `--parse-special`, whether `--nextn` and `--process-output`
were used, and the output of `llama-imatrix --show-statistics` for the file actually used.

## 7. Risks and open questions

- The public KLD numbers are for the base model on wikitext; the engine's base forms match that checkpoint but its
  text differs in genre and its number bundles the KV cache. The weight-only run of section 6 on T2 is the only way
  to make the comparison exact; until it exists the "one bit of effective precision" gap is an estimate with a
  plausible 2x uncertainty from text.
- Only the deployed `nvfp4-compressed` describes what ships; `q3` and `nvfp4` are base conversions scored against the
  base BF16. Until Swift conversions of those forms exist, nothing in `00-baseline.md` says how the Q3 route behaves
  on the served checkpoint.
- Every Hugging Face card in this chapter was read through search summaries; row values could be transcription
  errors in those summaries. bartowski's IQ3_S and IQ4_XS rows for this model were not visible at all.
- The ISTA "task-lossless" claim and the GSQ-RCO benchmark table are publisher numbers; the one independent test
  used 60 questions.
- Whether Swift 1.5's MTP block was trained, whether the vision tower was touched, its license terms, and its
  recommended sampling are all undocumented in what was found; the license should be read before any
  redistribution of artifacts.
- Both importance matrices in use are public files whose underlying texts are known only at genre level (UkisAI) or
  not at all (ISTA). A calibration study can hold them fixed by hash, but cannot reproduce them; UkisAI's file is a
  third the size of ISTA's (1.27 M against 4.1 M tokens), which is itself a confound when the two are compared.
- UkisAI's own GSQ-RCO GGUF of Swift 1.5, the one public quantization of the deployed checkpoint with the same
  matrix, has unknown bit widths and quality here; its card is the first thing to read next.
- MTP acceptance on Qwen3.8-27B in llama.cpp is reported at 44 to 50% at n-max 4 but 67 to 77% at n-max 2 and 79 to
  90% in vLLM; the metrics differ and none is the engine's "tokens per verification".
- The GDN recurrent path has open correctness issues in llama.cpp beyond 80K to 130K tokens; a reference computed
  there may itself be wrong, and the CPU BF16 reference at 128K is slow. Validate the reference's own NLL curve
  against depth before trusting any long-context KL.
- The web search budget for this session ran out before the UkisAI GSQ-RCO card, bartowski's full percentile
  table, the unsloth Dynamic 3.0 methodology page, the ISTA imatrix text, the Swift Open License text and the Swift
  1.5 sampling guidance could be searched further; each is a one-query follow-up.

## 8. References

1. bartowski/Qwen3.8-27B-GGUF model card. https://huggingface.co/bartowski/Qwen3.8-27B-GGUF (via search summary)
2. "Independent KLD benchmark: UD-IQ2_XXS beats bartowski IQ2_XXS", unsloth/Qwen3.8-27B-GGUF discussion 49. https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/49
3. "Introducing Unsloth Dynamic v3 Qwen3.8", discussion 74. https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/74 ; Dynamic 3.0 docs https://unsloth.ai/docs/basics/dynamic-3.0-ggufs
4. hanxiao, Qwen3.8-27B UD-Q4_K_XL on an NVIDIA L4 (README, cloned). https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4
5. kingy.ai, "Best Qwen3.8-27B GGUF". https://kingy.ai/blog/qwen3-8-27b-best-quantization-gguf/
6. akopytko/Qwen3.8-27B-NVFP4-GGUF. https://huggingface.co/akopytko/Qwen3.8-27B-NVFP4-GGUF (via summary)
7. AtomicChat/Qwen3.8-27B-GGUF. https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF (via summary)
8. Qwen/Qwen3.8-27B discussion 65, "we quantized Qwen3.8-27B and compared it with community GGUFs on 4x RTX 5090". https://huggingface.co/Qwen/Qwen3.8-27B/discussions/65
9. ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF. https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF (via summary)
10. ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ. https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ
11. RentedNoodle/den_llama.cpp (README, cloned). https://github.com/RentedNoodle/den_llama.cpp
12. cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF https://huggingface.co/cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF ; forum post https://forums.developer.nvidia.com/t/qwen3-8-27b-at-256k-on-a-24-gb-blackwell-target-gpu-imatrix-nvfp4-mtp-55-4-tok-s/380456
13. nvidia/Qwen3.8-27B-NVFP4. https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4 (via summary)
14. unsloth/Qwen3.8-27B-NVFP4. https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4 (via summary)
15. vLLM recipes, Qwen3.8-27B. https://recipes.vllm.ai/Qwen/Qwen3.8-27B
16. MiaAI-Lab/Qwen3.8-27B-NVFP4-RTX-5090 (README, cloned). https://github.com/MiaAI-Lab/Qwen3.8-27B-NVFP4-RTX-5090 ; Inferact https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4 ; RadixArk https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4
17. huginnfork/Qwen3.8-27B-NVFP4A16. https://huggingface.co/huginnfork/Qwen3.8-27B-NVFP4A16 (via summary)
18. TelperionAI/Qwen3.8-27B-NVFP4-AWQ-GPTQ. https://huggingface.co/TelperionAI/Qwen3.8-27B-NVFP4-AWQ-GPTQ
19. Qwen/Qwen3.8-27B-FP8. https://huggingface.co/Qwen/Qwen3.8-27B-FP8 ; absence of official GGUF/AWQ/GPTQ/MLX per https://kingy.ai/blog/qwen3-8-27b-specs-benchmarks-local-hardware/
20. mlx-community/Qwen3.8-27B-4bit https://huggingface.co/mlx-community/Qwen3.8-27B-4bit ; mlx-community/qwen3.8-27b-mtp-4bit ; lmstudio-community/Qwen3.8-27B-MLX-4bit ; PocketAiHub/Qwen3.8-27B-MLX
21. NVIDIA forum, "Qwen3.8-27B-NVFP4 @ TP=4" (doth4580 model). https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-tp-4-running-at-80-tok-s-sustained-peak-at-92-tok-s/380904
22. Unsloth Dynamic 2.0 GGUFs. https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs
23. Unsloth, Qwen3.5 GGUF Benchmarks. https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks ; Kaitchup, "Qwen3.5 9B MoQ" https://kaitchup.substack.com/p/qwen35-9b-moq-inside-a-strong-36
24. kaushall13, Qwen3.5-9B quantization plan. https://github.com/kaushall13/qwen3.5-9b-quantization/blob/main/docs/qwen35_9b_q4_quantization_plan.md
25. GSQ paper, arXiv:2604.18556. https://arxiv.org/abs/2604.18556
26. IST-DASLab/GSQ repository (README and configs, cloned). https://github.com/IST-DASLab/GSQ
27. RCO paper, arXiv:2605.00649. https://arxiv.org/abs/2605.00649
28. IST-DASLab/RCO repository (README, cloned). https://github.com/IST-DASLab/RCO
29. zephel01, test of Qwen3.8-27B-GSQ-RCO with and without MTP. https://note.com/zephel01/n/nb849cea8359d
30. llama.cpp issue 27756 (instant EOS beyond about 130K). https://github.com/ggml-org/llama.cpp/issues/27756
31. llama.cpp issue 27623 (decode collapse beyond about 80K). https://github.com/ggml-org/llama.cpp/issues/27623
32. stew675/llama-cpp-rdna-boosts issue 113 (BF16 chunked GDN prefill KLD about 0.1). https://github.com/stew675/llama-cpp-rdna-boosts/issues/113
33. llama.cpp PR 22673, "llama + spec: MTP Support". https://github.com/ggml-org/llama.cpp/pull/22673
34. llama.cpp at commit 23b0202a (cloned): docs/speculative.md, tools/imatrix/README.md, convert_hf_to_gguf.py, src/models/qwen35.cpp, src/llama-quant.cpp. https://github.com/ggml-org/llama.cpp
35. thc1006/qwen3.6-speculative-decoding-rtx3090 (README, cloned). https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090
36. PierpaoloPernici, Qwen3.8-27B NVFP4-MTP benchmark on RTX 5090 (gist, via summary). https://gist.github.com/PierpaoloPernici/2e6f6f42965d531b364f39ab0e8a52ad
37. NodeNestor/qwen3.5-27b-mtp-llamacpp (README, cloned). https://github.com/NodeNestor/qwen3.5-27b-mtp-llamacpp
38. UkisAI/Swift-Qwen3.8-27B-evals (cloned; README, 02-mmlu-pro/aggregate.json, 09-livecodebench-v6/logs-run-20260904). https://github.com/UkisAI/Swift-Qwen3.8-27B-evals
39. ukisai/Swift-Qwen3.8-27b. https://huggingface.co/ukisai/Swift-Qwen3.8-27b (via summary)
40. ukisai/Swift-1.5-Qwen3.8-27b. https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b (via summary)
41. Show HN: Swift-Qwen3.8-27B. https://news.ycombinator.com/item?id=49727511
42. ukisai/Swift-1.5-Qwen3.8-27B-GGUF https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF ; ukisai/Swift-1.5-Qwen3.8-27b-NVFP4 https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-NVFP4
43. UkisAI, Swift 1.5 Qwen3.8-27B. https://ukisai.com/swift-1-5-27b ; introducing Swift https://ukisai.com/news/introducing-swift
44. HackerNoon, "Swift-Qwen3.8-27B Cuts Reasoning Tokens Without Sacrificing Much Accuracy". https://hackernoon.com/swift-qwen38-27b-cuts-reasoning-tokens-without-sacrificing-much-accuracy
45. Featherless listing. https://featherless.ai/models/ukisai/Swift-1.5-Qwen3.8-27b ; LLM Explorer https://llm-explorer.com/model/ukisai%2FSwift-1.5-Qwen3.8-27b,3jHOVbzg1EPZp5GUvwUGrh
46. ukisai Swift 1.5 27B collection. https://huggingface.co/collections/ukisai/swift-15-27b ; ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/tree/main ; INT4 https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-INT4 ; Quark FP8 https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-Quark-FP8-dynamic-AMD
47. bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF. https://huggingface.co/bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF
48. MindStudio, "Qwen 3.8 27B Swift vs Original". https://www.mindstudio.ai/blog/qwen3-8-27b-swift-vs-original-thinking
49. traictory, "UkisAI's Swift-Qwen3.8-27B sells a median, not a mean". https://traictory.com/news/2026-09-16-ukisai-swift-qwen38-27b
50. HackerNoon, "The Qwen3.8-27B Variant Built to Stop Overthinking". https://hackernoon.com/the-qwen38-27b-variant-built-to-stop-overthinking
51. ikawrakow, ik_llama.cpp discussion 8, "New quantization types IQ2_K, IQ3_K, IQ4_K, IQ5_K". https://github.com/ikawrakow/ik_llama.cpp/discussions/8
52. ikawrakow, ik_llama.cpp discussion 359, "Qwen3 quantization experiments". https://github.com/ikawrakow/ik_llama.cpp/discussions/359
53. ikawrakow, ik_llama.cpp PR 325, "Fix KLD precision". https://github.com/ikawrakow/ik_llama.cpp/pull/325
54. "Accuracy is Not All You Need", arXiv:2407.09141. https://arxiv.org/abs/2407.09141
55. smcleod, "Measuring Model Quantisation Quality with KL Divergence" (2026-04). https://smcleod.net/2026/04/measuring-model-quantisation-quality-with-kl-divergence/
56. llama.cpp tools/perplexity/README.md and perplexity.cpp (cloned). https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md
57. DavidAU Qwen3.6-27B imatrix GGUF card. https://huggingface.co/DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF (via summary)
58. AesSedai/Qwen3.8-Flash-Next-GGUF. https://huggingface.co/AesSedai/Qwen3.8-Flash-Next-GGUF (via summary)
59. ubergarm, ik_llama.cpp discussion 334, iq4_ks on gemma-3-27b-it-qat. https://github.com/ikawrakow/ik_llama.cpp/discussions/334
60. ubergarm, ik_llama.cpp discussion 434, "Quant Cookers Basic Guide". https://github.com/ikawrakow/ik_llama.cpp/discussions/434
61. "Statistically-Lossless Quantization of Large Language Models", arXiv:2605.02404. https://arxiv.org/abs/2605.02404
62. "Displacement Is Not Direction: Evaluating Fidelity Metrics for Quantized LLM Deployment", arXiv:2606.19558. https://arxiv.org/abs/2606.19558
63. Kaitchup, "Qwen3.8 27B GGUF Benchmark: Q4 to Q1 Accuracy and Token Efficiency". https://kaitchup.substack.com/p/qwen38-27b-gguf-benchmark-q4-to-q1 (seen via its Flash-Next companion post)
64. "Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models", arXiv:2606.25519. https://arxiv.org/abs/2606.25519
65. "Quantized Reasoning Models Think They Need to Think Longer, but They Do Not", arXiv:2606.00206. https://arxiv.org/abs/2606.00206
66. "Does quantization affect models' performance on long-context tasks?", arXiv:2505.20276, EMNLP 2025. https://arxiv.org/abs/2505.20276
67. bartowski, calibration_datav3 gist. https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8 ; v5 https://gist.github.com/bartowski1182/82ae9b520227f57d79ba04add13d0d0d ; "New imatrix dataset" https://huggingface.co/blog/bartowski/imatrix-dataset
68. lemon07r/bartowski-imatrix-v3-semantic (dataset card describing v3). https://huggingface.co/datasets/lemon07r/bartowski-imatrix-v3-semantic
69. "Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models", arXiv:2504.04823. https://arxiv.org/abs/2504.04823
70. "Think Before You Prune: Selective Self-Generated Calibration for Pruning", arXiv:2511.18864. https://arxiv.org/abs/2511.18864
71. "Attend to Your Own Thoughts", arXiv:2608.01078. https://arxiv.org/abs/2608.01078
72. "LAQuant", arXiv:2605.08755. https://arxiv.org/abs/2605.08755
73. "A KL Lens on Quantization: Fast, Forward-Only Sensitivity for Mixed-Precision SSM-Transformer Models", arXiv:2604.13440 (KL-guided per-layer sensitivity for hybrid models; abstract only). https://arxiv.org/abs/2604.13440
74. Euhedral repository, `docs/QUALITY.md` and PR 119 at commit 269a890 (source checkpoints, importance matrices `imatrix-qwen3.8-27b.gguf` and `imatrix-swift15-v1mix.gguf` with their chunk counts, and the measured numbers), as summarized in `00-baseline.md` of this folder.
