# Lossless and entropy coding of the quantized weights

What this chapter covers: storing a given set of quantized values in fewer bytes (codes, scales, high bits), the decode cost of
doing so inside the engine's kernels, and the rate-distortion comparison that decides whether to spend bits on lossless coding of
a coarse quantizer or on a finer quantizer. Lossy formats are other chapters' business. Everything measured comes from the
repository (`docs/COMPRESSED_Q3.md`, `docs/NVFP4_LOSSLESS.md`, `docs/NVFP4_COMPRESSED.md`, `native/src/q3/p2e2.cuh`,
`tools/euhedral_artifacts/grouped.py`, `tools/nvfp4z/`) or from the repositories listed in section 7. Everything else is an
estimate from numpy simulations written for this chapter (`sim_search.py`, `sim2.py`, `ecsq_rd.py`; the method is described
where the numbers appear) and is labelled "simulation".

## 1. Summary

Lossless coding cannot deliver a compressed *calibrated* Q3 artifact at anything like the 81.5% of the round-to-nearest one. The
problem is not only that P2E2 has no symbol for code -4: the calibrated scale search flattens the code distribution. A Gaussian
within-group model that reproduces the repository's two measured RTN statistics (2.3 bits of entropy, 80% of codes in -1, 0, +1)
predicts that `grouped.py`'s search raises the code entropy to about 2.7 bits and the BIG fraction from 20% to 37%, puts a -4 in
71% of groups (often several per group, 95% of the time beside a -3), and would push even a hypothetical -4-free P2E2 to 92% of the
row-split bytes; the best lossless layout that stores -4 (a two-level payload with a one-bit escape stream for the -3/-4 class)
lands at about 94% for about 1.7x P2E2's decode instructions. The recommendation is therefore not a layout but a measurement
first: a read-only script over the existing calibrated `q3` artifact settles the real histogram in minutes, and the go/no-go is
P(BIG) below 0.25. If the maintainer wants a compressed calibrated artifact regardless, the two-level layout is the one to build,
because it keeps the 2-bit payload units and the warp's popc/scan addressing intact; the alternative that keeps today's layout and
kernels untouched is a search restricted to -3..3 (73% of the calibrated MSE gain at about 89% of row-split bytes in the model)
or an entropy-constrained search that trades MSE for payload bits (32% of the gain at today's 82%). Per plane, the reachable
lossless savings in the `q3` artifact are: codes 1.56 GiB today (RTN) or 0.5 to 1.2 GiB if calibrated; FP16 scales 0.12 to 0.16
GiB with a 12-bit exponent-table or dictionary index; Q4/Q5 codes and the Q5 fifth-bit plane essentially nothing (the fifth bit is
the sign bit, 1.0 bit of entropy). In the NVFP4 artifacts only the scale plane was compressible and SD4 already takes it. On
rate-distortion grounds "entropy-coded calibrated Q3" is a 3.0 bpw point, not a 2.65 bpw point, and sits about 4 dB above the
Shannon bound at its rate; an ideal entropy-coded scalar quantizer sits 1.5 dB above it, so a native 2.65 to 2.75 bpw quantizer of
QTIP or E8 quality should beat RTN-P2E2 at equal bytes by roughly 0.1 nats in this model's NLL. Codecs such as rANS, tANS, Huffman
and Tunstall cannot run inside the weight-streaming GEMV within the instruction budget; they belong to expansion before prefill,
to a host-backed path (where nothing compressible remains in the current artifacts), and to archives.

## 2. Mechanism

### 2.1 What P2E2 does today

The layout (`p2e2.cuh`, `q3_p2e2.py`) stores every Q3 code as a 2-bit primary symbol (0: -1, 1: 0, 2: +1, 3: BIG) in a fixed-rate
plane, and every BIG code's value as a 2-bit unit (-3, -2, +2, +3) in one payload stream in row-major order. Addressing costs one
u32 per row (the unit index of the row's first BIG code); inside a row a warp walks 1024-code slices in order, each lane extracts
the BIG flags of its 32 codes with `word & (word >> 1) & 0x55555555`, counts them with `__popc`, and an inclusive warp scan
(`__shfl_up_sync`, five steps, two rows packed per 32-bit lane value) gives every lane its unit offset and the warp the slice
total, which advances the row base. The lane fetches a 64-bit window of its units with three shuffles out of 32 prefetched payload
words and two funnel shifts. Per pair of codes the fast path then does: popc of the doubled BIG mask below the pair (the doubling
makes popc return a bit offset directly), a funnel shift of the window, a `lop3` that keeps only the units of BIG positions, a
nibble extraction, a `mad` into a 256-entry `float2` table and two FMAs: about 11 instructions against 7 for the contiguous
kernel. The FMA chain is the contiguous kernel's, so the result is bitwise identical. Two conditions fall back to a per-code slow
path (`DotAny`): a lane with more than 16 BIG codes among its 32, or a slice whose units outrun the 32-word window (more than 512
units in 1024 codes). The expansion kernel rebuilds row-split bytes at about 450 GB/s of output (160 us for gate_up, 73 us for
down, from the repository's timings).

### 2.2 The code distribution decides everything

The transcoder measured 81.5% on the RTN artifact. What the calibrated artifact's codes look like is not measured anywhere in the
repository, so I simulated `search_group_codes_torch` exactly as written (39 candidates per group: the RTN scale, then for each
end in {3, 4} and each step in -0.9..0.9 a trial scale `amax/(end + step)`, rounding, a weighted least-squares refit, binary16
rounding and requantization; the group keeps the least weighted error) on i.i.d. groups of 64 from several distributions.
The Gaussian reproduces both measured RTN statistics, so it is the primary model; Student-t with 5 degrees of freedom and
Laplacian give the sensitivity range. Column weights drawn log-normally (sigma 1) changed nothing material.

| Simulation (groups of 64, uniform weights) | Entropy of codes | P(BIG), i.e. abs code >= 2 | P(-4) | Groups with a -4 | Of those, with >= 2 | -3 present when -4 is | MSE / variance |
|---|---|---|---|---|---|---|---|
| Gaussian, RTN | 2.325 bits (repo: 2.3) | 0.205 (repo: about 0.20) | 0 | 0 | | | 0.0632 |
| Gaussian, calibrated (grouped.py) | 2.735 | 0.371 | 0.022 | 71% | 54% (up to 10 per group) | 95% | 0.0359 |
| Gaussian, calibrated restricted to -3..3 | 2.577 | 0.315 | 0 | 0 | | | 0.0432 |
| Student-t(5), RTN | 2.006 | 0.114 | 0 | 0 | | | 0.110 |
| Student-t(5), calibrated | 2.474 | 0.239 | 0.0165 | 66% | 40% | 77% | 0.0625 |
| Laplacian, calibrated | 2.379 | 0.217 | 0.0166 | 66% | 42% | 79% | 0.0585 |

Three consequences. First, a -4 is not rare and not unique: any negative element with magnitude at least 3.5 scales codes to -4,
so the "at most one -4 per group, only at the group's maximum" picture is wrong, and per-group exception schemes are not lossless.
Second, the search picks the finer "end = 4" branch for 76% of Gaussian groups, which is where the calibrated gain comes from (MSE
0.57x; the measured NLL penalties, +0.170 against +0.275 nats, have the ratio 0.62) and also where the BIG fraction comes from.
Third, P2E2's fast path assumes the RTN distribution: with P(BIG) = 0.37 a lane exceeds 16 BIG codes with probability 4.7%, so
99.8% of slices (128 lane-rows each) would take the slow path; at P(BIG) = 0.315 (restricted search) 68% of slices would. The
window arithmetic in `p2e2.cuh` appears to accommodate about 30 units per lane, so the check may be loosenable, but that is a
kernel change to verify, not a given.

### 2.3 Layouts that store -4

Bits per weight below include the FP16 scale plane (0.25 bpw); ratios are against the 3.25 bpw row-split layout. "Delta
instructions" are my reading of the fast path in `p2e2.cuh` against its current 11 per pair; none is measured.

| Layout | How -4 is stored | bpw, Gaussian calibrated (simulation) | Ratio | bpw, Student-t(5) | Delta instructions per pair | FMA order unchanged | Other kernel changes |
|---|---|---|---|---|---|---|---|
| P2E2 as is | cannot | 2.993 if -4 did not exist | 0.921 | 2.729 | 0 | yes | fast-path limit (2.2) |
| A: 3-bit payload units | 5 BIG values in 3 bits (0.68 bit per BIG wasted) | 3.363 | 1.035 | 2.970 | +0 to +1 (a `mad` by 3 replaces the doubled-mask trick; the `lop3` mask goes, a 1024-entry table absorbs garbage bits) | yes | 16 units per half-lane need 48 bits: shifts above 32 need a third window word and a select, about +2 more; a slice of 1024 codes at 37% BIG needs 36 words, more than the 32-word prefetch |
| B: two-level payload | unit 0 means "-3 or -4"; a third stream holds one bit per such unit; a second u32 row base | 3.063 | 0.942 | 2.771 | +6 per pair (zero-class mask, popc below the pair, funnel shift of the escape window, index merge) plus about +1.6 amortized per pair for a second warp scan and window fetch per slice; 1024-entry table | yes | none to the primary or payload addressing |
| S: per-group unit width | the free sign bit of the FP16 scale flags groups holding a -4; those groups use 3-bit units | 3.269 | 1.006 | 2.89 | +1 to +3 (width-dependent shift, wider window as in A, per-lane table base) | yes | sign bit must be cleared before `__half2float`; both unit widths in one code path |
| K: exception plane | 7 bits per group: position of the single -4 | not lossless (54% of -4 groups hold several) | | | | | |
| Sparse per-row position list | u16 position per -4 | +0.35 bpw at P(-4) = 0.022; warp must merge about 22 exceptions per slice | | | | | |
| Entropy bound (order 0) | | 2.985 | 0.918 | 2.724 | | | |

B is the only one that both stays lossless and keeps the addressing that makes P2E2 cheap; its extra cost over P2E2 is about the
cost of P2E2 over the contiguous kernel again, so one-row decode would be roughly 25% below the `q3` kernel instead of 13% (an
extrapolation of the measured 11 against 7 relationship, to be measured). A is simpler but saves nothing on the Gaussian
model and 9% on the heavy-tailed one. S has A's problems plus a mixed-width code path.

Two alternatives change the rounding instead of the layout and so belong partly to other chapters, but they are what the
layouts compete with. Restricting the search to -3..3 (the fallback named in `docs/P2E2_FULL_RANGE.md`) keeps layout, transcoder
and kernels: in the Gaussian model it keeps 73% of the calibrated MSE gain at 2.88 bpw (ratio 0.886), with Student-t 2.63 bpw
(0.81). An entropy-constrained search, one line in `grouped.py` (choose the candidate minimizing weighted error plus lambda times
the group's P2E2 payload bits), moves along a curve: lambda 0.05 keeps 63% of the gain at 2.78 bpw (0.855), lambda 0.1 keeps 32%
at 2.66 bpw, today's bytes. These are MSE figures from the model; the NLL mapping is in section 3.

### 2.4 Other code families and why they do not fit the GEMV

The contiguous kernel's budget is about 1.3x its 7 instructions per pair before one-row decode slows, and the 4-row twins are
already ALU-bound. Estimates per decoded symbol for a static-model decoder, one lane per stream:

| Decoder | Per symbol | Per pair with FMAs | Sequential dependency | Addressing of a lane's 32 codes |
|---|---|---|---|---|
| Table Huffman (DFloat11 style: peek 8 bits, LUT, length LUT, shift) | 4 to 5 | 10 to 12 | yes (bit position) | needs a stored offset per lane and slice (0.25 bpw) or one stream per lane |
| tANS (FSE style: one table read, bit refill, state update) | 5 to 6 | 12 to 14 | yes (state) | same |
| rANS, 16-bit renormalization (nvfp4z kernel) | slot, symbol table, freq and cum reads, mul-add, renorm branch and load: 9 to 10 | 20 to 22 | yes (64-bit state) | same, plus 64-bit state and a backwards-written stream per lane |
| Tunstall (variable-to-fixed, 12-bit words into 3-bit code strings) | 6 per word of about 3 to 4 codes: about 1.8 | 7 + 3.5, about 10.5 | yes (running length) | per-lane word counts cannot be derived from a fixed-rate plane: 0.25 bpw of offsets, which erases its gain over P2E2 |
| Bit-plane coding of the three code bits | order-0 planes do not beat the joint entropy; gains need contexts | | | |
| P2E2 (fixed-rate primary plus 2-bit units) | | 11 | no: ranks come from popc of the fixed-rate plane | derived, 1 u32 per row |

The structural point is that P2E2's fixed-rate primary plane makes every lane's payload offset computable from data the lane
loads anyway. Every variable-rate code above either needs stored per-lane offsets (which cost more than they save at 32 codes
per lane) or serializes the lane. For the RTN distribution P2E2's 2.46 bits per code is within 0.14 bits of the entropy, so even
an ideal coder would save only 5% more of the code plane; for the calibrated distribution the ideal saves 0.27 bits per code over
plain 3 bits and P2E2-B saves 0.19. There is no lossless code worth building inside the GEMV beyond P2E2 and its -4 extension.

### 2.5 Scales and high bits

FP16 scales (0.65 GiB of the `q3` artifact's 8.45 GiB of Q3 tensors) compress 43% under GDeflate (repository), about 9 bits of
information per 16. Lossless options that a kernel can read in place: a per-tensor dictionary index of fixed width b, saving
(16 - b)/16 of the plane (10 bits: 0.24 GiB, only if a tensor uses at most 1024 distinct scales; 12 bits: 0.16 GiB), or an
exponent table (the scales of one tensor use a handful of binary16 exponents; store a 2 or 3 bit table index plus the raw 10-bit
mantissa, 12 to 13 bits, 0.12 to 0.16 GiB, with no limit on distinct values). Cost: the kernels already load one scale per half
group per slice; an index load, a shared-memory lookup and an insert add about 3 instructions per 16 pairs, under 0.2 per pair,
but in every Q3 kernel family (contiguous, rows twins, P2E2 decode and expand, the MX prefill producer, the embedding gather, the
scalar reference). SD4 is the lossy version of the same idea with a 4-bit index (0.78 GiB in the NVFP4 artifact, chapter on
scales). E4M3 scales carry 3.63 of 8 bits (repository); a 6-bit lossless index saves 0.39 GiB and SD4 saves twice that.

Q4 and Q5 (2.99 GiB, the GDN and attention input projections): simulated code entropies are 3.50 of 4 (RTN) and 3.67 (calibrated)
for Q4, 4.58 and 4.64 of 5 for Q5, so the order-0 bound is 0.07 to 0.1 GiB for Q4 and 0.15 to 0.19 GiB for Q5. Escape layouts in
the P2E2 style do not reach it: a 3-bit primary with a 4-bit escape unit costs 3.82 bits per Q4 code under RTN (4.5% saving, 0.035
GiB) and 4.06 under calibration (a loss); a 4-bit primary with a 5-bit escape costs 5.02 to 5.12 per Q5 code (a loss). The Q5
fifth-bit plane is bit 4 of the two's complement code, which is the sign bit: entropy 0.99 to 1.0 bit, incompressible on its own.
The repository's "about 0.1 GiB each" is an upper bound that no fixed-structure code reaches.

NVFP4 codes carry 3.90 of 4 bits (repository), so the code plane of the nvfp4 artifacts is incompressible to any practical code
(0.31 GiB theoretical); nvfp4z's context-modelled rANS reaches 3.86 to 3.87 bits on the Flash-Next weights and is an archive tool.

## 3. Published results and rate-distortion gaps

The web-search budget of this session was exhausted before the literature searches ran, so the table holds only what was read
directly from the named repositories and from the engine; the papers' own throughput tables (DFloat11, ZipNN, NeuZip) were not
fetched and their arXiv identifiers are the ones printed in their READMEs.

| Method | Data | Ratio | Decode throughput | Hardware | Source |
|---|---|---|---|---|---|
| P2E2 (fused GEMV) | Q3 RTN codes of Qwen3.8-27B | 0.815 of row-split | in the GEMV: decode 109 to 95 tok/s at 4K (-13%); expansion 453 GB/s (gate_up), 496 GB/s (down), output bytes | RTX 5070 Ti | `docs/COMPRESSED_Q3.md` |
| nvCOMP ANS | the same Q3 row-split tensors | 0.893 (0.771 on top of P2E2) | 154 GB/s (gate_up, 0.47 ms), 129 GB/s (down); whole pages into a separate buffer | RTX 5070 Ti, nvCOMP 5.3 | `docs/COMPRESSED_Q3.md` |
| GDeflate (nvCOMP) | the same | 0.885 (entropy-only and max-ratio alike) | 35 GB/s (gate_up, 2.04 ms) | RTX 5070 Ti | `docs/COMPRESSED_Q3.md` |
| nvfp4z (static rANS, 16-bit renormalization, one thread per lane, 2^16 lanes) | NVFP4 checkpoint, 104.81 GB | 0.899 (94.25 GB); codes 3.86 to 3.87 of 4 bits, scales 0.22 of 0.5 | 14.2 GB/s decode (66 ms for 0.94 GB incl. table build), 7.3 GB/s encode | RTX 5070 Ti | `docs/NVFP4_LOSSLESS.md`, `tools/nvfp4z/kernels.cu` |
| DFloat11 (Huffman of the BF16 exponent byte; sign and 7 mantissa bits raw) | BF16 LLMs and diffusion models | about 0.70 | GPU decode before each matmul: 8 code bytes per thread, 5-bit start gap, 256-entry LUT chain up to 4 levels (escape values >= 240), two passes (count, block prefix sum in shared memory, write); README: about 2x slower than BF16 at batch 1, overhead independent of batch size | CUDA 12 GPUs | repo `LeanModels/DFloat11`, arXiv 2504.11651 (NeurIPS 2025) |
| ZipNN (byte grouping of exponent and mantissa, zstd-style entropy coding) | BF16 models | 1.51x (66.3%) | CPU: 1660 MB/s decompress, 1120 MB/s compress single-threaded, 256 KB chunks; up to 80 GB/s with 16 workers on two Xeon 8480+ sockets (224 cores); storage and load-time format | CPU | repo `zipnn/zipnn`, arXiv 2411.05239 |
| NeuZip (exponent byte split off, nvCOMP ANS; bitcomp, LZ4, zstd, GDeflate selectable; optional mantissa truncation) | BF16 training and inference weights | not read | nvCOMP per layer, compress on write, decompress on use; chunk 2^16 | NVIDIA GPUs | repo `BorealisAI/neuzip`, arXiv 2410.20650 |
| rANS (Duda) and interleaved rANS (Giesen) | general | at entropy | CPU: 14 to 23 clocks per symbol (i7-2600K) | CPU | repo `rygorous/ryg_rans`, arXiv 1311.2540, arXiv 1402.3392 |
| Huff0 and FSE (tANS) | synthetic, 32 KB blocks | at entropy (FSE), 1 bit per symbol floor (Huff0) | CPU: Huff0 555 to 1350 MB/s, FSE 440 to 460 MB/s, one core (i7-5600U) | CPU | repo `Cyan4973/FiniteStateEntropy` |
| llama.cpp TQ1_0, TQ2_0 | ternary weights | 1.6875 bpw (5 trits per byte, times 243), 2.0625 bpw (2 bits, FP16 scale per 256) | fixed-rate packing, dequantized in the matmul | CPU/GPU | `ggml/src/ggml-common.h`, `ggml-quants.c` |
| QTIP (bitshift trellis, L = 16, V = 2, compute-based codebooks 1MAD, 3INST, HYB) | Llama 2 at 2, 3, 4 bits | fixed rate | GEMV at the compressed bandwidth: 188 tok/s at 2 bits against 55.9 for FP16 (Llama-2-7B, RTX 6000 Ada), same as QuIP# | RTX 6000 Ada | repo `Cornell-RelaxML/qtip`, arXiv 2406.11235 |
| QuIP# (E8 lattice codebooks E8P12, Hadamard incoherence) | Llama at 2 to 4 bits | fixed rate | 176 tok/s at 2 bits (Llama-2-7B) | RTX 6000 Ada | repo `Cornell-RelaxML/quip-sharp`, arXiv 2402.04396 |

Decode rates in the first rows are output (row-split) bytes per second, computed from the repository's timings and the tensor
sizes (gate_up 72.4 MB, down 36.2 MB in row-split). The ordering is the useful part: fused in-GEMV decode at 700 GB/s of weight
stream, expansion at 450 GB/s, nvCOMP ANS at 150 GB/s, GDeflate at 35 GB/s, a one-thread-per-lane rANS at 14 GB/s. DFloat11
decodes whole matrices into a BF16 buffer before the matmul, which is the engine's expansion route, not its fused route; its
"2x slower at batch 1" is what expansion before every GEMV costs when the weights are read once per token.

### Rate-distortion gaps

Computed for this chapter by numerical integration (unit-variance sources, `ecsq_rd.py`) and by simulation (Lloyd-Max iteration;
a 4-state Ungerboeck trellis with the doubled uniform alphabet, Viterbi search, `sim2.py`). The Gaussian rate-distortion function
is D = 2^(-2R); for the Laplacian the Shannon lower bound (e/pi) 2^(-2R) is used, so Laplacian gaps are upper bounds.

| Quantizer, Gaussian source | Rate (bits) | D / variance | Gap to Shannon |
|---|---|---|---|
| Entropy-coded uniform scalar, centroid reconstruction | 2.0 / 2.65 / 3.0 / 4.0 | 0.0894 / 0.0365 / 0.0225 / 0.00556 | 0.255 bits (1.53 dB) at every rate: the classical Gish and Pierce figure, reproduced |
| Same, midpoint reconstruction (what Q3 does) | 2.0 / 2.65 / 3.0 | 0.0982 / 0.0379 / 0.0230 | 0.32 / 0.28 / 0.27 bits (1.9 to 1.6 dB) |
| 8-level uniform, levels -4..3, midpoint (Q3's alphabet) | 2.5 / 2.65 / floor at 2.72 | 0.0478 / 0.0415 / 0.0406 | 0.30 / 0.35 / 0.41 bits; cannot go below 0.0406 |
| Laplacian, entropy-coded uniform, centroid | 2.0 / 2.65 / 3.0 | 0.0740 / 0.0308 / 0.0193 | 0.22 / 0.24 / 0.25 bits |
| Lloyd-Max, 4 and 8 levels (fixed rate) | 2 / 3 | 0.1175 / 0.0346 | 2.74 dB / 3.44 dB |
| 4-state TCQ, fixed rate (this simulation) | 2 / 3 | 0.0950 / 0.0299 | 1.82 dB / 2.82 dB (0.9 and 0.6 dB over Lloyd-Max) |
| Shannon bound | 2 / 2.65 / 3 | 0.0625 / 0.0254 / 0.0156 | 0 |

The engine's operating points in the same model (bpw includes the 0.25 scale overhead, MSE from the search simulation):

| Operating point | bpw | MSE / variance | Gap to Shannon at that bpw | NLL penalty, measured or estimated |
|---|---|---|---|---|
| RTN Q3, row-split | 3.25 | 0.0632 | 7.6 dB | +0.275 nats (measured) |
| RTN Q3, P2E2 | 2.66 | 0.0632 | 4.0 dB | +0.275 (the same values) |
| Calibrated Q3, row-split | 3.25 | 0.0359 | 5.1 dB | +0.170 (measured) |
| Calibrated Q3, layout B | 3.06 | 0.0359 | 4.0 dB | +0.170 |
| Calibrated Q3 restricted to -3..3, P2E2 | 2.88 | 0.0432 | 3.7 dB | about +0.20 (estimate) |
| Entropy-constrained search, lambda 0.1, P2E2 | 2.66 | 0.0544 | 3.4 dB | about +0.26 (estimate) |
| Ideal entropy-coded scalar quantizer | 2.66 | 0.0365 | 1.5 dB | about +0.17 (estimate) |
| Shannon bound | 2.66 | 0.0254 | 0 | about +0.12 (lower bound, estimate) |

The NLL estimates assume the penalty is proportional to the weighted squared error, which the two measured points support
(MSE ratio 0.57, NLL ratio 0.62; the single-document harness has 0.02 to 0.025 nats of standard error). The verdict follows from
the gaps: at 2.65 bpw the RTN-P2E2 point is 4 dB from the bound and 2.5 dB from what an ideal entropy-coded scalar quantizer
achieves at the same rate. A native 2.65 to 2.75 bpw quantizer with a better alphabet and shaping (trellis or E8 lattice, with
rotations to make the weights Gaussian-like, as QTIP and QuIP# do) can plausibly land between the 4-state TCQ and the ideal ECSQ
row, which maps to about +0.14 to +0.19 nats against +0.275 for RTN-P2E2 at equal bytes, and to about +0.17 against the calibrated
artifact's +0.170 at 0.6 bpw fewer bytes. Lossless coding of a 3-bit scalar quantizer is therefore not competitive with a finer
quantizer at the same rate; its one advantage is that the scalar quantizer's codes expand exactly to E4M3 and FP32 and so satisfy
constraints 1 and 2 without new numerics, which trellis and lattice codebooks do not (the codebook chapter's problem).

## 4. Fit with Euhedral

Against the eight constraints of 00-baseline.md section 6, for layout B (the only -4 layout worth considering), for the
scale-plane index, and for the search-side alternatives:

1. Row-exact 1 to 8 row decode. B and the scale index change only how a code value or a scale is obtained; the FMA chain is the
   contiguous kernel's, so the one-row result and the twins stay bitwise identical, exactly as P2E2 proves today. The restricted
   and entropy-constrained searches change artifacts, not kernels. Any rANS, tANS, Huffman or Tunstall decoder would also keep the
   FMA order if it placed each code at its position; they fail constraint 3, not this one.
2. Prefill at 9+ rows. Unchanged: P2E2 tensors expand to row-split before the MXFP8 route, and B expands the same way with one more
   stream to merge (expansion stays near the DRAM rate; about 0.25 ms per layer per chunk today). The scale index must be applied
   by the expansion kernel or by the MX producer; both are small changes, but the producer variant touches the hot loop.
3. Decode instruction budget. This is where B hurts: about 18 to 19 instructions per pair against 11 (P2E2) and 7 (Q3). Measured
   today, 11 against 7 costs 13% of decode; 18 against 7 is likely 25% or more (estimate), and the 4-row twins, already ALU-bound,
   lose more. The scale index costs under 0.2 instructions per pair. The fast-path limit (section 2.2) must be raised before any
   calibrated-like distribution is run through the P2E2 kernels, or they fall into the slow path almost always.
4. Determinism. Static tables and fixed addressing; nothing changes.
5. Offline conversion only. B is a new layout with its own ordinal (`WeightLayout`, `P2e2Layout`, `CompactTensorLayout`,
   `TensorLoader`, `CompactWeightLoader`, `ArtifactProfile`, `ExecutionPlan`, `ExecutionGpu`, `CudaGpuMemory`, `ScratchUse`;
   natively `p2e2.cuh`, `kernels.cu`, `embedding/kernels.cu`, `host/q3_p2e2_geometry.h`, `host/q3_linear_bf16.c`,
   `host/q3_embedding.c`; the transcoder `q3_p2e2.py` and the refusal in `pipeline.py`), which is the list `docs/P2E2_FULL_RANGE.md`
   already gives. The scale index is a new layout too, across every Q3 kernel family. The search alternatives touch only
   `grouped.py` (and the refusal, since a -4-free calibrated artifact is representable by today's transcoder).
6. The artifact decides everything. All variants are new recipes; `ArtifactProfile` reads the layout as it does for P2E2.
7. Memory is the product. Bytes saved per artifact and plane, with the Gaussian model where simulation is involved:

| Artifact and plane | Size | Lossless saving reachable in kernels | Order-0 entropy bound | Note |
|---|---|---|---|---|
| `q3`: Q3 codes, 195 tensors | 7.80 GiB | RTN: 1.56 GiB (P2E2, measured). Calibrated: 0.49 GiB (B, Gaussian) to 1.2 GiB (B, heavy-tailed) | RTN 1.8 GiB; calibrated 0.7 to 1.4 GiB | restricted search with today's P2E2: 0.9 to 1.6 GiB |
| `q3`: FP16 scales | 0.65 GiB | 0.12 to 0.16 GiB (12 to 13 bit index or exponent table); 0.24 GiB if 10 bits suffice | 0.28 GiB (43% under GDeflate) | needs a table in every Q3 kernel |
| `q3`: Q4 codes | 0.74 GiB | 0 to 0.035 GiB | 0.06 to 0.09 GiB | escape layouts lose under calibration |
| `q3`: Q5 low nibbles and fifth bits | 1.68 + 0.42 GiB | 0 | 0.15 to 0.19 GiB jointly; fifth bit alone 0 | fifth bit is the sign bit |
| `q3`: BF16 small objects | 0.05 GiB | about 0.015 GiB (exponent Huffman) | | not worth a kernel |
| `nvfp4`: E2M1 codes | 12.4 GiB | 0 | 0.31 GiB (3.90 of 4 bits) | |
| `nvfp4`: E4M3 scales | 1.55 GiB | 0.39 GiB (6-bit index) | 0.76 GiB (3.63 of 8 bits) | SD4 takes 0.78 GiB lossily |
| `nvfp4-compressed`: 4-bit indices | 0.78 GiB | 0 | small | |
| any artifact on disk | | nvfp4z 10%; P2E2 plus GDeflate 23% of Q3 row-split (measured); a context-modelled rANS about 25% (RTN) or 12% (calibrated) of Q3 (estimate) | | archive only |

8. Drafting precision. The draft head (Q3, 0.25 GiB in `q3`) and the Q3 embedding follow whatever the Q3 planes do; the MTP layer
   is NVFP4 and incompressible.

Host-backed weights. Staged bytes cross PCIe at 44 GB/s and are read once per verification, so a codec helps only if it shrinks
the staged bytes and decodes fast enough. Serially (copy, then decode) the break-even decode rate is 44 r / (1 - r): 194 GB/s at
r = 0.815, 356 GB/s at r = 0.89, 690 GB/s at r = 0.94; overlapped on the ring (decode of one slot while the next copies) the
requirement is only that decode exceeds 44 GB/s, and the staging time falls by 1 - r while the decode's own DRAM traffic
((1 + r) of the expanded bytes at 855 GB/s) is small. P2E2 expansion (450 GB/s) qualifies under either rule; nvCOMP ANS (150 GB/s)
only overlapped; GDeflate and the lane-serial rANS of nvfp4z (14 GB/s) never. The catch is that nothing compressible is
host-backed: the `q3` artifact fits whole at 128K, and in the nvfp4 forms the codes carry 3.90 of 4 bits while the scales are
already SD4 indices. A P2E2-style path would matter only for a future artifact that stages Q3 or Q5 planes.

## 5. Experiment plan

In order, each one decisive for the next:

1. Measure the calibrated artifact (hours, no GPU). A read-only script using `euhedral_artifacts.edrl.read_table` and
   `q3_p2e2.unpack_q3` over the existing calibrated `q3.edrl`, per tensor and summed: the Q3 code histogram and entropy, P(BIG),
   P(-4), -4 count per group and co-occurrence with -3, the BIG count per 32-code lane (fraction above 16 and above 30), payload
   words per 1024-code slice (fraction above 32), the Q4 and Q5 histograms and the fifth-bit plane's entropy, and for the FP16
   scales the distinct count per tensor and the order-0 entropy of the 16-bit pattern and of its exponent and mantissa halves. Then
   compute the bytes of P2E2 (if no -4), A, B, S and the restricted layout directly from the histograms. Go for a -4 layout only if
   P(BIG) <= 0.25 and P(-4) <= 0.01 (the Student-t rows), which gives B at about 0.85 to 0.87; no-go above P(BIG) = 0.30 (B at or
   above 0.92, under 0.7 GiB saved for a slower kernel). The same run decides the scale index: 10-bit if every tensor has at most
   1024 distinct scales, otherwise 12 to 13 bits.
2. Cheapest kernel-free test of calibrated-like statistics (hours). Convert with the search restricted to -3..3 (one argument in
   `grouped.py`), transcode with the unchanged `q3_p2e2.py`, record the ratio from the manifest, run the teacher-forced report
   (`TeacherForcedQualityCudaIntegrationTest`, `compare_reference.py`) and the 4K and 32K decode benchmarks. This measures at once
   how much of the calibrated gain survives the restriction (model: 73%, about +0.20 nats; the harness resolves 0.05), the real
   ratio (model: 0.886), and the slow-path effect on the P2E2 kernels (model: 68% of slices). Go/no-go for the restricted artifact:
   NLL within 0.05 nats of the calibrated `q3` and decode within 15% of `q3`. Add two entropy-constrained points (lambda 0.05 and
   0.1 in the units of unit-variance weights, scaled by the group's weighted variance) to see whether the MSE-bytes curve is as flat
   near the RTN byte count as the model says.
3. Kernel fast-path limit (days). Raise the 16-BIG limit (the window arithmetic suggests about 30) and the slice window from 32 to
   48 words, verify against the scalar reference and `P2e2CudaIntegrationTest`, and time cold gate_up and down at 1 to 4 rows with
   an RTN artifact (must not regress) and the restricted artifact of step 2.
4. Layout B in the transcoder (days): encode, independent decode-back, tests with tensors that use -4, the manifest's ratio on the
   real calibrated artifact. No kernel yet; this fixes the bytes.
5. Layout B kernels (weeks): decode for 1 to 4 rows, expansion, embedding gather, host geometry, Java layout classes. Acceptance:
   bitwise equality with the calibrated `q3` in `QwenP2e2CudaIntegrationTest` and byte-identical teacher-forced reports; decode at
   4K within 25% of `q3` (no-go beyond 30%); 3-row verification time on cold gate_up within 1.5x of today's 134 us.
6. Scale index (weeks, independent of 1 to 5): converter writes the per-tensor table, loader validates it, every Q3 kernel family
   gets a twin; acceptance is bitwise identity and under 2% decode cost for 0.12 to 0.24 GiB.

## 6. Risks and open questions

- The central numbers are from an i.i.d. Gaussian model of a group of 64, chosen because it reproduces the two RTN statistics the
  repository reports. Real groups have outliers, correlated columns and importance weights with a wide dynamic range; the
  heavy-tailed rows show the direction (fewer BIG codes, more saving). Step 1 of the plan replaces the model with data.
- The instruction counts for layouts A, B and S are read off the kernel source, not measured; the 11-versus-7 figure is the
  repository's and the extrapolation to 18 is linear in instructions, which the ALU-bound 4-row twins may not follow.
- The 16-BIG fast-path limit: whether the check can be loosened to about 30 without a new window word needs a proof against the
  funnel-shift clamping, not the reading in section 2.2.
- The NLL estimates rest on proportionality between weighted MSE and NLL penalty across methods, supported by two measured points
  on one document; the harness cannot resolve differences under 0.05 nats for Q3, so steps 2 and 5 need a larger held-out set
  if they come out close.
- Byte identity between `q3` and `q3-compressed` is a product decision if the search is restricted or entropy-constrained: either
  both artifacts give up part of the calibrated gain, or the two recipes diverge (the repository's tests currently assume identity).
- Literature coverage is limited to repositories read in this session; the published GPU throughput of DFloat11's decoder, nvCOMP's
  own benchmark tables, and 2025 to 2026 in-kernel entropy decoders were not fetched. None of them changes the arithmetic of
  section 2.4, which is set by the engine's own instruction budget.
- Nothing here covers NVFP4 beyond restating the repository's entropies: at 3.90 of 4 bits the only remaining lever is lossy code
  reduction, which is other chapters' subject.

## 7. References

1. Euhedral repository, branch `perf/mtp-gated-depth`: `docs/COMPRESSED_Q3.md`, `docs/P2E2_FULL_RANGE.md`,
   `docs/COMPACT_Q3_REFERENCE.md`, `docs/NVFP4_LOSSLESS.md`, `docs/NVFP4_COMPRESSED.md`, `native/src/q3/p2e2.cuh`,
   `native/src/q3/contiguous.cuh`, `native/src/q3/kernels.cu`, `native/src/embedding/kernels.cu`,
   `tools/euhedral_artifacts/q3_p2e2.py`, `tools/euhedral_artifacts/grouped.py`, `tools/nvfp4z/codec.py`, `tools/nvfp4z/kernels.cu`.
2. Zhang, Hariri, Zhong, Chaudhary, Sui, Hu, Shrivastava. "70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU
   Inference via Dynamic-Length Float (DFloat11)". NeurIPS 2025. arXiv 2504.11651. Code read: https://github.com/LeanModels/DFloat11
   (`dfloat11/decode.cu`, `dfloat11/dfloat11.py`, `dfloat11/dfloat11_utils.py`, README).
3. "ZipNN: Lossless Compression for AI Models". arXiv 2411.05239. Code and README read: https://github.com/zipnn/zipnn.
4. "NeuZip: Memory-Efficient Training and Inference with Dynamic Compression of Neural Networks". arXiv 2410.20650. Code read:
   https://github.com/BorealisAI/neuzip (`cuda/high_level.cu`, `neuzip/binding.py`).
5. Duda. "Asymmetric numeral systems". arXiv 1311.2540 (cited by references 6 and 7; not fetched).
6. Giesen. "Interleaved entropy coders". arXiv 1402.3392; implementation and CPU timings: https://github.com/rygorous/ryg_rans (README).
7. Collet. FiniteStateEntropy (Huff0, FSE): https://github.com/Cyan4973/FiniteStateEntropy (README benchmarks).
8. NVIDIA nvCOMP: https://github.com/NVIDIA/nvcomp (archived README; no throughput figures there; the ANS and GDeflate numbers in
   this chapter are the Euhedral measurements of reference 1).
9. ggml / llama.cpp: `ggml/src/ggml-common.h` (`block_tq1_0`, `block_tq2_0`) and `ggml/src/ggml-quants.c`, fetched from
   https://raw.githubusercontent.com/ggml-org/llama.cpp/master/.
10. Tseng, Sun, Hou, De Sa. "QTIP: Quantization with Trellises and Incoherence Processing". NeurIPS 2024. arXiv 2406.11235.
    README read: https://github.com/Cornell-RelaxML/qtip.
11. Tseng et al. "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks". ICML 2024. arXiv 2402.04396.
    README read: https://github.com/Cornell-RelaxML/quip-sharp.
12. Gish and Pierce, "Asymptotically efficient quantizing", IEEE Trans. Inf. Theory, 1968 (the 0.255-bit high-rate gap; not
    fetched, reproduced numerically in section 3). Farvardin and Modestino, "Optimum quantizer performance for a class of
    non-Gaussian memoryless sources", 1984; Marcellin and Fischer, "Trellis coded quantization of memoryless and Gauss-Markov
    sources", 1990; Max, "Quantizing for minimum distortion", 1960 (Lloyd-Max values reproduced numerically); Tunstall,
    "Synthesis of noiseless compression codes", 1967. Classical results cited by title; none was fetched in this session.
13. Simulation scripts written for this chapter (numpy): `sim_search.py` (the grouped.py search on synthetic groups),
    `sim2.py` (restricted and entropy-constrained searches, Lloyd-Max, 4-state TCQ, Q4/Q5 escape layouts), `ecsq_rd.py`
    (entropy-coded uniform quantization against the Shannon bound), kept in the research scratchpad.
