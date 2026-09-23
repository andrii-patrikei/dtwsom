# DTW-SOM in Rcpp - plan, design and roadmap

Paper: Silva & Henriques (2020), *Exploring time-series motifs through DTW-SOM*, IJCNN. arXiv:2004.08176.
Reference code (Python, MIT): https://github.com/misilva73/dtw_som (`pip install dtw_som`), built on
`pyclustering.nnet.som` + `dtaidistance`. Other Python takes: `Kenan-Li/dtwsom` (minisom fork, uni/multivariate),
`somtime` (PyPI). None of them is parallel or GPU-aware; the reference one is pure Python loops.

## 1. What the paper changes vs. a vanilla SOM

1. **Distance** - DTW instead of Euclidean (dtaidistance, with `window`, `max_step`, `max_length_diff`).
2. **Initialisation** - (a) *random sample*: prototypes are copies of randomly chosen input series (so they have real,
   variable lengths); (b) *anchors*: user-supplied series placed on the two diagonals of the map, rest filled with a sample.
3. **Adaptation with variable lengths** - the BMU search also returns the warping path. For prototype index `i` the
   update target is the **mean of all x[j] matched to i** on the optimal path:
   `w[i] += lr * h * (mean_{j~i} x[j] - w[i])`. Prototypes keep the length they were initialised with.
4. Small details of their code worth knowing: online training, one series at a time; neighbourhood
   `h = exp(-d / (2 r²))` if `d < r²` with `r = r0·exp(-epoch/E)` (the pyclustering formula, note the squared radius);
   `lr = lr0·exp(-epoch/E)`; after each epoch DTW `max_dist` is set to 1.1 × the largest BMU distance and used to
   abandon hopeless DTW computations early (falls back to unbounded if everything is pruned).

## 2. Why not just `kohonen`

`kohonen` (Rcpp) is, as far as I can tell, single-threaded, and its update is the index-wise weighted mean of
fixed-length vectors. Since 3.0 you *can* plug a custom C++ distance (external pointer) into `supersom()`, so
"DTW for BMU + Euclidean update on fixed-length series" is a 30-line baseline - but the paper's whole point is the
path-based update on variable-length series, and that needs its own core.

## 3. What is in this repository (R package layout)

```
dtwsom/
├── src/dtwsom_core.hpp     R-free C++17 + OpenMP core, templated on the number type: banded DTW with early
│                            abandoning, DTW with backtracking that accumulates path matches, online (paper) and
│                            batch training, exact mapping, parallel pairwise/cross DTW matrices
├── src/dtwsom_rcpp.cpp     Rcpp glue (sourceCpp-ready) with run-time `precision` dispatch:
│                            float / double / longdouble / quad / bin50 (+ mpfr100 when built with MPFR)
├── src/dtw_lanes.cpp       SIMD-across-prototypes DTW kernel (AVX-512 demo, no intrinsics)
├── src/Makevars(.win)      OpenMP + -O3 when built as a package
├── R/dtwsom.R              dtw_som(), predict(), plot() (codes / counts / umatrix / qe / elastic / members /
│                            members_dtw), plot_members(), plot_members_kohonen(), dtw_warp(), anchor_init(),
│                            umatrix(), elastic_layout(), dtw_pairwise(), as_kohonen() (kohonen/aweSOM interop)
├── example.R               synthetic-motif demo (as in the paper), batch vs online, predict, super-clusters, timing
├── example_compare.R       iris + kohonen::som equivalence test, warped motifs vs kohonen, aweSOM evaluation/plots
├── precision_test.Rmd      does more precision help? + AVX-512 build check and lane-kernel benchmark
├── tests/test_core.cpp     g++-only tests: DTW == brute force, pruning is exact, both modes separate motifs
└── tests/test_precision.cpp g++-only: every number type compiles and trains (incl. MPFR)
```

Install: `remotes::install_github("andrii-patrikei/dtwsom")` (or `R CMD INSTALL .` from a clone); examples live in
`inst/examples/` and run with `source(system.file("examples", "example.R", package = "dtwsom"))`.
Put `CXX17FLAGS = -O3 -march=native` in `~/.R/Makevars` (Rtools 4.x on Windows supports OpenMP).
Knit `precision_test.Rmd` for the precision and AVX-512 experiments (takes ~2 min at double, longer
for the software float types).

Verified here: distances identical to a full-matrix reference on 200 random pairs (multivariate too);
`prune=TRUE` gives bit-identical QE/TE to `prune=FALSE`; on 400 synthetic motifs of length 40-90, 4×4 hex map,
15 epochs: batch purity 0.99, online purity 1.00, ~1.4 s **single-threaded** at -O2.

**Equivalence to kohonen (example_compare.R).** With `window = 0` the DTW path is the diagonal and the batch
update is the plain weighted mean, so the DTW-SOM *is* a batch Euclidean SOM. On scaled iris, same initial
codebook, kohonen's radius schedule `r0 − r0·m/E`, Gaussian neighbourhood, `nb_cutoff = 0`: unit assignments
agree 100 % and codebooks agree to 2·10⁻¹⁶ at 3, 30 and 100 epochs; `aweSOM::somQuality` returns identical
numbers for both objects. Two things had to be aligned to get there, both worth knowing when you compare
against kohonen elsewhere: (1) kohonen's hexagonal `somgrid` shifts the *odd* rows by 0.5 (I had shifted the
even rows - a valid lattice, but a different neighbourhood); (2) kohonen breaks exact BMU ties **at random**
(`++nind * UNIF < 1.0` in `FindBestMatchingUnit`), so with a bubble neighbourhood and a large first radius -
where many units collapse to identical codes - its batch results are not reproducible without `set.seed()`
and will not match a deterministic implementation. Use Gaussian for comparisons.

## 4. Where the parallelism is

The paper's algorithm is *online*: observation after observation, so it can only parallelise the K DTW calls per
observation (K = 25-100 → poor use of 32 threads). That is why the core has two modes:

* **online** (paper-faithful): OpenMP over prototypes for the BMU search (shared atomic running-best for
  early abandoning), then OpenMP over the prototypes inside the neighbourhood for the path updates.
* **batch** (recommended): epoch = pass 1 (OpenMP over the N series: BMU with pruning, previous BMU tried
  first as a tight bound) + pass 2 (per prototype, OpenMP over the series with `h ≥ 1e-3`, thread-local
  accumulators, then `w[i] = Σ h·Σ_{j~i} x[j] / Σ h·|{j~i}|`). This is a neighbourhood-weighted DBA step, so the
  batch DTW-SOM is "DBA k-means with a topology". Deterministic, order-independent, scales ~linearly in cores.

Cost per epoch ≈ N·K distance-only DTWs (O(T·(2w+1)·D), two rows of memory, early abandoning) plus
N·|neighbourhood| path DTWs (O(T·(2w+1)) plus a (T+1)×(T'+1) byte matrix for backtracking, per thread).
For a typical IMU study, say N = 648 windows, K = 25, T = 1000, w = 100, D = 6: ~30-50 k DTWs/epoch ≈ 30-60 s on
one core, ~3-4 s on a 16-core Ryzen 9 9950X. GPU is not needed at that scale.

## 5. Roadmap

**Phase 0 (done)** - sourceCpp workflow, core tests, demo. Next: run it on the Acta Gymnica IMU windows.

**Phase 1 - package polish**
- Done: `DESCRIPTION`, `NAMESPACE`, `Rcpp::compileAttributes()`, builds and installs. Still to do: roxygen `man/`
  pages (until then `R CMD check` warns about undocumented exports), a testthat port of `cpp_tests/`, and a
  GitHub Actions `R CMD check` workflow (r-lib/actions) once the docs exist so the check passes.
- Benchmarks: vs `kohonen::som` (Euclidean, fixed length) and vs the Python `dtw_som` on the paper's GunPoint
  (UCR) data so the README has an apples-to-apples comparison and a thread-scaling plot.

**Phase 2 - CPU speed (where AVX-512 pays off)**
- `precision = "float"` exists now; make it safe by default (z-normalise) and accumulate means in double.
- Lower bounds before DTW in the BMU search: LB_Kim (endpoints, free) and LB_Keogh (envelope of the query
  under the band) - typically prunes 80-95 % of full DTWs. Needs interpolation to a common length for
  unequal lengths, or use it only when |n−m| ≤ w.
- SIMD across pairs: the DTW inner loop is a serial dependence (`cur[j-1]`), so the profitable vectorisation is
  lockstep over 8 (AVX-512) or 16 (float) *prototypes* for one query. `src/dtw_lanes.cpp` proves it
  (~15-28× vs the scalar loop); integrate it into the BMU pass with per-lane band masks and length buckets.
  This is the same layout a GPU kernel wants, so it is the natural stepping stone to Phase 3.

**Phase 3 - GPU (optional, only if Phase 2 numbers say so)**
- Keep the core's backend split: BMU pass (distances only) is the embarrassingly parallel N×K part - one CUDA
  thread per (series, prototype) pair with a Sakoe-Chiba band in registers/shared memory; the path/backtracking
  pass stays on the CPU (it only covers the neighbourhood, which shrinks as the radius decays).
- Linking into R: package with a `Makevars` that runs `nvcc` into a static lib and links it into the shared
  object (Linux is straightforward; Windows + Rtools/gcc + nvcc/MSVC ABI is the pain point - do it on Linux/WSL).
  Alternative that avoids all of that: keep the GPU kernel in Python (CuPy/Numba, or `pytorch-softdtw-cuda`)
  and call it through `reticulate` - the R side of this package already speaks lists of matrices.
- Rule of thumb: worth it above ~10⁵ pairs per epoch or T in the thousands; below that a 16-core CPU with
  Phase 2 wins on effort/return.

## 6. Improvements to the SOM itself (prioritised)

1. **Batch mode + pruning** - done; makes everything else affordable.
2. **Smoother prototypes.** The mean-of-matched-points update (and plain DBA) produces noisy prototypes with
   plateaus and endpoint spikes. Done: `smooth = k` running mean after each batch update (k = 3 recommended, see
   section 10). Further options: tighter `window`; per-series z-normalisation (`normalize = "z"`);
   slope-constrained steps (Itakura / max-step); **soft-DTW barycenters** (Cuturi & Blondel 2017, γ ≈ 0.1-1) -
   smooth, differentiable, and GPU-native, which links Phase 3 and this item.
3. **Prototype length adaptation.** The paper freezes lengths at init. Every few epochs re-sample each prototype
   to the median length of its members (linear interpolation), or resample all to one length; needs a repack of
   the codebook (small change in the core).
4. **Dead-unit handling.** Units with no hits and no neighbourhood contribution: re-initialise to the worst-fitted
   series (largest BMU distance). Same repack machinery as 3.
5. **Multivariate options.** Implemented: dependent DTW (one path for all channels, `DTW_D`). Add independent
   DTW (`DTW_I`, sum of per-channel DTWs) and per-channel weights - for 6-axis IMU data the choice matters
   (Shokoohi-Yekta et al. 2017 show neither dominates).
6. **Distance variants.** Derivative DTW / shapeDTW (match local shape descriptors) as `distance=` options for
   accelerometer signals where offsets are not meaningful.
7. **Better initialisation.** Besides sample/anchor: MDS of a DTW distance matrix on a subsample, place the grid
   on the first two MDS axes and pick the nearest series for each unit (the DTW analogue of kohonen's PCA init).
   More reproducible maps, fewer topological folds.
8. **Quality metrics & model selection.** QE, TE, U-matrix are in. Add Silhouette / Davies-Bouldin /
   Calinski-Harabasz on the DTW distance matrix of the mapped data (the validity indices used in the Acta Gymnica paper),
   and a grid-size / window sweep helper.
9. **Supervised layer.** For labelled data (e.g. the 8 movement tasks): add a class-indicator layer with a weight, like
   kohonen's `xyf`, and report purity per unit, e.g. on the movement-classification data.
10. **Second-level clustering** of prototypes via `dtw_pairwise(codes)` + `hclust` (in both examples, with
    aweSOM's dendrogram) - the "super-clusters" step from the Acta Gymnica paper, now with a DTW-consistent distance.
11. **Members overlay** (done): `plot(fit, "members")` draws each prototype in red over the series mapped to
    it on their own sample index - the variable lengths are visible, and a DTW-SOM is the only SOM that can
    hold them; `plot(fit, "members_dtw")` warps the members onto the prototype's axis along their DTW paths
    (`dtw_warp()`), which shows what the barycenter averages and why a step edge stays sharp under DTW but
    smears into a slope under a Euclidean mean (`plot_members_kohonen(koh)` for the comparison).
12. **Interactive reporting.** Plotly/trelliscope codebook with member overlays and per-unit hit tables,
    reusing the reporting approach from the original IT4I analysis.

## 7. Precision: does anything above `double` help? (measured, see precision_test.Rmd)

The core is templated on the number type, so `precision = "quad"` etc. run the identical algorithm.
Findings on the synthetic motif set (400 series, 25/9 units) and its copy on a 10⁶ DC offset:

| question | float | double | long double / quad / 50 digits |
|---|---|---|---|
| relative error of one DTW distance | 2·10⁻⁷ (scaled), 10⁻³ (offset) | 2·10⁻¹⁶ | 10⁻²⁰ / 10⁻³⁴ / 0 |
| BMU flips vs 50-digit reference, scaled data | 0 / 400 | 0 | 0 |
| BMU flips, 10⁶-offset data | 52 / 400 | 0 | 0 |
| trained map vs quad (assignments / prototypes) | 99.5 % / 4·10⁻³ (scaled); 9 % / broken (offset) | 100 % / 2·10⁻¹⁴ | - |
| cost relative to double | 1.0 (same speed today) | 1 | 1.9 / 25 / 32 |

Interpretation: a BMU can only flip when the margin between best and second-best unit is below the
arithmetic error; with `double` the margin (10⁻²…10⁻¹) is fourteen orders of magnitude above the error.
So MPFR/quad are a *diagnostic* (run a suspicious subset at `"quad"` and confirm nothing changes), not a
production option. The direction worth pursuing is the opposite one: `float` is exact enough on
z-normalised data and halves memory traffic / doubles SIMD width - but it must be guarded by
normalisation, because a DC offset destroys it. If you need MPFR anyway, build with
`-DDTWSOM_WITH_MPFR` and `PKG_LIBS="-lmpfr -lgmp"`; Boost's `mpfr_float_backend<100>` is wired in.

## 8. AVX-512 on Zen 5 (Ryzen 9 9950X)

Nothing AMD-specific: it is Intel's ISA, implemented double-pumped (256-bit datapath) on Zen 4 and with
a full 512-bit datapath on Zen 5. What matters in practice:

1. `-march=native` (or `-march=znver5` on GCC ≥ 14 / Rtools 4.5, `znver4` on GCC 13 / Rtools 4.4)
   in `~/.R/Makevars`; `dtwsom_build_info_cpp()` reports whether the build saw `__AVX512F__`.
2. GCC still prefers 256-bit vectors on most targets; add `-mprefer-vector-width=512` and measure
   (`-fopt-info-vec-optimized` prints "64 byte vectors" when it took effect). On the Intel Xeon used
   for the first benchmarks, 512-bit was *not* faster than 256-bit; Zen 5 is expected to behave better - test it.
3. The DTW recursion has a loop-carried dependency and will never vectorise as written. The
   `src/dtw_lanes.cpp` kernel lays the codebook out lane-major (prototype = lane, time step = row)
   and runs all prototypes in lockstep; `#pragma omp simd` then vectorises the lane loop. Measured
   against the scalar recursion: ~15× (double) and ~28× (float) for 64 prototypes × 1000 samples,
   with no intrinsics. This is the Phase-2 design; what is missing is the band (per-lane masks) and
   bucketing prototypes by length, after which it replaces `dtw_distance` in the BMU pass.
4. Portable binaries: on Linux use `__attribute__((target_clones("avx512f","avx2","default")))`; on
   Windows (no ifunc) switch manually with `__builtin_cpu_supports("avx512f")`.

## 9. "Pairwise elastic SOM" (Hartono & Take, WSOM 2017) - is it worth adding?

What it is (from the abstract and the authors' later papers that build on it; the full text is behind
IEEE's paywall, so I have not read the training equations): the map units are not fixed on a rigid
2-D grid but move ("elastic") so that their 2-D distances reflect the pairwise distances of the reference
vectors - a SOM/MDS hybrid aimed at visual fidelity, not at better clustering.

Assessment: the goal is legitimate and old - ViSOM (Yin, 2002) and Merkl & Rauber's adaptive
coordinates (1997) do the same thing - and this paper is a 7-page workshop contribution with few
citations, later superseded by the same author's Topological/Siamese Topological Neural Networks.
It would be unfair to call it badly written without reading it, but it is not a method the field
adopted, and folding node motion into the training loop would complicate a DTW-SOM for a purely
visual benefit.

What I did instead: the benefit is obtainable *post hoc*. `elastic_layout()` / `plot(fit, "elastic")`
relax the unit positions by Sammon mapping of the prototype DTW distance matrix, initialised from the
grid so orientation is preserved. It costs one K×K DTW matrix, leaves training untouched, keeps the hex
grid for the U-matrix, and shows exactly what PE-SOM promises (in the demo the three motif families
pull apart into three lobes). Recommendation: keep it as a plot option, do not restructure the trainer.

## 10. kohonen / aweSOM interoperability

`as_kohonen(fit, len)` wraps a fit as a `kohonen` object (variable-length series are linearly resampled to a
common length, multivariate channels are concatenated), so `kohonen::plot`, `aweSOM::aweSOMplot` (Hitmap,
UMatrix, Line = prototype series per cell, CatBarplot of labels, …), `aweSOMsmoothdist`, `aweSOMscreeplot`,
`aweSOMdendrogram` and `somQuality` all work. Caveat: aweSOM's quality measures are Euclidean on the
resampled vectors - exact for `window = 0` on fixed-length data, an approximation for real DTW fits, where
`fit$qe` / `fit$te` are the DTW-consistent numbers. For super-classes cluster the prototypes on
`dtw_pairwise(fit$codes)` and hand the `hclust` object to `aweSOMdendrogram`.

### Euclidean SOM vs DTW-SOM - where DTW actually helps (5 initialisations each, same init & schedule)

| data | kohonen purity | DTW-SOM purity | kohonen TE | DTW-SOM TE |
|---|---|---|---|---|
| warped motifs (global stretch `u^γ`, resampled to 64) | 0.987 | 1.000 | 0.045 | 0.090 (0.043 with `smooth = 3`) |
| Cylinder-Bell-Funnel, z-normalised (local onset/duration) | 0.891 | 0.980 | 0.199 | 0.138 |
| same shapes, random onset only | 1.000 | 1.000 | 0.040 | 0.117 |

Honest reading: on my motif set the classical SOM is at least as good - a smooth *global* time-stretch is
neutralised by resampling to a common length, and the shapes are too distinct for the distance to matter. DTW
pays off when instances of one class are misaligned *locally* (CBF: every one of 5 inits, 0.96-0.99 vs
0.86-0.93). Expect real IMU data to sit between these cases: repetitions of one movement differ in onset and
local tempo, which is the CBF situation.

**Why the DTW prototypes looked noisy.** DTW aligns noise with noise, so a barycenter of matched points keeps
the noise instead of averaging it out (a known DBA property). `smooth = 3` (running mean of width 3 after each
batch update) brings prototype roughness from 0.22 to 0.036 - kohonen's Euclidean means are at 0.042 - and
the mean topographic error from 0.090 to 0.043, with purity unchanged (1.000 → 0.995). Widths above 5 start
to cost topology. Default stays 0 (paper-faithful; also `window = 0` on non-series data like iris must not be
smoothed); use 3 for time series.

## 11. Known limitations of this version

- Prototype lengths are fixed after initialisation (as in the paper) - see items 3-4.
- Path pass allocates a (T+1)×(T'+1) byte matrix per thread; fine up to T ≈ 10⁴ (100 MB/thread at 10⁴×10⁴).
- Epoch QE printed during training is measured at the *start* of the epoch (before that epoch's update);
  the final `qe`/`te` in the returned object are exact.
- No NA handling inside series (rows with NA are dropped by `as_seq_list` for matrix/array input).
