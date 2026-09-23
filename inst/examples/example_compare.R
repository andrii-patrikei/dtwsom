# example_compare.R - DTW-SOM vs kohonen::som, evaluated and plotted with aweSOM
#
#   install.packages(c("kohonen", "aweSOM")); remotes::install_github("andrii-patrikei/dtwsom")
#   source(system.file("examples", "example_compare.R", package = "dtwsom"))
#
# Part A  iris: with window = 0 the DTW-SOM *is* a batch Euclidean SOM, so it must
#         reproduce kohonen's batch mode from the same initial codebook. It does, to
#         1e-16 - provided the neighbourhood is Gaussian (kohonen breaks exact BMU
#         ties at random, and a bubble neighbourhood with a large first radius makes
#         many units identical, so ties are common) and nb_cutoff = 0 (kohonen keeps
#         every h > 0, however tiny).
# Part B  synthetic motifs with random time-warping: kohonen (Euclidean on series
#         resampled to a common length) vs DTW-SOM (variable length, elastic).
#         Honest result: a draw. The warp is a smooth *global* stretch, which
#         resampling already neutralises, and the four shapes are trivially distinct,
#         so both reach ~99-100% purity; the Euclidean map has the better topology.
# Part C  Cylinder-Bell-Funnel (Saito 1994), the standard synthetic benchmark where
#         the misalignment is *local* (random onset and duration): here DTW-SOM wins
#         clearly (purity ~0.98 vs ~0.89 over 5 inits).
# Part D  a real UCR dataset (GunPoint, used in the DTW-SOM paper) if you download it.
# aweSOMplot() returns htmlwidgets: shown in the RStudio viewer when interactive,
# otherwise written to ./aweSOM_plots/*.html.

library(dtwsom); library(kohonen); library(aweSOM)

show <- function(w, name) {
  if (interactive()) print(w) else { dir.create("aweSOM_plots", showWarnings = FALSE)
    htmlwidgets::saveWidget(w, file.path(getwd(), "aweSOM_plots", paste0(name, ".html")), selfcontained = FALSE) }
  invisible(w)
}
purity <- function(bmu, lab) sum(apply(table(bmu, lab), 1, max)) / length(lab)

# ============================================================================
# A. iris - sanity check: DTW-SOM(window = 0) must equal kohonen batch SOM
# ============================================================================
X <- scale(as.matrix(iris[, 1:4]))
K <- 25; grid <- somgrid(5, 5, "hexagonal", neighbourhood.fct = "gaussian")
set.seed(3); init <- X[sample(nrow(X), K), ]
E <- 30
r0 <- quantile(unit.distances(grid), 2/3)                 # kohonen's default start radius
sched <- r0 - (r0 - 0) * (0:(E - 1)) / E                  # kohonen's batch schedule, r0 -> 0

set.seed(1)
koh <- som(X, grid = grid, rlen = E, radius = c(r0, 0), mode = "batch", init = list(init))

x_iris <- lapply(seq_len(nrow(X)), function(i) X[i, ])   # each row = a "series" of length 4
fit_e <- dtw_som(x_iris, 5, 5, topo = "hexagonal", epochs = E, mode = "batch", window = 0,
                 neighborhood = "gaussian", nb_cutoff = 0, radius = sched,
                 codes = lapply(seq_len(K), function(k) init[k, ]), verbose = FALSE)

cat("\n=== iris: DTW-SOM(window=0) vs kohonen batch ===\n")
cat(sprintf("unit assignment agreement : %.1f%%\n", 100 * mean(fit_e$unit.classif == koh$unit.classif)))
cat(sprintf("max |codebook difference| : %.2e\n", max(abs(as_kohonen(fit_e)$codes[[1]] - koh$codes[[1]]))))
cat(sprintf("purity vs Species         : dtwsom %.3f   kohonen %.3f\n",
            purity(fit_e$unit.classif, iris$Species), purity(koh$unit.classif, iris$Species)))

# aweSOM quality measures on both (Euclidean - exactly right here)
cat("\n-- aweSOM::somQuality, kohonen --\n");  print(somQuality(koh, X))
cat("\n-- aweSOM::somQuality, dtwsom(window=0) --\n"); print(somQuality(as_kohonen(fit_e), X))

# What DTW does to *unordered* features: warping across Sepal/Petal columns is
# meaningless, so never run DTW on iris without window = 0. Shown for the record.
fit_w <- dtw_som(x_iris, 5, 5, epochs = E, window = NULL, neighborhood = "gaussian", nb_cutoff = 0,
                 radius = sched, codes = lapply(seq_len(K), function(k) init[k, ]), verbose = FALSE)
cat(sprintf("\n(for the record) full DTW across the 4 iris columns: purity %.3f, QE %.3f  vs  window=0: purity %.3f, QE %.3f\n",
            purity(fit_w$unit.classif, iris$Species), fit_w$qe, purity(fit_e$unit.classif, iris$Species), fit_e$qe))

# aweSOM plots (iris)
sc_iris <- cutree(hclust(dist(koh$codes[[1]]), "ward.D2"), 3)
show(aweSOMplot(koh, type = "Hitmap", superclass = sc_iris), "iris_kohonen_hitmap")
show(aweSOMplot(as_kohonen(fit_e), type = "Hitmap", superclass = sc_iris), "iris_dtwsom_hitmap")
show(aweSOMplot(koh, type = "Circular", data = X, variables = colnames(X), superclass = sc_iris), "iris_kohonen_circular")
show(aweSOMplot(koh, type = "CatBarplot", data = iris, variables = "Species", superclass = sc_iris), "iris_kohonen_species")
aweSOMsmoothdist(koh)

# ============================================================================
# B. synthetic motifs with time-warping - Euclidean SOM vs DTW-SOM
# ============================================================================
make_motifs <- function(n = 400, seed = 1, warp = TRUE) {
  set.seed(seed)
  shapes <- c("sine", "triangle", "square", "ramp")
  lab <- sample(shapes, n, replace = TRUE)
  x <- lapply(lab, function(s) {
    L <- sample(40:90, 1); u <- seq(0, 1, length.out = L); a <- runif(1, 0.7, 1.3)
    if (warp) u <- u^runif(1, 0.6, 1.7)                    # random monotone time-warp
    y <- switch(s, sine = sin(2 * pi * u), triangle = 1 - 4 * abs(u - 0.5),
                square = ifelse(u < 0.5, 1, -1), ramp = 2 * u - 1)
    a * y + rnorm(L, sd = 0.15)
  })
  list(x = x, label = lab)
}
d <- make_motifs()
Xm <- flatten_seqs(as_seq_list(d$x), 64); colnames(Xm) <- paste0("t", 1:64)   # fixed length for kohonen
gm <- somgrid(4, 4, "hexagonal", neighbourhood.fct = "gaussian")
set.seed(42); idx <- sample(length(d$x), 16)
Em <- 30; r0m <- quantile(unit.distances(gm), 2/3); schedm <- r0m - r0m * (0:(Em - 1)) / Em   # same schedule for both

set.seed(1)
koh_m <- som(Xm, grid = gm, rlen = Em, radius = c(r0m, 0), mode = "batch", init = list(Xm[idx, ]))
# smooth = 3: running mean over the prototypes after each update. DTW aligns noise
# with noise, so DBA-style barycenters keep it; width 3 brings prototype roughness
# and topographic error to kohonen's level without changing purity (see PLAN.md).
fit_m <- dtw_som(d$x, 4, 4, topo = "hexagonal", epochs = Em, mode = "batch", window = 10,
                 neighborhood = "gaussian", nb_cutoff = 0, radius = schedm, codes = d$x[idx],
                 smooth = 3, verbose = FALSE)

cat("\n=== warped motifs: kohonen (Euclidean, resampled to 64) vs DTW-SOM ===\n")
cat(sprintf("purity : kohonen %.3f   dtwsom %.3f\n", purity(koh_m$unit.classif, d$label), purity(fit_m$unit.classif, d$label)))
cat(sprintf("topographic error : kohonen %.3f (aweSOM)   dtwsom %.3f (DTW)\n",
            somQuality(koh_m, Xm)$err.topo, fit_m$te))
cat("\nkohonen units x shape\n"); print(table(koh_m$unit.classif, d$label))
cat("\ndtwsom units x shape\n");  print(table(fit_m$unit.classif, d$label))

# prototypes (red) with their members: kohonen needs one common length, the DTW-SOM
# takes the series as they are; the third plot shows how DTW aligns them.
plot_members_kohonen(koh_m)
plot_members(fit_m, "index")
plot_members(fit_m, "dtw")

# superclasses from the prototypes: hierarchical clustering on *DTW* distances
# between prototypes (aweSOM's own screeplot clusters the flattened Euclidean codes)
kfit <- as_kohonen(fit_m, len = 64)
aweSOMscreeplot(kfit, nclass = 4, method = "hierarchical")
hc <- hclust(as.dist(dtw_pairwise(fit_m$codes, window = 10)), "ward.D2"); aweSOMdendrogram(hc, nclass = 4)
sc_m <- cutree(hc, 4)
cat("\nsuperclasses (from DTW-SOM prototypes) x shape\n"); print(table(sc_m[fit_m$unit.classif], d$label))

# interactive aweSOM plots: hit map, U-matrix, prototype line plots per cell
show(aweSOMplot(kfit,  type = "Hitmap",  superclass = sc_m), "motifs_dtwsom_hitmap")
show(aweSOMplot(kfit,  type = "UMatrix", superclass = sc_m), "motifs_dtwsom_umatrix")
show(aweSOMplot(kfit,  type = "Line", data = kfit$data[[1]], variables = colnames(Xm),
                superclass = sc_m, values = "prototypes", showAxes = FALSE), "motifs_dtwsom_prototypes")
show(aweSOMplot(koh_m, type = "Line", data = Xm, variables = colnames(Xm),
                values = "prototypes", showAxes = FALSE), "motifs_kohonen_prototypes")
show(aweSOMplot(kfit,  type = "CatBarplot", data = data.frame(shape = d$label), variables = "shape",
                superclass = sc_m), "motifs_dtwsom_shapes")
# ============================================================================
# C. Cylinder-Bell-Funnel - local misalignment, 5 initialisations each
# ============================================================================
make_cbf <- function(n = 300, seed = 1, L = 128) {
  set.seed(seed); lab <- rep(c("cylinder", "bell", "funnel"), length.out = n)
  x <- lapply(lab, function(cl) {
    a <- runif(1, 16, 32); b <- a + runif(1, 32, 96); t <- 1:L; on <- t >= a & t <= b; eta <- rnorm(1)
    base <- switch(cl, cylinder = (6 + eta) * on, bell = (6 + eta) * on * (t - a) / (b - a),
                   funnel = (6 + eta) * on * (b - t) / (b - a))
    base + rnorm(L)
  })
  list(x = x, label = lab)
}
cbf <- make_cbf()
Xc <- flatten_seqs(normalize_seqs(as_seq_list(cbf$x), "z"), 128)
gc <- somgrid(4, 4, "hexagonal", neighbourhood.fct = "gaussian")
Ec <- 30; r0c <- quantile(unit.distances(gc), 2/3); schedc <- r0c - r0c * (0:(Ec - 1)) / Ec
res_cbf <- t(sapply(1:5, function(s) {
  set.seed(s); idx <- sample(length(cbf$x), 16)
  k <- som(Xc, grid = gc, rlen = Ec, radius = c(r0c, 0), mode = "batch", init = list(Xc[idx, ]))
  f <- dtw_som(cbf$x, 4, 4, epochs = Ec, window = 12, neighborhood = "gaussian", nb_cutoff = 0,
               radius = schedc, codes = cbf$x[idx], normalize = "z", smooth = 3, verbose = FALSE)
  c(kohonen = purity(k$unit.classif, cbf$label), dtwsom = purity(f$unit.classif, cbf$label),
    kohonen_TE = somQuality(k, Xc)$err.topo, dtwsom_TE = f$te)
}))
cat("
=== CBF (z-normalised, window 12 = 9% of length), purity and TE per init ===
")
print(round(res_cbf, 3)); cat("mean:
"); print(round(colMeans(res_cbf), 3))

# ============================================================================
# D. a real dataset: UCR GunPoint (as in the DTW-SOM paper)
#    download GunPoint_TRAIN.tsv / GunPoint_TEST.tsv from timeseriesclassification.com
#    (UCR archive) into ./data/ - first column = class, remaining 150 columns = series
# ============================================================================
read_ucr <- function(file) { m <- as.matrix(read.delim(file, header = FALSE)); list(label = m[, 1], x = lapply(seq_len(nrow(m)), function(i) m[i, -1])) }
if (file.exists("data/GunPoint_TRAIN.tsv")) {
  gp <- read_ucr("data/GunPoint_TRAIN.tsv"); if (file.exists("data/GunPoint_TEST.tsv")) { te <- read_ucr("data/GunPoint_TEST.tsv"); gp$x <- c(gp$x, te$x); gp$label <- c(gp$label, te$label) }
  Xg <- flatten_seqs(normalize_seqs(as_seq_list(gp$x), "z"), 150)
  set.seed(1); idx <- sample(length(gp$x), 16)
  kg <- som(Xg, grid = gc, rlen = Ec, radius = c(r0c, 0), mode = "batch", init = list(Xg[idx, ]))
  fg <- dtw_som(gp$x, 4, 4, epochs = Ec, window = 15, neighborhood = "gaussian", nb_cutoff = 0, radius = schedc,
                codes = gp$x[idx], normalize = "z", smooth = 3, verbose = FALSE)
  cat(sprintf("
=== GunPoint (%d series): purity kohonen %.3f   dtwsom %.3f ===
", length(gp$x),
              purity(kg$unit.classif, gp$label), purity(fg$unit.classif, gp$label)))
  show(aweSOMplot(as_kohonen(fg, 150), type = "Line", data = Xg, variables = colnames(Xg), values = "prototypes", showAxes = FALSE), "gunpoint_dtwsom_prototypes")
} else cat("
(Part D skipped: put GunPoint_TRAIN.tsv in ./data/ to run the real-data comparison)
")

if (!interactive()) cat("\nhtml widgets written to ./aweSOM_plots/\n")
