# example.R - DTW-SOM demo on a synthetic motif set (as in the paper) ------
# For full speed install with  CXX17FLAGS = -O3 -march=native  in ~/.R/Makevars

library(dtwsom)      # remotes::install_github("andrii-patrikei/dtwsom")
cat("OpenMP threads available:", dtwsom_threads_cpp(), "\n")

# ---- synthetic motifs: 4 shapes, random length 40-90, noise, amplitude ----
make_motifs <- function(n = 400, seed = 1) {
  set.seed(seed)
  shapes <- c("sine", "triangle", "square", "ramp")
  lab <- sample(shapes, n, replace = TRUE)
  x <- lapply(lab, function(s) {
    L <- sample(40:90, 1); u <- seq(0, 1, length.out = L); a <- runif(1, 0.7, 1.3)
    y <- switch(s, sine = sin(2 * pi * u), triangle = 1 - 4 * abs(u - 0.5),
                square = ifelse(u < 0.5, 1, -1), ramp = 2 * u - 1)
    a * y + rnorm(L, sd = 0.15)
  })
  list(x = x, label = lab)
}
d <- make_motifs()

# ---- batch DTW-SOM (parallel, recommended) --------------------------------
fit <- dtw_som(d$x, xdim = 4, ydim = 4, topo = "hexagonal", epochs = 15,
               mode = "batch", window = 10, seed = 42)
print(fit)
cat("purity:", sum(apply(table(fit$unit.classif, d$label), 1, max)) / length(d$x), "\n")

# ---- the paper's online algorithm (pyclustering neighbourhood, exp decay) --
fit_on <- dtw_som(d$x, 4, 4, topo = "rectangular", epochs = 15, mode = "online",
                  neighborhood = "pyclustering", radius = 2, alpha = 0.1, alpha_decay = "exp",
                  window = 10, seed = 42)
print(fit_on)

# ---- plots ----------------------------------------------------------------
plot(fit, "codes", labels = d$label)
plot(fit, "counts")
plot(fit, "umatrix")
plot(fit, "qe")
plot(fit, "elastic")      # unit positions relaxed to reflect prototype DTW distances (post-hoc "elastic SOM")
plot(fit, "members")      # red prototype + mapped series on their own sample index (lengths differ!)
plot(fit, "members_dtw")  # the same series warped onto the prototype's time axis along their DTW paths

# ---- mapping new series --------------------------------------------------
new <- make_motifs(20, seed = 7)
pr <- predict(fit, new$x)
print(table(bmu = pr$bmu, new$label))

# ---- second-level clustering of the prototypes (as in the Acta Gymnica paper)
D <- dtw_pairwise_cpp(fit$codes, window = 10L, max_step = Inf, threads = 0L)
hc <- hclust(as.dist(D), "ward.D2")
super <- cutree(hc, 4)
print(table(super = super[fit$unit.classif], d$label))

# ---- thread scaling --------------------------------------------------------
if (dtwsom_threads_cpp() > 1) {
  for (th in c(1, dtwsom_threads_cpp()))
    cat(sprintf("threads=%2d  %.2fs\n", th,
        dtw_som(d$x, 5, 5, epochs = 10, threads = th, seed = 1, verbose = FALSE)$elapsed))
}
