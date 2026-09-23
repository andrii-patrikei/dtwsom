# dtwsom.R - R interface to the DTW-SOM core (see src/dtwsom_core.hpp)
#
# Input `x` can be: a list of numeric vectors / T x D matrices (variable
# length), a numeric matrix (one series per row, NA-padded rows allowed),
# or a 3-D array N x T x D.

# ---- helpers ---------------------------------------------------------------

as_seq_list <- function(x) {
  if (is.list(x)) {
    lapply(x, function(s) if (is.matrix(s)) s else matrix(as.numeric(s), ncol = 1))
  } else if (is.array(x) && length(dim(x)) == 3) {
    lapply(seq_len(dim(x)[1]), function(i) {
      m <- x[i, , , drop = FALSE]; m <- matrix(m, nrow = dim(x)[2])
      m[rowSums(is.na(m)) == 0, , drop = FALSE]
    })
  } else if (is.matrix(x)) {
    lapply(seq_len(nrow(x)), function(i) { v <- x[i, ]; matrix(v[!is.na(v)], ncol = 1) })
  } else if (is.numeric(x)) {
    list(matrix(x, ncol = 1))
  } else stop("x must be a list of series, a matrix (rows = series) or an N x T x D array")
}

normalize_seqs <- function(seqs, how = c("none", "z", "minmax")) {
  how <- match.arg(how)
  if (how == "none") return(seqs)
  lapply(seqs, function(m) {
    if (how == "z") { s <- apply(m, 2, sd); s[s == 0] <- 1; sweep(sweep(m, 2, colMeans(m)), 2, s, "/") }
    else { r <- apply(m, 2, range); d <- r[2, ] - r[1, ]; d[d == 0] <- 1; sweep(sweep(m, 2, r[1, ]), 2, d, "/") }
  })
}

make_grid <- function(xdim, ydim, topo = c("hexagonal", "rectangular")) {
  topo <- match.arg(topo)
  pts <- expand.grid(x = seq_len(xdim), y = seq_len(ydim))      # unit k = x + (y-1)*xdim
  if (topo == "hexagonal") { pts$x <- pts$x + 0.5 * (pts$y %% 2); pts$y <- pts$y * sqrt(3) / 2 }   # same convention as kohonen::somgrid
  pts <- as.matrix(pts)
  list(pts = pts, xdim = xdim, ydim = ydim, topo = topo, udist = as.matrix(dist(pts)))
}

# Anchor initialisation exactly as in the reference Python code: anchors go on
# the two diagonals of the largest square inside the grid, the remaining units
# get leftover anchors and a random sample of the data.
anchor_init <- function(seqs, anchors, xdim, ydim) {
  rows <- ydim; cols <- xdim; K <- rows * cols
  anchors <- as_seq_list(anchors)
  n_anch <- length(anchors)
  if (n_anch == 0) stop("anchor init needs at least one anchor")
  max_sq <- min(rows, cols); n_diag <- 2 * max_sq - 1
  if (n_anch < K) {
    sample_idx <- sample(length(seqs), K - n_anch)
    extra <- seqs[sample_idx]
    if (n_anch < n_diag) { diag_l <- c(anchors, extra[seq_len(n_diag - n_anch)]); others <- extra[-seq_len(n_diag - n_anch)] }
    else { diag_l <- anchors[seq_len(n_diag)]; others <- c(anchors[-seq_len(n_diag)], extra) }
  } else {
    if (n_anch > K) warning("more anchors than units; only the first ", K, " are used")
    diag_l <- anchors[seq_len(n_diag)]; others <- anchors[(n_diag + 1):K]
  }
  diag_l <- diag_l[sample(length(diag_l))]; others <- others[sample(length(others))]
  codes <- vector("list", K); o <- 1
  for (i in 0:(rows - 1)) for (j in 0:(cols - 1)) {
    k <- i * cols + j + 1
    if (i == j && (2 * j != max_sq - 1)) codes[[k]] <- diag_l[[i + 1]]
    else if (i == max_sq - 1 - j) codes[[k]] <- diag_l[[i + max_sq]]
    else { codes[[k]] <- others[[o]]; o <- o + 1 }
  }
  codes
}

# ---- main ------------------------------------------------------------------

dtw_som <- function(x, xdim = 5, ydim = 5, topo = c("hexagonal", "rectangular"),
                    epochs = 20, mode = c("batch", "online"),
                    init = c("sample", "anchor"), anchors = NULL, codes = NULL,
                    window = NULL, max_step = NULL, max_length_diff = NULL,
                    radius = NULL, radius_decay = c("linear", "exp"),
                    alpha = c(0.05, 0.01), alpha_decay = c("linear", "exp"),
                    neighborhood = c("gaussian", "bubble", "pyclustering"), nb_cutoff = 1e-3,
                    normalize = c("none", "z", "minmax"),
                    precision = c("double", "float", "longdouble", "quad", "bin50", "mpfr100"),
                    smooth = 0, prune = TRUE, threads = 0, seed = NULL, verbose = TRUE)
{
  precision <- match.arg(precision)
  topo <- match.arg(topo); mode <- match.arg(mode); init <- match.arg(init)
  radius_decay <- match.arg(radius_decay); alpha_decay <- match.arg(alpha_decay)
  neighborhood <- match.arg(neighborhood); normalize <- match.arg(normalize)
  if (!is.null(seed)) set.seed(seed)

  seqs <- normalize_seqs(as_seq_list(x), normalize)
  N <- length(seqs); K <- xdim * ydim
  grid <- make_grid(xdim, ydim, topo)

  # initial prototypes
  if (is.null(codes)) {
    if (init == "sample") {
      if (N < K) stop("need at least K = ", K, " sequences for random-sample init")
      codes <- seqs[sample(N, K)]
    } else codes <- anchor_init(seqs, normalize_seqs(as_seq_list(anchors), normalize), xdim, ydim)
  } else codes <- as_seq_list(codes)
  stopifnot(length(codes) == K)

  # schedules (one value per epoch)
  nb_type <- match(neighborhood, c("gaussian", "bubble", "pyclustering")) - 1L
  if (is.null(radius)) {
    r0 <- unname(quantile(grid$udist[grid$udist > 0], 2 / 3))
    radius <- if (neighborhood == "gaussian") c(r0, 0.5) else if (neighborhood == "bubble") c(r0, 0) else r0
  }
  ep <- seq_len(epochs)
  if (neighborhood == "pyclustering") {          # paper: r_e = r0 * exp(-e/E), squared inside
    radius_sched <- radius[1] * exp(-ep / epochs)
  } else if (length(radius) == epochs && epochs > 2) {   # explicit per-epoch schedule
    radius_sched <- radius
  } else if (length(radius) == 1) {
    radius_sched <- rep(radius, epochs)
  } else if (radius_decay == "linear") {
    radius_sched <- seq(radius[1], radius[2], length.out = epochs)
  } else radius_sched <- radius[1] * (radius[2] / radius[1])^((ep - 1) / max(1, epochs - 1))
  if (length(alpha) == 1) alpha_sched <- rep(alpha, epochs)
  else if (alpha_decay == "linear") alpha_sched <- seq(alpha[1], alpha[2], length.out = epochs)
  else alpha_sched <- alpha[1] * (alpha[2] / alpha[1])^((ep - 1) / max(1, epochs - 1))

  params <- list(epochs = as.integer(epochs), batch = mode == "batch",
                 nb_type = nb_type, nb_cutoff = as.numeric(nb_cutoff),   # 0 = use every h > 0 like kohonen
                 radius = as.numeric(radius_sched), alpha = as.numeric(alpha_sched),
                 window = if (is.null(window)) -1L else as.integer(window),
                 max_step = if (is.null(max_step)) Inf else as.numeric(max_step),
                 max_length_diff = if (is.null(max_length_diff)) -1L else as.integer(max_length_diff),
                 prune = isTRUE(prune), smooth = as.integer(smooth), threads = as.integer(threads),
                 seed = as.integer(if (is.null(seed)) sample.int(.Machine$integer.max, 1) else seed),
                 verbose = isTRUE(verbose))

  t0 <- proc.time()[["elapsed"]]
  res <- dtwsom_train_cpp(seqs, codes, grid$udist, params, precision)
  elapsed <- proc.time()[["elapsed"]] - t0

  adj <- grid$udist < 1.5 & grid$udist > 0                  # neighbouring units
  te <- mean(!adj[cbind(res$bmu, res$bmu2)])
  structure(list(codes = res$codes, unit.classif = res$bmu, unit.classif2 = res$bmu2,
                 distances = res$distances, distmat = res$distmat,
                 qe = mean(res$distances), te = te, qe_epoch = res$qe_epoch,
                 counts = tabulate(res$bmu, K), grid = grid,
                 data = seqs, params = params, normalize = normalize, precision = precision,
                 elapsed = elapsed, call = match.call()),
            class = "dtwsom")
}

print.dtwsom <- function(x, ...) {
  cat(sprintf("DTW-SOM  %dx%d %s grid, %d series, %d channels, %s mode, %s precision\n",
              x$grid$xdim, x$grid$ydim, x$grid$topo, length(x$data), ncol(x$data[[1]]),
              if (x$params$batch) "batch" else "online", x$precision))
  cat(sprintf("  epochs %d   quantization error %.4f   topographic error %.3f   %.1fs\n",
              x$params$epochs, x$qe, x$te, x$elapsed))
  cat("  hits per unit:", x$counts, "\n")
  invisible(x)
}

predict.dtwsom <- function(object, newdata, threads = 0, precision = object$precision, ...) {
  seqs <- normalize_seqs(as_seq_list(newdata), object$normalize)
  p <- object$params
  dtwsom_map_cpp(object$codes, seqs, p$window, p$max_step, p$max_length_diff, threads, precision)
}

# DTW distance matrix between the prototypes (or any list of series)
dtw_pairwise <- function(x, window = NULL, max_step = NULL, threads = 0, precision = "double") {
  dtw_pairwise_cpp(as_seq_list(x), if (is.null(window)) -1L else as.integer(window),
                   if (is.null(max_step)) Inf else max_step, as.integer(threads), precision)
}

# U-matrix: mean DTW distance from each unit's prototype to its neighbours
umatrix <- function(obj, threads = 0) {
  p <- obj$params
  D <- dtw_pairwise_cpp(obj$codes, p$window, p$max_step, threads, obj$precision)
  adj <- obj$grid$udist < 1.5 & obj$grid$udist > 0
  sapply(seq_len(nrow(D)), function(k) mean(D[k, adj[k, ]]))
}

# "Elastic" layout: let the unit positions move so that their 2-D distances
# reflect the DTW distances between prototypes (Sammon stress, started from
# the grid so the map keeps its orientation). This is the post-hoc version of
# the idea in Hartono & Take (2017) "Pairwise elastic SOM" and of Merkl &
# Rauber's adaptive coordinates; training is untouched.
elastic_layout <- function(obj, threads = 0, niter = 200) {
  if (!requireNamespace("MASS", quietly = TRUE)) stop("install.packages('MASS')")
  p <- obj$params
  D <- dtw_pairwise_cpp(obj$codes, p$window, p$max_step, threads, obj$precision)
  D[D == 0 & row(D) != col(D)] <- 1e-8 * max(D)     # sammon() refuses zero distances
  y <- MASS::sammon(as.dist(D), y = obj$grid$pts, niter = niter, trace = FALSE)$points
  colnames(y) <- c("x", "y"); y
}

# ---- kohonen / aweSOM interoperability --------------------------------------

# Resample every series to `len` points per channel (linear interpolation) and
# flatten to one row: [channel-1 values, channel-2 values, ...].
flatten_seqs <- function(seqs, len) {
  t(sapply(seqs, function(m) {
    if (nrow(m) == len) as.vector(m)
    else as.vector(apply(m, 2, function(v) approx(seq(0, 1, length.out = length(v)), v, seq(0, 1, length.out = len))$y))
  }))
}

# Wrap a dtwsom fit as a `kohonen` object so kohonen::plot, aweSOM::aweSOMplot,
# aweSOM::somQuality etc. accept it. Variable-length series are resampled to a
# common length (default: median length). aweSOM's quality measures are
# Euclidean on the flattened vectors - identical to ours when window = 0 on
# fixed-length data (iris), an approximation for real DTW fits; use fit$qe / fit$te
# for the DTW-consistent values.
as_kohonen <- function(fit, len = NULL) {
  if (!requireNamespace("kohonen", quietly = TRUE)) stop("install.packages('kohonen')")
  lens <- vapply(fit$data, nrow, 1L); D <- ncol(fit$data[[1]])
  if (is.null(len)) len <- if (length(unique(c(lens, vapply(fit$codes, nrow, 1L)))) == 1) lens[1] else round(median(lens))
  X <- flatten_seqs(fit$data, len); C <- flatten_seqs(fit$codes, len)
  colnames(X) <- colnames(C) <- if (D == 1) paste0("t", seq_len(len)) else paste0("ch", rep(seq_len(D), each = len), ".t", seq_len(len))
  g <- kohonen::somgrid(fit$grid$xdim, fit$grid$ydim, fit$grid$topo,
                        neighbourhood.fct = if (fit$params$nb_type == 1) "bubble" else "gaussian")
  structure(list(data = list(X), unit.classif = fit$unit.classif, distances = fit$distances,
                 grid = g, codes = list(C), changes = matrix(fit$qe_epoch, ncol = 1),
                 alpha = range(fit$params$alpha), radius = range(fit$params$radius),
                 na.rows = integer(0), user.weights = 1, distance.weights = 1, whatmap = 1,
                 maxNA.fraction = 0, dist.fcts = "sumofsquares", dtwsom = TRUE),
            class = "kohonen")
}

# Warp one series onto a prototype's time axis along the optimal DTW path
dtw_warp <- function(proto, x, window = NULL, max_step = NULL) {
  dtw_warp_cpp(as_seq_list(list(proto))[[1]], as_seq_list(list(x))[[1]],
               if (is.null(window)) -1L else as.integer(window), if (is.null(max_step)) Inf else max_step)
}

# Prototype (red) with the series mapped to it (semi-transparent), one panel per
# unit, laid out like the map (top row of the grid on top).
#   align = "index": each member on its own sample index -> different lengths visible
#   align = "dtw":   each member warped onto the prototype's time axis along its DTW path
plot_members <- function(fit, align = c("index", "dtw"), channel = 1, max_members = 60,
                         alpha = 0.25, col_members = "steelblue", col_proto = "red", main = NULL) {
  align <- match.arg(align); g <- fit$grid; K <- g$xdim * g$ydim; p <- fit$params
  op <- par(mfrow = c(g$ydim, g$xdim), mar = c(0.6, 0.6, 1.3, 0.4), oma = c(0, 0, 2.2, 0)); on.exit(par(op))
  yl <- range(unlist(lapply(fit$data, function(m) m[, channel])), unlist(lapply(fit$codes, function(m) m[, channel])))
  for (row in g$ydim:1) for (col in seq_len(g$xdim)) {
    k <- col + (row - 1) * g$xdim
    proto <- fit$codes[[k]][, channel]
    mem <- which(fit$unit.classif == k)
    if (length(mem) > max_members) mem <- sample(mem, max_members)
    series <- lapply(mem, function(s) if (align == "dtw")
      dtw_warp(fit$codes[[k]], fit$data[[s]], p$window, p$max_step)[, channel] else fit$data[[s]][, channel])
    xl <- c(1, max(length(proto), vapply(series, length, 1L), 2))
    plot(NA, xlim = xl, ylim = yl, axes = FALSE, xlab = "", ylab = "", main = "")
    box(col = "grey80")
    for (v in series) lines(v, col = adjustcolor(col_members, alpha))
    lines(proto, col = col_proto, lwd = 2)
    title(sprintf("unit %d  n=%d", k, fit$counts[k]), cex.main = 0.75, line = 0.3)
  }
  mtext(if (is.null(main)) sprintf("DTW-SOM: prototypes (red) and mapped series%s",
        if (align == "dtw") ", DTW-aligned to the prototype" else " on their own sample index (lengths differ)") else main,
        outer = TRUE, cex = 1)
}

# Same overlay for a kohonen object: every series had to be resampled to one length.
plot_members_kohonen <- function(koh, channel_cols = NULL, max_members = 60, alpha = 0.25,
                                 col_members = "steelblue", col_proto = "red", main = NULL) {
  g <- koh$grid; K <- nrow(g$pts); X <- koh$data[[1]]; C <- koh$codes[[1]]
  if (!is.null(channel_cols)) { X <- X[, channel_cols, drop = FALSE]; C <- C[, channel_cols, drop = FALSE] }
  op <- par(mfrow = c(g$ydim, g$xdim), mar = c(0.6, 0.6, 1.3, 0.4), oma = c(0, 0, 2.2, 0)); on.exit(par(op))
  yl <- range(X, C); cnt <- tabulate(koh$unit.classif, K)
  for (row in g$ydim:1) for (col in seq_len(g$xdim)) {
    k <- col + (row - 1) * g$xdim
    mem <- which(koh$unit.classif == k); if (length(mem) > max_members) mem <- sample(mem, max_members)
    plot(NA, xlim = c(1, ncol(X)), ylim = yl, axes = FALSE, xlab = "", ylab = "", main = ""); box(col = "grey80")
    for (s in mem) lines(X[s, ], col = adjustcolor(col_members, alpha))
    lines(C[k, ], col = col_proto, lwd = 2)
    title(sprintf("unit %d  n=%d", k, cnt[k]), cex.main = 0.75, line = 0.3)
  }
  mtext(if (is.null(main)) sprintf("kohonen: prototypes (red) and mapped series, all resampled to %d points", ncol(X)) else main,
        outer = TRUE, cex = 1)
}

# ---- plotting (base graphics) ----------------------------------------------

plot.dtwsom <- function(x, type = c("codes", "counts", "umatrix", "qe", "elastic", "members", "members_dtw"), channel = 1,
                        labels = NULL, main = NULL, ...) {
  type <- match.arg(type)
  g <- x$grid; K <- g$xdim * g$ydim
  if (type == "members")     return(invisible(plot_members(x, "index", channel = channel, main = main, ...)))
  if (type == "members_dtw") return(invisible(plot_members(x, "dtw",   channel = channel, main = main, ...)))
  if (type == "elastic") {
    y <- elastic_layout(x)
    adj <- which(g$udist < 1.5 & g$udist > 0 & upper.tri(g$udist), arr.ind = TRUE)
    plot(y, type = "n", asp = 1, axes = FALSE, xlab = "", ylab = "",
         main = if (is.null(main)) "elastic layout (Sammon on prototype DTW distances)" else main)
    segments(y[adj[, 1], 1], y[adj[, 1], 2], y[adj[, 2], 1], y[adj[, 2], 2], col = "grey75")
    points(y, pch = 21, bg = "steelblue", cex = 1 + 3 * sqrt(x$counts / max(x$counts)))
    text(y, labels = seq_len(K), pos = 3, cex = 0.7)
    return(invisible(y))
  }
  if (type == "qe") {
    plot(x$qe_epoch, type = "b", pch = 19, xlab = "epoch", ylab = "mean BMU distance",
         main = if (is.null(main)) "training quantization error" else main); return(invisible())
  }
  if (type == "codes") {
    op <- par(mfrow = c(g$ydim, g$xdim), mar = c(0.5, 0.5, 1.2, 0.5), oma = c(0, 0, 2, 0)); on.exit(par(op))
    yl <- range(unlist(lapply(x$codes, function(m) m[, channel])))
    for (row in g$ydim:1) for (col in seq_len(g$xdim)) {      # top row printed first
      k <- col + (row - 1) * g$xdim
      plot(x$codes[[k]][, channel], type = "l", lwd = 2, col = "steelblue", ylim = yl,
           axes = FALSE, xlab = "", ylab = "", main = "")
      box(col = "grey70")
      ttl <- sprintf("unit %d  n=%d", k, x$counts[k])
      if (!is.null(labels)) { tb <- table(labels[x$unit.classif == k]); if (length(tb)) ttl <- paste0(ttl, "  ", names(tb)[which.max(tb)]) }
      title(ttl, cex.main = 0.75)
    }
    mtext(if (is.null(main)) "prototypes (codebook series)" else main, outer = TRUE, cex = 1.1)
    return(invisible())
  }
  val <- if (type == "counts") x$counts else umatrix(x)
  pal <- hcl.colors(20, if (type == "counts") "YlGnBu" else "Inferno", rev = type == "umatrix")
  bin <- cut(val, 20, labels = FALSE, include.lowest = TRUE)
  cols <- pal[bin]; tcol <- ifelse(bin > 12, "white", "black")
  plot(g$pts, type = "n", asp = 1, axes = FALSE, xlab = "", ylab = "",
       main = if (is.null(main)) type else main)
  r <- 0.5
  if (g$topo == "hexagonal") {
    a <- seq(30, 390, by = 60) * pi / 180
    for (k in seq_len(K)) polygon(g$pts[k, 1] + r / cos(pi / 6) * cos(a), g$pts[k, 2] + r / cos(pi / 6) * sin(a), col = cols[k], border = "white")
  } else for (k in seq_len(K)) rect(g$pts[k, 1] - r, g$pts[k, 2] - r, g$pts[k, 1] + r, g$pts[k, 2] + r, col = cols[k], border = "white")
  text(g$pts, labels = if (type == "counts") val else sprintf("%.2f", val), cex = 0.7, col = tcol)
  invisible(val)
}
