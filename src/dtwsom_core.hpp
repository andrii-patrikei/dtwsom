// dtwsom_core.hpp - DTW-SOM core (C++17 + OpenMP, no R dependency)
//
// Implements the DTW-SOM of Silva & Henriques (2020), "Exploring time-series
// motifs through DTW-SOM", arXiv:2004.08176, following their reference Python
// code (github.com/misilva73/dtw_som) for the update rule:
//
//   * BMU by DTW distance (variable-length, multivariate "dependent" DTW,
//     Sakoe-Chiba window, max_step, max_length_diff, early abandoning).
//   * Adaptation through the optimal warping path: for prototype index i,
//     the target value is the mean of all x[j] matched to i.
//       online:  w[i] += alpha * h * (mean_j x[j] - w[i])          (paper)
//       batch:   w[i]  = sum_s h_s * sum_{j~i} x_s[j] / sum_s h_s * |{j~i}|
//                (neighbourhood-weighted DBA barycenter, one pass per epoch)
//   * Per-epoch max_dist pruning = 1.1 * max BMU distance of the previous
//     epoch (paper trick), plus running-best early abandoning.
//
// Prototypes keep the length they had at initialisation (as in the paper).
// The core is R-free so it can be unit-tested with plain g++, bound to R via
// Rcpp (dtwsom_rcpp.cpp), or later to Python via pybind11.
//
// Everything is templated on the number type T: float, double, long double,
// or a Boost.Multiprecision type (cpp_bin_float_quad, cpp_bin_float_50,
// mpfr_float_backend<...>). Grid geometry, schedules and neighbourhood
// weights stay in double; only the DTW arithmetic and the prototypes use T.

#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <atomic>
#include <random>
#include <functional>
#include <string>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace dtwsom {

template <class T> inline T inf() { return std::numeric_limits<T>::infinity(); }
template <class T> inline bool finite_val(const T& x) { return x < inf<T>(); }

inline int n_threads_available() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
inline int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Packed set of variable-length, D-channel sequences (time-major storage).
// value(s, t, d) = values[(off[s] + t) * dim + d]
// ---------------------------------------------------------------------------
template <class T>
struct SeqSet {
    int n = 0;
    int dim = 1;
    std::vector<T> values;
    std::vector<int> off;   // size n + 1

    int len(int s) const { return off[s + 1] - off[s]; }
    const T* ptr(int s) const { return values.data() + (size_t)off[s] * dim; }
    T* ptr(int s) { return values.data() + (size_t)off[s] * dim; }

    static SeqSet empty(int dim) { SeqSet s; s.dim = dim; s.off.push_back(0); return s; }
    void push(const T* v, int len) {
        values.insert(values.end(), v, v + (size_t)len * dim);
        off.push_back(off.back() + len);
        ++n;
    }
};

struct DtwOpts {
    int window = -1;          // Sakoe-Chiba half-width (samples); <0 = unconstrained
    double max_step = std::numeric_limits<double>::infinity(); // max single-step distance (data units)
    int max_length_diff = -1; // <0 = no limit
};

// Per-thread scratch memory (allocated once, reused across DTW calls)
template <class T>
struct Workspace {
    std::vector<T> rows;         // 2 * (m + 1)
    std::vector<uint8_t> dir;    // (n + 1) * (m + 1) backtracking moves
    std::vector<T> sum;          // n * dim accumulators
    std::vector<T> cnt;          // n
};

inline int effective_window(int window, int n, int m) {
    if (window < 0) return std::max(n, m);
    return std::max(window, std::abs(n - m));   // band must contain the end corner
}

template <class T>
inline T local_cost(const T* a, const T* b, int dim) {
    T c = 0;
    for (int d = 0; d < dim; ++d) { T t = a[d] - b[d]; c += t * t; }
    return c;
}

// ---------------------------------------------------------------------------
// DTW distance only (O(m) memory), with early abandoning at max_dist.
// Returns sqrt(sum of squared local costs) or INF if abandoned/infeasible.
// ---------------------------------------------------------------------------
template <class T>
inline T dtw_distance(const T* a, int n, const T* b, int m, int dim,
                      const DtwOpts& o, T max_dist, Workspace<T>& ws)
{
    using std::sqrt;
    const T INF = inf<T>();
    if (n == 0 || m == 0) return INF;
    const int win = effective_window(o.window, n, m);
    const T max_step2 = std::isfinite(o.max_step) ? T(o.max_step) * T(o.max_step) : INF;
    const T max_dist2 = finite_val(max_dist) ? max_dist * max_dist : INF;

    if ((int)ws.rows.size() < 2 * (m + 1)) ws.rows.resize(2 * (m + 1));
    T* prev = ws.rows.data();
    T* cur  = prev + (m + 1);
    std::fill(prev, prev + m + 1, INF);
    prev[0] = 0;

    for (int i = 1; i <= n; ++i) {
        const int jlo = std::max(1, i - win), jhi = std::min(m, i + win);
        cur[jlo - 1] = INF;
        if (jhi < m) cur[jhi + 1] = INF;
        const T* ai = a + (size_t)(i - 1) * dim;
        T rowmin = INF;
        for (int j = jlo; j <= jhi; ++j) {
            const T c = local_cost(ai, b + (size_t)(j - 1) * dim, dim);
            if (c > max_step2) { cur[j] = INF; continue; }
            const T best = std::min(prev[j - 1], std::min(prev[j], cur[j - 1]));
            const T v = c + best;
            cur[j] = v;
            if (v < rowmin) rowmin = v;
        }
        if (rowmin > max_dist2) return INF;   // no path through this row can beat max_dist
        std::swap(prev, cur);
    }
    const T total = prev[m];
    if (total > max_dist2 || !finite_val(total)) return INF;
    return T(sqrt(total));
}

// ---------------------------------------------------------------------------
// DTW with backtracking. For each prototype index i (0..n-1) accumulates
//   out_sum[i*dim + d] += weight * x[j][d]   for every j matched to i
//   out_cnt[i]         += weight
// Returns the DTW distance (INF if infeasible under max_step).
// ---------------------------------------------------------------------------
template <class T>
inline T dtw_path_accumulate(const T* w, int n, const T* x, int m, int dim,
                             const DtwOpts& o, T weight, Workspace<T>& ws,
                             T* out_sum, T* out_cnt)
{
    using std::sqrt;
    const T INF = inf<T>();
    if (n == 0 || m == 0) return INF;
    const int win = effective_window(o.window, n, m);
    const T max_step2 = std::isfinite(o.max_step) ? T(o.max_step) * T(o.max_step) : INF;
    const size_t W = (size_t)m + 1;

    if ((int)ws.rows.size() < 2 * (m + 1)) ws.rows.resize(2 * (m + 1));
    if (ws.dir.size() < (size_t)(n + 1) * W) ws.dir.resize((size_t)(n + 1) * W);
    T* prev = ws.rows.data();
    T* cur  = prev + (m + 1);
    std::fill(prev, prev + m + 1, INF);
    prev[0] = 0;

    for (int i = 1; i <= n; ++i) {
        const int jlo = std::max(1, i - win), jhi = std::min(m, i + win);
        cur[jlo - 1] = INF;
        if (jhi < m) cur[jhi + 1] = INF;
        const T* wi = w + (size_t)(i - 1) * dim;
        uint8_t* di = ws.dir.data() + (size_t)i * W;
        for (int j = jlo; j <= jhi; ++j) {
            const T c = local_cost(wi, x + (size_t)(j - 1) * dim, dim);
            if (c > max_step2) { cur[j] = INF; di[j] = 0; continue; }
            T best = prev[j - 1]; uint8_t mv = 1;                 // diagonal
            if (prev[j]   < best) { best = prev[j];   mv = 2; }   // from (i-1, j)
            if (cur[j - 1] < best) { best = cur[j - 1]; mv = 3; } // from (i, j-1)
            if (!finite_val(best)) { cur[j] = INF; di[j] = 0; continue; }
            cur[j] = c + best; di[j] = mv;
        }
        std::swap(prev, cur);
    }
    const T total = prev[m];
    if (!finite_val(total)) return INF;

    // Backtrack from (n, m); every visited cell was written in this call.
    int i = n, j = m;
    while (i >= 1 && j >= 1) {
        const T* xj = x + (size_t)(j - 1) * dim;
        T* si = out_sum + (size_t)(i - 1) * dim;
        for (int d = 0; d < dim; ++d) si[d] += weight * xj[d];
        out_cnt[i - 1] += weight;
        const uint8_t mv = ws.dir[(size_t)i * W + j];
        if (mv == 1) { --i; --j; }
        else if (mv == 2) { --i; }
        else if (mv == 3) { --j; }
        else break;
    }
    return T(sqrt(total));
}

// ---------------------------------------------------------------------------
// Neighbourhood functions. d = grid distance between units, r = current radius
//   0 gaussian     : exp(-d^2 / (2 r^2))
//   1 bubble       : 1 if d <= r
//   2 pyclustering : the paper's code - exp(-d / (2 r^2)) if d < r^2
// ---------------------------------------------------------------------------
inline double nb_weight(int type, double d, double r) {
    switch (type) {
    case 0: if (r <= 0) return d == 0 ? 1.0 : 0.0;
            return std::exp(-(d * d) / (2.0 * r * r));
    case 1: return d <= r ? 1.0 : 0.0;
    case 2: { const double r2 = r * r;
              if (r2 <= 0) return d == 0 ? 1.0 : 0.0;
              return d < r2 ? std::exp(-d / (2.0 * r2)) : 0.0; }
    }
    return 0.0;
}

struct Params {
    int epochs = 20;
    bool batch = true;
    int nb_type = 0;
    double nb_cutoff = 1e-3;      // skip path computation when h < cutoff
    std::vector<double> radius;   // per epoch
    std::vector<double> alpha;    // per epoch (online only)
    DtwOpts dtw;
    bool prune = true;            // paper's max_dist trick + running-best abandoning
    int smooth = 0;               // odd width of a running mean applied to each prototype after
                                  // every batch update (0/1 = off). DTW aligns noise with noise, so
                                  // DBA-style barycenters keep it; a light smoothing removes it.
    int threads = 0;              // 0 = OpenMP default
    unsigned seed = 1;
};

// running mean of odd width w over a prototype (n x dim, time-major), edges shrink the window
template <class T>
inline void smooth_prototype(T* wk, int n, int dim, int w, std::vector<T>& tmp) {
    if (w < 3 || n < 3) return;
    const int h = w / 2;
    tmp.assign(wk, wk + (size_t)n * dim);
    for (int i = 0; i < n; ++i) {
        const int lo = std::max(0, i - h), hi = std::min(n - 1, i + h);
        for (int d = 0; d < dim; ++d) {
            T acc = 0; for (int j = lo; j <= hi; ++j) acc += tmp[(size_t)j * dim + d];
            wk[(size_t)i * dim + d] = acc / T(hi - lo + 1);
        }
    }
}

template <class T>
struct Result {
    SeqSet<T> codes;
    std::vector<int> bmu, bmu2;   // 0-based unit indices (exact, final pass)
    std::vector<T> dist;          // DTW distance to BMU
    std::vector<T> distmat;       // N * K row-major, exact
    std::vector<double> qe_epoch; // mean BMU distance seen during each epoch
    int epochs_run = 0;
};

using Logger  = std::function<void(const std::string&)>;
using OnEpoch = std::function<bool(int, double)>;   // return false to stop early

// ---------------------------------------------------------------------------
// Sequential BMU search over K prototypes with running-best early abandoning.
// prev (>=0) is tried first: it is usually still the BMU and gives a tight bound.
// ---------------------------------------------------------------------------
template <class T>
inline void find_bmu(const SeqSet<T>& codes, const T* x, int m, const DtwOpts& o,
                     T max_dist, bool prune, int prev, Workspace<T>& ws,
                     int& bmu, T& bdist)
{
    const T INF = inf<T>();
    const int K = codes.n, dim = codes.dim;
    bmu = -1; bdist = INF;
    auto try_k = [&](int k, bool use_thr) {
        if (o.max_length_diff >= 0 && std::abs(codes.len(k) - m) > o.max_length_diff) return;
        const T thr = use_thr ? std::min(max_dist, bdist) : INF;
        const T d = dtw_distance(codes.ptr(k), codes.len(k), x, m, dim, o, thr, ws);
        if (d < bdist) { bdist = d; bmu = k; }
    };
    if (prune && prev >= 0 && prev < K) try_k(prev, true);
    for (int k = 0; k < K; ++k) if (!(prune && k == prev)) try_k(k, prune);
    if (bmu < 0) {                       // everything pruned away -> exact fallback
        for (int k = 0; k < K; ++k) try_k(k, false);
    }
}

// Exact N x K distance matrix + BMU / second BMU (used for the final mapping
// and for predict()).
template <class T>
inline void map_exact(const SeqSet<T>& codes, const SeqSet<T>& data, const DtwOpts& o,
                      int threads, std::vector<T>& distmat,
                      std::vector<int>& bmu, std::vector<int>& bmu2, std::vector<T>& dist)
{
    const T INF = inf<T>();
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif
    const int N = data.n, K = codes.n, dim = codes.dim;
    distmat.assign((size_t)N * K, INF);
    bmu.assign(N, -1); bmu2.assign(N, -1); dist.assign(N, INF);
    std::vector<Workspace<T>> wss(n_threads_available());
#pragma omp parallel for schedule(dynamic, 8)
    for (int s = 0; s < N; ++s) {
        Workspace<T>& ws = wss[thread_id()];
        T b1 = INF, b2 = INF; int i1 = -1, i2 = -1;
        for (int k = 0; k < K; ++k) {
            T d = INF;
            if (o.max_length_diff < 0 || std::abs(codes.len(k) - data.len(s)) <= o.max_length_diff)
                d = dtw_distance(codes.ptr(k), codes.len(k), data.ptr(s), data.len(s), dim, o, INF, ws);
            distmat[(size_t)s * K + k] = d;
            if (d < b1) { b2 = b1; i2 = i1; b1 = d; i1 = k; }
            else if (d < b2) { b2 = d; i2 = k; }
        }
        bmu[s] = i1; bmu2[s] = i2; dist[s] = b1;
    }
}

// ---------------------------------------------------------------------------
// Training. `codes` holds the initial prototypes (random sample / anchors -
// chosen by the caller). `udist` is the K x K matrix of grid distances.
// ---------------------------------------------------------------------------
template <class T>
inline Result<T> train(const SeqSet<T>& data, SeqSet<T> codes, const std::vector<double>& udist,
                       const Params& p, Logger log = nullptr, OnEpoch on_epoch = nullptr)
{
    const T INF = inf<T>();
    const int N = data.n, K = codes.n, dim = data.dim;
    if (codes.dim != dim) throw std::invalid_argument("codes and data must have the same number of channels");
    if ((int)udist.size() != K * K) throw std::invalid_argument("udist must be K x K");
    if ((int)p.radius.size() < p.epochs) throw std::invalid_argument("radius must have one value per epoch");
    if (!p.batch && (int)p.alpha.size() < p.epochs) throw std::invalid_argument("alpha must have one value per epoch");
#ifdef _OPENMP
    if (p.threads > 0) omp_set_num_threads(p.threads);
#endif
    const int NT = n_threads_available();
    std::vector<Workspace<T>> wss(NT);

    Result<T> res;
    std::vector<int> bmu(N, -1);
    std::vector<T> bd(N, INF);
    T max_dist = INF;                             // paper: updated after each epoch
    std::mt19937 rng(p.seed);
    std::vector<int> order(N);
    for (int s = 0; s < N; ++s) order[s] = s;

    for (int e = 0; e < p.epochs; ++e) {
        const double r = p.radius[e];

        if (p.batch) {
            // ---- pass 1: BMUs (parallel over observations) ----------------
#pragma omp parallel for schedule(dynamic, 8)
            for (int s = 0; s < N; ++s) {
                Workspace<T>& ws = wss[thread_id()];
                find_bmu(codes, data.ptr(s), data.len(s), p.dtw, max_dist, p.prune, bmu[s], ws, bmu[s], bd[s]);
            }
            // ---- pass 2: neighbourhood-weighted DBA update per prototype --
            for (int k = 0; k < K; ++k) {
                const int nk = codes.len(k);
                std::vector<std::vector<T>> num(NT), den(NT);
                for (int t = 0; t < NT; ++t) { num[t].assign((size_t)nk * dim, T(0)); den[t].assign(nk, T(0)); }
#pragma omp parallel for schedule(dynamic, 8)
                for (int s = 0; s < N; ++s) {
                    const double h = nb_weight(p.nb_type, udist[(size_t)k * K + bmu[s]], r);
                    if (h < p.nb_cutoff) continue;
                    const int t = thread_id();
                    T d = dtw_path_accumulate(codes.ptr(k), nk, data.ptr(s), data.len(s), dim, p.dtw, T(h),
                                              wss[t], num[t].data(), den[t].data());
                    if (!finite_val(d)) {              // infeasible under max_step: retry unconstrained
                        DtwOpts o2 = p.dtw; o2.max_step = std::numeric_limits<double>::infinity();
                        dtw_path_accumulate(codes.ptr(k), nk, data.ptr(s), data.len(s), dim, o2, T(h),
                                            wss[t], num[t].data(), den[t].data());
                    }
                }
                T* wk = codes.ptr(k);
                bool updated = false;
                for (int i = 0; i < nk; ++i) {
                    T dsum = 0; for (int t = 0; t < NT; ++t) dsum += den[t][i];
                    if (dsum <= 0) continue;           // dead unit for this epoch: keep as is
                    updated = true;
                    for (int d = 0; d < dim; ++d) {
                        T nsum = 0; for (int t = 0; t < NT; ++t) nsum += num[t][(size_t)i * dim + d];
                        wk[(size_t)i * dim + d] = nsum / dsum;
                    }
                }
                if (updated && p.smooth >= 3) smooth_prototype(wk, nk, dim, p.smooth, wss[0].sum);
            }
        } else {
            // ---- online (paper): one observation at a time -----------------
            const double alpha = p.alpha[e];
            std::shuffle(order.begin(), order.end(), rng);
            std::vector<T> dk(K);
            std::vector<int> nbrs; nbrs.reserve(K);
            std::vector<double> hk(K);
            for (int idx = 0; idx < N; ++idx) {
                const int s = order[idx];
                const T* x = data.ptr(s); const int m = data.len(s);

                // competition: parallel over prototypes with a shared running best.
                // The shared bound is a double rounded *up*, so it never prunes a
                // candidate that would beat the true best (exact for any T).
                std::atomic<double> best{std::numeric_limits<double>::infinity()};
                for (int pass = 0; pass < 2; ++pass) {
                    const bool use_thr = p.prune && pass == 0;
#pragma omp parallel for schedule(dynamic)
                    for (int k = 0; k < K; ++k) {
                        T d = INF;
                        if (p.dtw.max_length_diff < 0 || std::abs(codes.len(k) - m) <= p.dtw.max_length_diff) {
                            const T thr = use_thr ? std::min(max_dist, T(best.load(std::memory_order_relaxed))) : INF;
                            d = dtw_distance(codes.ptr(k), codes.len(k), x, m, dim, p.dtw, thr, wss[thread_id()]);
                        }
                        dk[k] = d;
                        if (finite_val(d)) {
                            const double du = std::nextafter(static_cast<double>(d), std::numeric_limits<double>::infinity());
                            double curb = best.load(std::memory_order_relaxed);
                            while (du < curb && !best.compare_exchange_weak(curb, du)) {}
                        }
                    }
                    if (best.load() < std::numeric_limits<double>::infinity()) break;   // else: all pruned -> exact pass
                }
                int b = -1; T bdist = INF;
                for (int k = 0; k < K; ++k) if (dk[k] < bdist) { bdist = dk[k]; b = k; }
                bmu[s] = b; bd[s] = bdist;
                if (b < 0) continue;                    // no feasible prototype (max_length_diff)

                // adaptation: parallel over the prototypes inside the neighbourhood
                nbrs.clear();
                for (int k = 0; k < K; ++k) {
                    hk[k] = nb_weight(p.nb_type, udist[(size_t)k * K + b], r);
                    if (hk[k] >= p.nb_cutoff) nbrs.push_back(k);
                }
#pragma omp parallel for schedule(dynamic)
                for (int q = 0; q < (int)nbrs.size(); ++q) {
                    const int k = nbrs[q];
                    const int nk = codes.len(k);
                    Workspace<T>& ws = wss[thread_id()];
                    ws.sum.assign((size_t)nk * dim, T(0)); ws.cnt.assign(nk, T(0));
                    T d = dtw_path_accumulate(codes.ptr(k), nk, x, m, dim, p.dtw, T(1), ws, ws.sum.data(), ws.cnt.data());
                    if (!finite_val(d)) { DtwOpts o2 = p.dtw; o2.max_step = std::numeric_limits<double>::infinity();
                        dtw_path_accumulate(codes.ptr(k), nk, x, m, dim, o2, T(1), ws, ws.sum.data(), ws.cnt.data()); }
                    T* wk = codes.ptr(k);
                    const T lr = T(alpha * hk[k]);
                    for (int i = 0; i < nk; ++i) {
                        if (ws.cnt[i] <= 0) continue;
                        for (int dd = 0; dd < dim; ++dd) {
                            const T target = ws.sum[(size_t)i * dim + dd] / ws.cnt[i];
                            wk[(size_t)i * dim + dd] += lr * (target - wk[(size_t)i * dim + dd]);
                        }
                    }
                }
            }
        }

        // epoch statistics + paper's max_dist update
        T qe = 0, mx = 0; int cnt = 0;
        for (int s = 0; s < N; ++s) if (finite_val(bd[s])) { qe += bd[s]; if (bd[s] > mx) mx = bd[s]; ++cnt; }
        const double qed = cnt ? static_cast<double>(qe / T(cnt)) : std::numeric_limits<double>::infinity();
        max_dist = (p.prune && cnt) ? mx * T(1.1) : INF;
        res.qe_epoch.push_back(qed);
        res.epochs_run = e + 1;
        if (log) log("epoch " + std::to_string(e + 1) + "/" + std::to_string(p.epochs) +
                     "  radius " + std::to_string(r) + "  mean BMU distance " + std::to_string(qed));
        if (on_epoch && !on_epoch(e + 1, qed)) break;
    }

    map_exact(codes, data, p.dtw, p.threads, res.distmat, res.bmu, res.bmu2, res.dist);
    res.codes = std::move(codes);
    return res;
}

// Pairwise / cross DTW distance matrices (parallel), handy for U-matrix and
// second-level clustering of the prototypes.
template <class T>
inline std::vector<T> dtw_cross(const SeqSet<T>& a, const SeqSet<T>& b, const DtwOpts& o, int threads)
{
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif
    if (a.dim != b.dim) throw std::invalid_argument("channel mismatch");
    const int A = a.n, B = b.n;
    std::vector<T> out((size_t)A * B, inf<T>());
    std::vector<Workspace<T>> wss(n_threads_available());
#pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < A; ++i) {
        Workspace<T>& ws = wss[thread_id()];
        for (int j = 0; j < B; ++j)
            out[(size_t)i * B + j] = dtw_distance(a.ptr(i), a.len(i), b.ptr(j), b.len(j), a.dim, o, inf<T>(), ws);
    }
    return out;
}

template <class T>
inline std::vector<T> dtw_pairwise(const SeqSet<T>& a, const DtwOpts& o, int threads)
{
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif
    const int A = a.n;
    std::vector<T> out((size_t)A * A, T(0));
    std::vector<Workspace<T>> wss(n_threads_available());
    // flatten the upper triangle so the parallel loop is balanced
    std::vector<std::pair<int,int>> pairs; pairs.reserve((size_t)A * (A - 1) / 2);
    for (int i = 0; i < A; ++i) for (int j = i + 1; j < A; ++j) pairs.emplace_back(i, j);
    const long P = (long)pairs.size();
#pragma omp parallel for schedule(dynamic, 16)
    for (long q = 0; q < P; ++q) {
        const int i = pairs[q].first, j = pairs[q].second;
        const T d = dtw_distance(a.ptr(i), a.len(i), a.ptr(j), a.len(j), a.dim, o, inf<T>(), wss[thread_id()]);
        out[(size_t)i * A + j] = d; out[(size_t)j * A + i] = d;
    }
    return out;
}

} // namespace dtwsom
