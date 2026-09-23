// dtw_lanes.cpp - demo: DTW of ONE query against L prototypes in lockstep so
// the compiler can vectorise across prototypes (AVX-512: 8 doubles / 16 floats).
// No intrinsics: the inner loop over lanes has no dependency, `#pragma omp simd`
// plus -O3 -march=native does the rest. Prototypes must share one length here
// (pad or bucket by length in a real implementation) and no band is applied.
//
//   Rcpp::sourceCpp("src/dtw_lanes.cpp")
//   P is an L x n matrix: row = prototype (lane), column = time step, so the
//   lanes of one time step are contiguous in memory (column-major R storage).

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
using namespace Rcpp;

template <class T>
static void dtw_lanes_kernel(const T* P, int L, int n, const T* q, int m, T* out, std::vector<T>& buf)
{
    const T INF = std::numeric_limits<T>::infinity();
    buf.assign(2 * (size_t)(m + 1) * L, INF);
    T* prev = buf.data();
    T* cur  = prev + (size_t)(m + 1) * L;
    for (int l = 0; l < L; ++l) prev[l] = 0;                 // cell (0,0) of every lane
    for (int i = 1; i <= n; ++i) {
        const T* pi = P + (size_t)(i - 1) * L;                // lane values of time step i
        for (int l = 0; l < L; ++l) cur[l] = INF;            // column 0
        for (int j = 1; j <= m; ++j) {
            const T qj = q[j - 1];
            T* c = cur + (size_t)j * L;
            const T* pd = prev + (size_t)(j - 1) * L;
            const T* pu = prev + (size_t)j * L;
            const T* pl = cur  + (size_t)(j - 1) * L;
#pragma omp simd
            for (int l = 0; l < L; ++l) {                     // <- vectorised across prototypes
                const T t = pi[l] - qj;
                const T best = std::min(pd[l], std::min(pu[l], pl[l]));
                c[l] = t * t + best;
            }
        }
        std::swap(prev, cur);
    }
    for (int l = 0; l < L; ++l) out[l] = std::sqrt(prev[(size_t)m * L + l]);
}

// [[Rcpp::export]]
NumericVector dtw_lanes_cpp(NumericMatrix P, NumericVector q, std::string precision = "double")
{
    const int L = P.nrow(), n = P.ncol(), m = q.size();
    NumericVector out(L);
    if (precision == "double") {
        std::vector<double> buf, pv(P.begin(), P.end());
        dtw_lanes_kernel<double>(pv.data(), L, n, q.begin(), m, out.begin(), buf);
    } else if (precision == "float") {
        std::vector<float> buf, pv(P.begin(), P.end()), qv(q.begin(), q.end()), o(L);
        dtw_lanes_kernel<float>(pv.data(), L, n, qv.data(), m, o.data(), buf);
        for (int l = 0; l < L; ++l) out[l] = o[l];
    } else stop("precision must be 'double' or 'float'");
    return out;
}

// Same arithmetic, one prototype at a time (the loop over j carries a
// dependency through cur[j-1], so this cannot vectorise) - the reference.
// [[Rcpp::export]]
NumericVector dtw_scalar_cpp(NumericMatrix P, NumericVector q)
{
    const int L = P.nrow(), n = P.ncol(), m = q.size();
    const double INF = std::numeric_limits<double>::infinity();
    NumericVector out(L);
    std::vector<double> prevv(m + 1), curv(m + 1);
    for (int l = 0; l < L; ++l) {
        double* prev = prevv.data(); double* cur = curv.data();
        std::fill(prev, prev + m + 1, INF); prev[0] = 0;
        for (int i = 1; i <= n; ++i) {
            const double a = P(l, i - 1);
            cur[0] = INF;
            for (int j = 1; j <= m; ++j) {
                const double t = a - q[j - 1];
                cur[j] = t * t + std::min(prev[j - 1], std::min(prev[j], cur[j - 1]));
            }
            std::swap(prev, cur);
        }
        out[l] = std::sqrt(prev[m]);
    }
    return out;
}
