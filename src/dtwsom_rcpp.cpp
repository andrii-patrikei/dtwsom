// dtwsom_rcpp.cpp - thin Rcpp layer over dtwsom_core.hpp
//
// Quick use:   Rcpp::sourceCpp("src/dtwsom_rcpp.cpp"); source("R/dtwsom.R")
// (put -O3 -march=native in ~/.R/Makevars as CXX17FLAGS for full speed)
//
// Number type is chosen at run time with `precision`:
//   "float"     32-bit                         "double"  64-bit (default)
//   "longdouble" x87 80-bit on x86 gcc          "quad"    IEEE binary128 (Boost, software)
//   "bin50"     50 decimal digits (Boost)       "mpfr100" 100 digits via MPFR, only if compiled
//                                                         with -DDTWSOM_WITH_MPFR and -lmpfr -lgmp
// Boost headers come from the BH package (install.packages("BH")).

#include <Rcpp.h>
#include <boost/multiprecision/cpp_bin_float.hpp>
#ifdef DTWSOM_WITH_MPFR
#include <boost/multiprecision/mpfr.hpp>
#endif
#include "dtwsom_core.hpp"

using namespace Rcpp;
namespace mp = boost::multiprecision;
using qfloat_t  = mp::number<mp::cpp_bin_float<113, mp::digit_base_2>, mp::et_off>;
using bfloat50_t = mp::number<mp::cpp_bin_float<50>, mp::et_off>;
#ifdef DTWSOM_WITH_MPFR
using mpfr100_t = mp::number<mp::mpfr_float_backend<100>, mp::et_off>;
#endif

// ---- conversions ----------------------------------------------------------
template <class T>
static dtwsom::SeqSet<T> as_seqset(List seqs, int dim_expected = -1)
{
    const int n = seqs.size();
    if (n == 0) stop("empty sequence list");
    int dim = -1;
    dtwsom::SeqSet<T> set;
    std::vector<T> buf;
    for (int s = 0; s < n; ++s) {
        RObject el = seqs[s];
        int Tn, D;
        NumericMatrix M;
        if (Rf_isMatrix(el)) { M = as<NumericMatrix>(el); Tn = M.nrow(); D = M.ncol(); }
        else { NumericVector v = as<NumericVector>(el); M = NumericMatrix(v.size(), 1, v.begin()); Tn = v.size(); D = 1; }
        if (dim < 0) { dim = D; set = dtwsom::SeqSet<T>::empty(dim); set.values.reserve((size_t)n * Tn * D); }
        if (D != dim) stop("sequence %d has %d channels, expected %d", s + 1, D, dim);
        if (Tn == 0) stop("sequence %d has length 0", s + 1);
        buf.assign((size_t)Tn * D, T(0));
        for (int t = 0; t < Tn; ++t)
            for (int d = 0; d < D; ++d) {
                const double v = M(t, d);
                if (!R_finite(v)) stop("sequence %d contains NA/NaN/Inf", s + 1);
                buf[(size_t)t * D + d] = T(v);
            }
        set.push(buf.data(), Tn);
    }
    if (dim_expected > 0 && dim != dim_expected) stop("expected %d channels, got %d", dim_expected, dim);
    return set;
}

template <class T>
static List seqset_to_list(const dtwsom::SeqSet<T>& set)
{
    List out(set.n);
    for (int s = 0; s < set.n; ++s) {
        const int Tn = set.len(s), D = set.dim;
        NumericMatrix M(Tn, D);
        const T* p = set.ptr(s);
        for (int t = 0; t < Tn; ++t) for (int d = 0; d < D; ++d) M(t, d) = static_cast<double>(p[(size_t)t * D + d]);
        out[s] = M;
    }
    return out;
}

template <class T>
static NumericMatrix to_matrix(const std::vector<T>& v, int nrow, int ncol)   // v is row-major
{
    NumericMatrix M(nrow, ncol);
    for (int i = 0; i < nrow; ++i) for (int j = 0; j < ncol; ++j) M(i, j) = static_cast<double>(v[(size_t)i * ncol + j]);
    return M;
}
template <class T>
static NumericVector to_vector(const std::vector<T>& v)
{
    NumericVector out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(v[i]);
    return out;
}

static dtwsom::DtwOpts dtw_opts(int window, double max_step, int max_length_diff)
{
    dtwsom::DtwOpts o;
    o.window = window;
    o.max_step = R_finite(max_step) ? max_step : std::numeric_limits<double>::infinity();
    o.max_length_diff = max_length_diff;
    return o;
}

// ---- precision dispatch ---------------------------------------------------
template <template <class> class F, class... A>
static auto dispatch(const std::string& precision, A&&... a)
{
    if (precision == "double")     return F<double>::run(std::forward<A>(a)...);
    if (precision == "float")      return F<float>::run(std::forward<A>(a)...);
    if (precision == "longdouble") return F<long double>::run(std::forward<A>(a)...);
    if (precision == "quad")       return F<qfloat_t>::run(std::forward<A>(a)...);
    if (precision == "bin50")      return F<bfloat50_t>::run(std::forward<A>(a)...);
#ifdef DTWSOM_WITH_MPFR
    if (precision == "mpfr100")    return F<mpfr100_t>::run(std::forward<A>(a)...);
#else
    if (precision == "mpfr100")    stop("compiled without MPFR: rebuild with -DDTWSOM_WITH_MPFR and PKG_LIBS='-lmpfr -lgmp'");
#endif
    stop("unknown precision '%s'", precision.c_str());
}

// ---- train ----------------------------------------------------------------
template <class T> struct TrainF {
    static List run(List data, List codes, NumericMatrix udist, List params)
    {
        dtwsom::SeqSet<T> X = as_seqset<T>(data);
        dtwsom::SeqSet<T> C = as_seqset<T>(codes, X.dim);
        const int K = C.n;
        if (udist.nrow() != K || udist.ncol() != K) stop("udist must be K x K");
        std::vector<double> ud((size_t)K * K);
        for (int i = 0; i < K; ++i) for (int j = 0; j < K; ++j) ud[(size_t)i * K + j] = udist(i, j);

        dtwsom::Params p;
        p.epochs    = as<int>(params["epochs"]);
        p.batch     = as<bool>(params["batch"]);
        p.nb_type   = as<int>(params["nb_type"]);
        p.nb_cutoff = as<double>(params["nb_cutoff"]);
        p.radius    = as<std::vector<double>>(params["radius"]);
        p.alpha     = as<std::vector<double>>(params["alpha"]);
        p.dtw       = dtw_opts(as<int>(params["window"]), as<double>(params["max_step"]), as<int>(params["max_length_diff"]));
        p.prune     = as<bool>(params["prune"]);
        p.smooth    = params.containsElementNamed("smooth") ? as<int>(params["smooth"]) : 0;
        p.threads   = as<int>(params["threads"]);
        p.seed      = as<unsigned>(params["seed"]);
        const bool verbose = as<bool>(params["verbose"]);

        dtwsom::Logger log = [verbose](const std::string& s) { if (verbose) Rcout << s << "\n"; };
        dtwsom::OnEpoch on_epoch = [](int, double) { Rcpp::checkUserInterrupt(); return true; };

        dtwsom::Result<T> r = dtwsom::train<T>(X, std::move(C), ud, p, log, on_epoch);

        IntegerVector b1(r.bmu.begin(), r.bmu.end()), b2(r.bmu2.begin(), r.bmu2.end());
        return List::create(_["codes"] = seqset_to_list(r.codes),
                            _["bmu"] = b1 + 1, _["bmu2"] = b2 + 1,          // 1-based for R
                            _["distances"] = to_vector(r.dist),
                            _["distmat"] = to_matrix(r.distmat, X.n, K),
                            _["qe_epoch"] = NumericVector(r.qe_epoch.begin(), r.qe_epoch.end()),
                            _["epochs_run"] = r.epochs_run);
    }
};

// [[Rcpp::export]]
List dtwsom_train_cpp(List data, List codes, NumericMatrix udist, List params, std::string precision = "double")
{
    return dispatch<TrainF>(precision, data, codes, udist, params);
}

// ---- map / predict --------------------------------------------------------
template <class T> struct MapF {
    static List run(List codes, List data, int window, double max_step, int max_length_diff, int threads)
    {
        dtwsom::SeqSet<T> C = as_seqset<T>(codes);
        dtwsom::SeqSet<T> X = as_seqset<T>(data, C.dim);
        std::vector<T> dmv, dist; std::vector<int> b1, b2;
        dtwsom::map_exact<T>(C, X, dtw_opts(window, max_step, max_length_diff), threads, dmv, b1, b2, dist);
        return List::create(_["bmu"] = IntegerVector(b1.begin(), b1.end()) + 1,
                            _["bmu2"] = IntegerVector(b2.begin(), b2.end()) + 1,
                            _["distances"] = to_vector(dist),
                            _["distmat"] = to_matrix(dmv, X.n, C.n));
    }
};

// [[Rcpp::export]]
List dtwsom_map_cpp(List codes, List data, int window, double max_step, int max_length_diff, int threads,
                    std::string precision = "double")
{
    return dispatch<MapF>(precision, codes, data, window, max_step, max_length_diff, threads);
}

// ---- distance matrices ----------------------------------------------------
template <class T> struct CrossF {
    static NumericMatrix run(List a, List b, int window, double max_step, int threads)
    {
        dtwsom::SeqSet<T> A = as_seqset<T>(a);
        dtwsom::SeqSet<T> B = as_seqset<T>(b, A.dim);
        return to_matrix(dtwsom::dtw_cross<T>(A, B, dtw_opts(window, max_step, -1), threads), A.n, B.n);
    }
};
template <class T> struct PairF {
    static NumericMatrix run(List a, int window, double max_step, int threads)
    {
        dtwsom::SeqSet<T> A = as_seqset<T>(a);
        return to_matrix(dtwsom::dtw_pairwise<T>(A, dtw_opts(window, max_step, -1), threads), A.n, A.n);
    }
};

// [[Rcpp::export]]
NumericMatrix dtw_cross_cpp(List a, List b, int window, double max_step, int threads, std::string precision = "double")
{
    return dispatch<CrossF>(precision, a, b, window, max_step, threads);
}

// [[Rcpp::export]]
NumericMatrix dtw_pairwise_cpp(List a, int window, double max_step, int threads, std::string precision = "double")
{
    return dispatch<PairF>(precision, a, window, max_step, threads);
}

// ---- single distance, returned as a string so extra digits survive --------
template <class T> struct OneF {
    static List run(NumericMatrix a, NumericMatrix b, int window, double max_step)
    {
        if (a.ncol() != b.ncol()) stop("channel mismatch");
        List la = List::create(a), lb = List::create(b);
        dtwsom::SeqSet<T> A = as_seqset<T>(la), B = as_seqset<T>(lb);
        dtwsom::Workspace<T> ws;
        T d = dtwsom::dtw_distance<T>(A.ptr(0), A.len(0), B.ptr(0), B.len(0), A.dim,
                                      dtw_opts(window, max_step, -1), dtwsom::inf<T>(), ws);
        std::ostringstream os; os.precision(std::numeric_limits<T>::max_digits10); os << d;
        // relative error against a 50-digit computation, evaluated in 50 digits
        // (so it is meaningful for types finer than double)
        dtwsom::SeqSet<bfloat50_t> A50 = as_seqset<bfloat50_t>(la), B50 = as_seqset<bfloat50_t>(lb);
        dtwsom::Workspace<bfloat50_t> ws50;
        bfloat50_t ref = dtwsom::dtw_distance<bfloat50_t>(A50.ptr(0), A50.len(0), B50.ptr(0), B50.len(0), A50.dim,
                                                          dtw_opts(window, max_step, -1), dtwsom::inf<bfloat50_t>(), ws50);
        bfloat50_t dv(os.str());
        const double rel = ref > 0 ? static_cast<double>(abs(dv - ref) / ref) : 0.0;
        return List::create(_["value"] = static_cast<double>(d), _["digits"] = os.str(),
                            _["mantissa_bits"] = std::numeric_limits<T>::digits,
                            _["rel_error_vs_bin50"] = rel);
    }
};

// [[Rcpp::export]]
List dtw_distance_cpp(NumericMatrix a, NumericMatrix b, int window, double max_step, std::string precision = "double")
{
    return dispatch<OneF>(precision, a, b, window, max_step);
}

// Warp x onto the time axis of prototype w: row i = mean of the x samples matched
// to prototype index i on the optimal DTW path (the quantity the update averages).
// [[Rcpp::export]]
NumericMatrix dtw_warp_cpp(NumericMatrix w, NumericMatrix x, int window, double max_step)
{
    if (w.ncol() != x.ncol()) stop("channel mismatch");
    List lw = List::create(w), lx = List::create(x);
    dtwsom::SeqSet<double> W = as_seqset<double>(lw), X = as_seqset<double>(lx);
    const int n = W.len(0), D = W.dim;
    dtwsom::Workspace<double> ws;
    std::vector<double> sum((size_t)n * D, 0.0), cnt(n, 0.0);
    dtwsom::DtwOpts o = dtw_opts(window, max_step, -1);
    double d = dtwsom::dtw_path_accumulate<double>(W.ptr(0), n, X.ptr(0), X.len(0), D, o, 1.0, ws, sum.data(), cnt.data());
    if (!(d < dtwsom::inf<double>())) { o.max_step = dtwsom::inf<double>();
        dtwsom::dtw_path_accumulate<double>(W.ptr(0), n, X.ptr(0), X.len(0), D, o, 1.0, ws, sum.data(), cnt.data()); }
    NumericMatrix out(n, D);
    for (int i = 0; i < n; ++i) for (int dd = 0; dd < D; ++dd) out(i, dd) = cnt[i] > 0 ? sum[(size_t)i * D + dd] / cnt[i] : NA_REAL;
    return out;
}

// [[Rcpp::export]]
int dtwsom_threads_cpp() { return dtwsom::n_threads_available(); }

// [[Rcpp::export]]
CharacterVector dtwsom_precisions_cpp()
{
    CharacterVector out = CharacterVector::create("float", "double", "longdouble", "quad", "bin50");
#ifdef DTWSOM_WITH_MPFR
    out.push_back("mpfr100");
#endif
    return out;
}

// [[Rcpp::export]]
List dtwsom_build_info_cpp()
{
    return List::create(
        _["sizeof_long_double"] = (int)sizeof(long double),
        _["avx512f"]  =
#ifdef __AVX512F__
            true,
#else
            false,
#endif
        _["avx2"] =
#ifdef __AVX2__
            true,
#else
            false,
#endif
        _["fma"] =
#ifdef __FMA__
            true,
#else
            false,
#endif
        _["openmp"] =
#ifdef _OPENMP
            true,
#else
            false,
#endif
        _["mpfr"] =
#ifdef DTWSOM_WITH_MPFR
            true,
#else
            false,
#endif
        _["compiler"] = std::string(
#ifdef __VERSION__
            __VERSION__
#else
            "unknown"
#endif
        ));
}
