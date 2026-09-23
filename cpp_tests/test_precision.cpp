// Checks that the core compiles and runs with every supported number type.
//   g++ -std=c++17 -O2 -fopenmp -DDTWSOM_WITH_MPFR tests/test_precision.cpp -lmpfr -lgmp -o test_precision
#include "../src/dtwsom_core.hpp"
#include <boost/multiprecision/cpp_bin_float.hpp>
#ifdef DTWSOM_WITH_MPFR
#include <boost/multiprecision/mpfr.hpp>
#endif
#include <cstdio>
#include <iostream>
#include <iomanip>
namespace mp = boost::multiprecision;
using quad  = mp::number<mp::cpp_bin_float<113, mp::digit_base_2>, mp::et_off>;   // IEEE binary128 layout
using bin50 = mp::number<mp::cpp_bin_float<50>, mp::et_off>;
#ifdef DTWSOM_WITH_MPFR
using mpfr100 = mp::number<mp::mpfr_float_backend<100>, mp::et_off>;
#endif

template <class T> void run(const char* name, const std::vector<double>& a, const std::vector<double>& b) {
    using namespace dtwsom;
    std::vector<T> ta(a.begin(), a.end()), tb(b.begin(), b.end());
    DtwOpts o; Workspace<T> ws;
    T d = dtw_distance<T>(ta.data(), ta.size(), tb.data(), tb.size(), 1, o, inf<T>(), ws);
    // small SOM: 4 units, 20 series
    SeqSet<T> data = SeqSet<T>::empty(1), codes = SeqSet<T>::empty(1);
    for (int s = 0; s < 20; ++s) { std::vector<T> v(30); for (int t = 0; t < 30; ++t) v[t] = T(std::sin(t * 0.3 + s)) ; data.push(v.data(), 30); }
    for (int k = 0; k < 4; ++k) codes.push(data.ptr(k * 5), data.len(k * 5));
    std::vector<double> ud = {0,1,1,1.414, 1,0,1.414,1, 1,1.414,0,1, 1.414,1,1,0};
    Params p; p.epochs = 3; p.radius = {1.0, 0.7, 0.4}; p.alpha = {0.1, 0.05, 0.02};
    Result<T> rb = train<T>(data, codes, ud, p);
    p.batch = false; Result<T> ro = train<T>(data, codes, ud, p);
    std::cout << std::setw(9) << name << "  dtw = " << std::setprecision(40) << d
              << "   batch QE " << std::setprecision(6) << rb.qe_epoch.back() << "  online QE " << ro.qe_epoch.back() << "\n";
}
int main() {
    std::mt19937 rng(7); std::normal_distribution<double> nd(0, 1);
    std::vector<double> a(300), b(280);
    for (auto& v : a) v = 1e6 + nd(rng);      // large DC offset: float should visibly break
    for (auto& v : b) v = 1e6 + nd(rng);
    run<float>("float", a, b); run<double>("double", a, b); run<long double>("longdbl", a, b);
    run<quad>("quad", a, b); run<bin50>("bin50", a, b);
#ifdef DTWSOM_WITH_MPFR
    run<mpfr100>("mpfr100", a, b);
#endif
}
