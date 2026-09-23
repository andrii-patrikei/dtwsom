// Standalone sanity test of the R-free core:  g++ -std=c++17 -O2 -fopenmp tests/test_core.cpp -o test_core
#include "../src/dtwsom_core.hpp"
#include <cstdio>
#include <cassert>
using namespace dtwsom;
static const double INF = inf<double>();

// brute-force DTW (full matrix, no band) for cross-checking
static double dtw_naive(const std::vector<double>& a, const std::vector<double>& b) {
    int n=a.size(), m=b.size(); std::vector<double> D((n+1)*(m+1), INF); D[0]=0;
    for(int i=1;i<=n;++i) for(int j=1;j<=m;++j){ double c=(a[i-1]-b[j-1])*(a[i-1]-b[j-1]);
        D[i*(m+1)+j]=c+std::min({D[(i-1)*(m+1)+j-1],D[(i-1)*(m+1)+j],D[i*(m+1)+j-1]}); }
    return std::sqrt(D[n*(m+1)+m]);
}

int main(){
    std::mt19937 rng(42); std::normal_distribution<double> nd(0,1);
    // 1. dtw_distance == naive, path version gives the same distance and counts cover every i
    for(int rep=0; rep<200; ++rep){
        int n=5+rng()%40, m=5+rng()%40; std::vector<double> a(n), b(m);
        for(auto& v:a) v=nd(rng);
        for(auto& v:b) v=nd(rng);
        DtwOpts o; Workspace<double> ws;
        double d1=dtw_distance(a.data(),n,b.data(),m,1,o,INF,ws), d0=dtw_naive(a,b);
        std::vector<double> sum(n,0), cnt(n,0);
        double d2=dtw_path_accumulate(a.data(),n,b.data(),m,1,o,1.0,ws,sum.data(),cnt.data());
        assert(std::fabs(d1-d0)<1e-9 && std::fabs(d2-d0)<1e-9);
        for(int i=0;i<n;++i) assert(cnt[i]>=1);
        // early abandon: threshold below the true distance -> INF, above -> exact
        assert(dtw_distance(a.data(),n,b.data(),m,1,o,d0*0.9,ws)==INF);
        assert(std::fabs(dtw_distance(a.data(),n,b.data(),m,1,o,d0*1.0001,ws)-d0)<1e-9);
    }
    // 2. window >= |n-m| still reaches the corner; identical series -> 0
    { std::vector<double> a(50), b(60); for(int i=0;i<50;++i) a[i]=std::sin(i*0.2); for(int i=0;i<60;++i) b[i]=std::sin(i*0.2);
      DtwOpts o; o.window=3; Workspace<double> ws; double d=dtw_distance(a.data(),50,b.data(),60,1,o,INF,ws); assert(d<INF);
      assert(dtw_distance(a.data(),50,a.data(),50,1,o,INF,ws)==0); }
    // 3. tiny SOM run (batch and online) on 3 motif shapes, multivariate D=2
    SeqSet<double> data=SeqSet<double>::empty(2);
    std::vector<int> label;
    for(int s=0;s<90;++s){ int cls=s%3, L=30+rng()%30; std::vector<double> v(L*2);
        for(int t=0;t<L;++t){ double u=(double)t/(L-1);
            double y = cls==0? std::sin(2*M_PI*u) : cls==1? (u<0.5?2*u:2-2*u) : (u<0.5?1:-1);
            v[t*2]=y+0.1*nd(rng); v[t*2+1]=0.5*y+0.1*nd(rng);} data.push(v.data(),L); label.push_back(cls); }
    const int K=9; std::vector<double> ud(K*K); for(int i=0;i<K;++i) for(int j=0;j<K;++j){ int xi=i%3,yi=i/3,xj=j%3,yj=j/3; ud[i*K+j]=std::hypot(xi-xj,yi-yj);}
    for(int batch=0; batch<2; ++batch){
        SeqSet<double> codes=SeqSet<double>::empty(2); for(int k=0;k<K;++k) codes.push(data.ptr(k*7%90), data.len(k*7%90));
        Params p; p.epochs=15; p.batch=batch; p.nb_type=0;
        for(int e=0;e<p.epochs;++e){ p.radius.push_back(1.5-1.0*e/(p.epochs-1)); p.alpha.push_back(0.1-0.08*e/(p.epochs-1)); }
        Result<double> r=train(data,codes,ud,p,[](const std::string& s){ std::printf("  %s\n",s.c_str()); });
        // purity: each unit dominated by one class
        std::vector<std::vector<int>> c(K,std::vector<int>(3,0)); for(int s=0;s<90;++s) c[r.bmu[s]][label[s]]++;
        int pur=0; for(int k=0;k<K;++k) pur+=*std::max_element(c[k].begin(),c[k].end());
        std::printf("%s: final QE %.4f  purity %.3f  epochs %d\n", batch?"batch":"online", 
            std::accumulate(r.dist.begin(),r.dist.end(),0.0)/90, pur/90.0, r.epochs_run);
        assert(pur/90.0 > 0.9);
    }
    std::printf("all core tests passed (threads=%d)\n", n_threads_available());
}
