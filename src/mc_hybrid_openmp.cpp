// mc_hybrid_openmp.cpp
// Hybrid MPI + OpenMP Monte Carlo Simulation

#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Params {
    int scenarios = 200000;
    int periods   = 52;
    int n_products = 3;
    int omp_threads = 4;

    std::vector<float> demand_means  = {500.f, 300.f, 200.f};
    std::vector<float> demand_stds   = {100.f,  50.f,  30.f};
    std::vector<float> init_inv      = {1000.f, 800.f, 600.f};
    std::vector<float> reorder_pts   = {200.f, 150.f, 100.f};
    std::vector<float> order_qtys    = {500.f, 400.f, 300.f};
    std::vector<float> lt_means      = {2.0f, 1.5f, 2.5f};
    std::vector<float> lt_stds       = {0.5f, 0.3f, 0.7f};
    std::vector<float> holding_costs = {2.0f, 2.5f, 3.0f};
    std::vector<float> stockout_costs= {5.0f, 6.0f, 7.0f};
    std::vector<float> ordering_costs= {50.f, 45.f, 40.f};
    unsigned long seed = 42UL;
};

static inline float clamp_nonneg(float x){ return x<0.f?0.f:x; }

static std::vector<float> split_csv_f(const std::string& csv){ std::vector<float> out; std::stringstream ss(csv); std::string item; while(std::getline(ss,item,',')){ if(!item.empty()) out.push_back(std::stof(item)); } return out; }

static void parse_cli(int argc,char** argv,Params& p){
    for(int i=1;i<argc;++i){ std::string a=argv[i]; auto need=[&](const char* k){ if(i+1>=argc) throw std::runtime_error(std::string("Missing ")+k); return std::string(argv[++i]); };
        if(a=="--scenarios") p.scenarios=std::stoi(need("--scenarios"));
        else if(a=="--periods") p.periods=std::stoi(need("--periods"));
        else if(a=="--products") p.n_products=std::stoi(need("--products"));
        else if(a=="--threads") p.omp_threads=std::stoi(need("--threads"));
        else if(a=="--demand-means") p.demand_means=split_csv_f(need("--demand-means"));
        else if(a=="--demand-stds") p.demand_stds=split_csv_f(need("--demand-stds"));
        else if(a=="--init-inv") p.init_inv=split_csv_f(need("--init-inv"));
        else if(a=="--reorder-pts") p.reorder_pts=split_csv_f(need("--reorder-pts"));
        else if(a=="--order-qtys") p.order_qtys=split_csv_f(need("--order-qtys"));
        else if(a=="--lt-means") p.lt_means=split_csv_f(need("--lt-means"));
        else if(a=="--lt-stds") p.lt_stds=split_csv_f(need("--lt-stds"));
        else if(a=="--holding") p.holding_costs=split_csv_f(need("--holding"));
        else if(a=="--stockout") p.stockout_costs=split_csv_f(need("--stockout"));
        else if(a=="--ordering") p.ordering_costs=split_csv_f(need("--ordering"));
        else if(a=="--seed") p.seed=std::stoul(need("--seed")); }
    auto ensure=[&](const std::vector<float>& v,const char* n){ if((int)v.size()!=p.n_products) throw std::runtime_error(std::string("Size mismatch ")+n); };
    ensure(p.demand_means,"demand_means"); ensure(p.demand_stds,"demand_stds"); ensure(p.init_inv,"init_inv"); ensure(p.reorder_pts,"reorder_pts"); ensure(p.order_qtys,"order_qtys"); ensure(p.lt_means,"lt_means"); ensure(p.lt_stds,"lt_stds"); ensure(p.holding_costs,"holding_costs"); ensure(p.stockout_costs,"stockout_costs"); ensure(p.ordering_costs,"ordering_costs"); }

struct Agg{ double sum_cost=0,sum_cost_sq=0,sum_service=0; long long cnt=0; };

int main(int argc,char** argv){
    MPI_Init(&argc,&argv); int rank=0,size=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    Params p; if(rank==0){ try{ parse_cli(argc,argv,p);}catch(const std::exception& e){ std::cerr<<"Arg error: "<<e.what()<<"\n"; MPI_Abort(MPI_COMM_WORLD,1);} }
    MPI_Bcast(&p.scenarios,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Bcast(&p.periods,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Bcast(&p.n_products,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Bcast(&p.omp_threads,1,MPI_INT,0,MPI_COMM_WORLD);
    auto b=[&](std::vector<float>& v){ if(rank!=0) v.resize(p.n_products); MPI_Bcast(v.data(),p.n_products,MPI_FLOAT,0,MPI_COMM_WORLD); };
    b(p.demand_means); b(p.demand_stds); b(p.init_inv); b(p.reorder_pts); b(p.order_qtys); b(p.lt_means); b(p.lt_stds); b(p.holding_costs); b(p.stockout_costs); b(p.ordering_costs); MPI_Bcast(&p.seed,1,MPI_UNSIGNED_LONG,0,MPI_COMM_WORLD);
    int S=p.scenarios; int per=S/size, rem=S%size; int start=rank*per+std::min(rank,rem); int cnt=per+(rank<rem?1:0); int end=start+cnt;
    omp_set_num_threads(p.omp_threads);
    auto t0=std::chrono::high_resolution_clock::now();
    Agg local{};
    #pragma omp parallel
    {
        int tid=omp_get_thread_num(); std::mt19937 rng((unsigned)(p.seed+100000U*rank+tid)); std::normal_distribution<float> N(0.f,1.f);
        Agg thr{};
        #pragma omp for schedule(dynamic,64)
        for(int s=start;s<end;++s){
            std::vector<float> inv=p.init_inv; std::vector<int> pending(p.n_products,0), lt_left(p.n_products,0);
            double cost=0; long long met=0, tot=0;
            for(int t=0;t<p.periods;++t){
                for(int pr=0;pr<p.n_products;++pr){ float d=clamp_nonneg(p.demand_means[pr]+p.demand_stds[pr]*N(rng)); tot += (long long)std::lround(d);
                    if(pending[pr]&&lt_left[pr]>0){ if(--lt_left[pr]==0){ inv[pr]+=p.order_qtys[pr]; pending[pr]=0; } }
                    if(inv[pr]>=d){ inv[pr]-=d; met += (long long)std::lround(d);} else { float sh=d-inv[pr]; cost+=sh*p.stockout_costs[pr]; met+=(long long)std::lround(inv[pr]); inv[pr]=0.f; }
                    cost += inv[pr]*p.holding_costs[pr];
                    if(inv[pr]<=p.reorder_pts[pr] && !pending[pr]){ pending[pr]=1; float lt=clamp_nonneg(p.lt_means[pr]+p.lt_stds[pr]*N(rng)); lt_left[pr]=std::max(1,(int)std::lround(lt)); cost += p.ordering_costs[pr]; }
                }
            }
            double service = (tot>0)? (double)met/(double)tot : 1.0;
            thr.sum_cost += cost; thr.sum_cost_sq += cost*cost; thr.sum_service += service; thr.cnt++;
        }
        #pragma omp atomic
        local.sum_cost += thr.sum_cost;
        #pragma omp atomic
        local.sum_cost_sq += thr.sum_cost_sq;
        #pragma omp atomic
        local.sum_service += thr.sum_service;
        #pragma omp atomic
        local.cnt += thr.cnt;
    }
    auto t1=std::chrono::high_resolution_clock::now(); double el=std::chrono::duration<double>(t1-t0).count();
    Agg g{}; MPI_Reduce(&local.sum_cost,&g.sum_cost,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); MPI_Reduce(&local.sum_cost_sq,&g.sum_cost_sq,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); MPI_Reduce(&local.sum_service,&g.sum_service,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); long long gcnt=0; MPI_Reduce(&local.cnt,&gcnt,1,MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD); double maxel=0; MPI_Reduce(&el,&maxel,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0){ double mean=g.sum_cost/(double)gcnt; double var=g.sum_cost_sq/(double)gcnt-mean*mean; double sd=var>0?std::sqrt(var):0; double serv=g.sum_service/(double)gcnt; std::cout<<"=== Hybrid MPI+OpenMP Results ===\n"; std::cout<<"Scenarios: "<<gcnt<<" Periods: "<<p.periods<<" Products: "<<p.n_products<<" OMP: "<<p.omp_threads<<" Ranks: "<<size<<"\n"; std::cout<<"Mean Cost: "<<std::fixed<<std::setprecision(2)<<mean<<"  StdDev: "<<sd<<"\n"; std::cout<<"Service Level: "<<(serv*100.0)<<"%\n"; std::cout<<"Elapsed (max rank): "<<std::setprecision(3)<<maxel<<" s\n"; }
    MPI_Finalize(); return 0;
}
