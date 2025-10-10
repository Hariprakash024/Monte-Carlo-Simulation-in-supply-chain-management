// mc_mpi_cpu.cpp
// Monte Carlo Supply Chain Simulation (MPI CPU version)

#include <mpi.h>
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
    int scenarios = 100000; // default small for demo
    int periods   = 52;
    int n_products = 3;

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

static inline float clamp_nonneg(float x) { return x < 0.f ? 0.f : x; }

static std::vector<float> split_csv_f(const std::string& csv) {
    std::vector<float> out; std::stringstream ss(csv); std::string item;
    while (std::getline(ss, item, ',')) { if (!item.empty()) out.push_back(std::stof(item)); }
    return out;
}

static void parse_cli(int argc, char** argv, Params& p) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* k){ if (i+1>=argc) throw std::runtime_error(std::string("Missing ")+k); return std::string(argv[++i]); };
        if (a=="--scenarios") p.scenarios = std::stoi(need("--scenarios"));
        else if (a=="--periods") p.periods = std::stoi(need("--periods"));
        else if (a=="--products") p.n_products = std::stoi(need("--products"));
        else if (a=="--demand-means") p.demand_means = split_csv_f(need("--demand-means"));
        else if (a=="--demand-stds")  p.demand_stds  = split_csv_f(need("--demand-stds"));
        else if (a=="--init-inv")     p.init_inv     = split_csv_f(need("--init-inv"));
        else if (a=="--reorder-pts")  p.reorder_pts  = split_csv_f(need("--reorder-pts"));
        else if (a=="--order-qtys")   p.order_qtys   = split_csv_f(need("--order-qtys"));
        else if (a=="--lt-means")     p.lt_means     = split_csv_f(need("--lt-means"));
        else if (a=="--lt-stds")      p.lt_stds      = split_csv_f(need("--lt-stds"));
        else if (a=="--holding")      p.holding_costs= split_csv_f(need("--holding"));
        else if (a=="--stockout")     p.stockout_costs=split_csv_f(need("--stockout"));
        else if (a=="--ordering")     p.ordering_costs=split_csv_f(need("--ordering"));
        else if (a=="--seed")         p.seed         = std::stoul(need("--seed"));
    }
    auto ensure = [&](const std::vector<float>& v,const char* n){ if((int)v.size()!=p.n_products) throw std::runtime_error(std::string("Size mismatch ")+n); };
    ensure(p.demand_means,"demand_means"); ensure(p.demand_stds,"demand_stds"); ensure(p.init_inv,"init_inv");
    ensure(p.reorder_pts,"reorder_pts"); ensure(p.order_qtys,"order_qtys"); ensure(p.lt_means,"lt_means");
    ensure(p.lt_stds,"lt_stds"); ensure(p.holding_costs,"holding_costs"); ensure(p.stockout_costs,"stockout_costs"); ensure(p.ordering_costs,"ordering_costs");
}

struct Results { double sum_cost=0, sum_service=0, sum_cost_sq=0; long long scen=0; };

static Results simulate_chunk(const Params& p,int start,int end,int rank){
    std::mt19937 rng((unsigned)(p.seed+1337U*rank)); std::normal_distribution<float> N(0.f,1.f);
    Results r{}; int P=p.n_products;
    for(int s=start;s<end;++s){
        std::vector<float> inv=p.init_inv; std::vector<int> pending(P,0), lt_left(P,0);
        double total_cost=0; long long met=0, demand_total=0;
        for(int t=0;t<p.periods;++t){
            for(int pr=0;pr<P;++pr){
                float demand = clamp_nonneg(p.demand_means[pr]+p.demand_stds[pr]*N(rng));
                demand_total += (long long)std::lround(demand);
                if(pending[pr] && lt_left[pr]>0){ if(--lt_left[pr]==0){ inv[pr]+=p.order_qtys[pr]; pending[pr]=0; } }
                if(inv[pr]>=demand){ inv[pr]-=demand; met += (long long)std::lround(demand); }
                else { float shortage=demand-inv[pr]; total_cost += shortage*p.stockout_costs[pr]; met += (long long)std::lround(inv[pr]); inv[pr]=0.f; }
                total_cost += inv[pr]*p.holding_costs[pr];
                if(inv[pr]<=p.reorder_pts[pr] && !pending[pr]){ pending[pr]=1; float lt=clamp_nonneg(p.lt_means[pr]+p.lt_stds[pr]*N(rng)); lt_left[pr]=std::max(1,(int)std::lround(lt)); total_cost += p.ordering_costs[pr]; }
            }
        }
        double service = (demand_total>0)? (double)met/(double)demand_total : 1.0;
        r.sum_cost += total_cost; r.sum_cost_sq += total_cost*total_cost; r.sum_service += service; r.scen++;
    }
    return r;
}

int main(int argc,char** argv){
    MPI_Init(&argc,&argv); int rank=0,size=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    Params p; if(rank==0){ try{ parse_cli(argc,argv,p);} catch(const std::exception& e){ std::cerr<<"Arg error: "<<e.what()<<"\n"; MPI_Abort(MPI_COMM_WORLD,1);} }
    MPI_Bcast(&p.scenarios,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Bcast(&p.periods,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Bcast(&p.n_products,1,MPI_INT,0,MPI_COMM_WORLD);
    auto bcast=[&](std::vector<float>& v){ if(rank!=0)v.resize(p.n_products); MPI_Bcast(v.data(),p.n_products,MPI_FLOAT,0,MPI_COMM_WORLD); };
    bcast(p.demand_means); bcast(p.demand_stds); bcast(p.init_inv); bcast(p.reorder_pts); bcast(p.order_qtys); bcast(p.lt_means); bcast(p.lt_stds); bcast(p.holding_costs); bcast(p.stockout_costs); bcast(p.ordering_costs);
    MPI_Bcast(&p.seed,1,MPI_UNSIGNED_LONG,0,MPI_COMM_WORLD);
    int S=p.scenarios; int per=S/size, rem=S%size; int start=rank*per+std::min(rank,rem); int cnt=per+(rank<rem?1:0); int end=start+cnt;
    auto t0=std::chrono::high_resolution_clock::now(); Results local=simulate_chunk(p,start,end,rank); auto t1=std::chrono::high_resolution_clock::now(); double el=std::chrono::duration<double>(t1-t0).count();
    Results g{}; MPI_Reduce(&local.sum_cost,&g.sum_cost,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); MPI_Reduce(&local.sum_service,&g.sum_service,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); MPI_Reduce(&local.sum_cost_sq,&g.sum_cost_sq,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    long long lsc=local.scen,gsc=0; MPI_Reduce(&lsc,&gsc,1,MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD); double maxel=0; MPI_Reduce(&el,&maxel,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0){ double mean=g.sum_cost/(double)gsc; double var=g.sum_cost_sq/(double)gsc-mean*mean; double sd=var>0?std::sqrt(var):0; double serv=g.sum_service/(double)gsc; std::cout<<"=== MPI Monte Carlo Results ===\n"; std::cout<<"Scenarios: "<<gsc<<" Periods: "<<p.periods<<" Products: "<<p.n_products<<"\n"; std::cout<<"Mean Total Cost: "<<std::fixed<<std::setprecision(2)<<mean<<"\n"; std::cout<<"StdDev Cost: "<<sd<<"\n"; std::cout<<"Mean Service Level: "<<(serv*100.0)<<"%\n"; std::cout<<"Elapsed (max rank): "<<std::setprecision(3)<<maxel<<" s\n"; }
    MPI_Finalize(); return 0;
}
