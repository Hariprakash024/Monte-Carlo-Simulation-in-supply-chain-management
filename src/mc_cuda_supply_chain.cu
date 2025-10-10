// mc_cuda_supply_chain.cu
// CUDA Monte Carlo demo: per-thread scenario with simple inventory logic
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CUDA_CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){
  printf("CUDA Error %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} }while(0)

struct DeviceParams {
  float *demand_means,*demand_stds,*init_inv,*reorder_pts,*order_qtys,*holding,*stockout,*lt_means,*lt_stds,*ordering; int P; int T; };

__global__ void setup_states(curandState* st, unsigned long seed, int N){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<N) curand_init(seed,i,0,&st[i]); }

__global__ void simulate(curandState* st, DeviceParams prm, int N, float* out_cost, float* out_service){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return; curandState rng=st[i];
  float inv[16]; int pending[16]; int ltleft[16];
  for(int p=0;p<prm.P;p++){ inv[p]=prm.init_inv[p]; pending[p]=0; ltleft[p]=0; }
  double total_cost=0; long long met=0, tot=0; 
  for(int t=0;t<prm.T;t++){
    for(int p=0;p<prm.P;p++){
      float z=curand_normal(&rng); float d=fmaxf(0.f, prm.demand_means[p]+prm.demand_stds[p]*z); tot += (long long)d;
      if(pending[p] && ltleft[p]>0){ ltleft[p]--; if(ltleft[p]==0){ inv[p]+=prm.order_qtys[p]; pending[p]=0; } }
      if(inv[p]>=d){ inv[p]-=d; met += (long long)d; } else { float sh=d-inv[p]; total_cost += sh*prm.stockout[p]; met += (long long)inv[p]; inv[p]=0.f; }
      total_cost += inv[p]*prm.holding[p];
      if(inv[p]<=prm.reorder_pts[p] && !pending[p]){ pending[p]=1; float lt=fmaxf(0.f, prm.lt_means[p]+prm.lt_stds[p]*curand_normal(&rng)); ltleft[p]=max(1,(int)llrintf(lt)); total_cost += prm.ordering[p]; }
    }
  }
  out_cost[i]=(float)total_cost; out_service[i]=(tot>0)? (float)met/(float)tot : 1.f; st[i]=rng; }

int main(int argc,char** argv){ int N = (argc>1? atoi(argv[1]): 100000); int P=3,T=52; 
  std::vector<float> mean={500,300,200}, sd={100,50,30}, init={1000,800,600}, rp={200,150,100}, oq={500,400,300}, hold={2,2.5,3}, so={5,6,7}, lt_m={2.0,1.5,2.5}, lt_s={0.5,0.3,0.7};
  float *d_mean,*d_sd,*d_init,*d_rp,*d_oq,*d_hold,*d_so,*d_ltm,*d_lts; CUDA_CHK(cudaMalloc(&d_mean,P*sizeof(float))); CUDA_CHK(cudaMalloc(&d_sd,P*sizeof(float))); CUDA_CHK(cudaMalloc(&d_init,P*sizeof(float)));
  CUDA_CHK(cudaMalloc(&d_rp,P*sizeof(float))); CUDA_CHK(cudaMalloc(&d_oq,P*sizeof(float))); CUDA_CHK(cudaMalloc(&d_hold,P*sizeof(float))); CUDA_CHK(cudaMalloc(&d_so,P*sizeof(float))); CUDA_CHK(cudaMalloc(&d_ltm,P*sizeof(float))); CUDA_CHK(cudaMalloc(&d_lts,P*sizeof(float)));
  CUDA_CHK(cudaMemcpy(d_mean,mean.data(),P*sizeof(float),cudaMemcpyHostToDevice)); CUDA_CHK(cudaMemcpy(d_sd,sd.data(),P*sizeof(float),cudaMemcpyHostToDevice)); CUDA_CHK(cudaMemcpy(d_init,init.data(),P*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHK(cudaMemcpy(d_rp,rp.data(),P*sizeof(float),cudaMemcpyHostToDevice)); CUDA_CHK(cudaMemcpy(d_oq,oq.data(),P*sizeof(float),cudaMemcpyHostToDevice)); CUDA_CHK(cudaMemcpy(d_hold,hold.data(),P*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHK(cudaMemcpy(d_so,so.data(),P*sizeof(float),cudaMemcpyHostToDevice)); CUDA_CHK(cudaMemcpy(d_ltm,lt_m.data(),P*sizeof(float),cudaMemcpyHostToDevice)); CUDA_CHK(cudaMemcpy(d_lts,lt_s.data(),P*sizeof(float),cudaMemcpyHostToDevice));
  DeviceParams prm{d_mean,d_sd,d_init,d_rp,d_oq,d_hold,d_so,d_ltm,d_lts,nullptr,nullptr,P,T}; // extra ptrs unused
  curandState* d_states; CUDA_CHK(cudaMalloc(&d_states,N*sizeof(curandState)));
  float *d_cost,*d_serv; CUDA_CHK(cudaMalloc(&d_cost,N*sizeof(float))); CUDA_CHK(cudaMalloc(&d_serv,N*sizeof(float)));
  int threads=256, blocks=(N+threads-1)/threads; auto t0=std::chrono::high_resolution_clock::now();
  setup_states<<<blocks,threads>>>(d_states,1234UL,N); cudaDeviceSynchronize();
  simulate<<<blocks,threads>>>(d_states,prm,N,d_cost,d_serv); cudaDeviceSynchronize();
  auto t1=std::chrono::high_resolution_clock::now(); double sec=std::chrono::duration<double>(t1-t0).count();
  std::vector<float> h_cost(1024), h_serv(1024); // sample to compute rough mean
  CUDA_CHK(cudaMemcpy(h_cost.data(),d_cost, h_cost.size()*sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHK(cudaMemcpy(h_serv.data(),d_serv, h_serv.size()*sizeof(float), cudaMemcpyDeviceToHost));
  double mcost=0, mserv=0; int K=h_cost.size(); for(int i=0;i<K;i++){ mcost+=h_cost[i]; mserv+=h_serv[i]; } mcost/=K; mserv/=K;
  std::cout<<"CUDA Monte Carlo ran "<<N<<" scenarios in "<<sec<<" s\n"; std::cout<<"Sample mean cost (1k): "<<mcost<<"  sample service: "<<(mserv*100)<<"%\n";
  cudaFree(d_states); cudaFree(d_cost); cudaFree(d_serv); cudaFree(d_mean); cudaFree(d_sd); cudaFree(d_init); cudaFree(d_rp); cudaFree(d_oq); cudaFree(d_hold); cudaFree(d_so); cudaFree(d_ltm); cudaFree(d_lts);
  return 0;
}
