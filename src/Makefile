# Makefile for local builds

NVCC ?= nvcc
ARCH ?= sm_75
CXX ?= mpicxx
CXXFLAGS = -O3 -std=c++17

all: mc_cuda_supply_chain

mc_cuda_supply_chain: mc_cuda_supply_chain.cu
	$(NVCC) -O3 -arch=$(ARCH) $< -o $@ -lcurand -lcublas

mpi: mc_mpi_cpu

mc_mpi_cpu: mc_mpi_cpu.cpp
	$(CXX) $(CXXFLAGS) mc_mpi_cpu.cpp -o mc_mpi_cpu

hybrid: mc_hybrid_openmp

mc_hybrid_openmp: mc_hybrid_openmp.cpp
	$(CXX) $(CXXFLAGS) -fopenmp mc_hybrid_openmp.cpp -o mc_hybrid_openmp

clean:
	rm -f mc_cuda_supply_chain mc_mpi_cpu mc_hybrid_openmp
