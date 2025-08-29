# Monte Carlo Simulation for Supply Chain Systems

##  Overview
This project implements a **Monte Carlo Simulation (MCS)** framework for modeling stochastic supply chain systems under uncertainty.  
The simulation captures variability in:
- Demand fluctuations
- Lead times
- Supplier reliability
- Holding, ordering, and stockout costs

We parallelized the computation using **MPI (Message Passing Interface)**, enabling large-scale scenario simulation on multiple CPU cores.

---

## Project Structure
- **`controller.py`** → Python script to compile, run, and aggregate results.  
- **`mc_mpi_cpu.cpp`** → C++ MPI implementation of the supply chain simulation.  
- **`results/`** → Folder where per-rank CSV and summary outputs are stored.  
- **`HPC.pdf`** → Report documenting methodology, problem statement, and results.

---

## ⚙Features Implemented
- **Parallel Monte Carlo simulation** using MPI across multiple processes.  
- **Correlated demand modeling** via Cholesky decomposition.  
- **Supply chain KPIs tracked:**  
  - Total cost, revenue, net profit  
  - Stockout rate, service level, inventory turnover, profit margin  
- **(s, Q) inventory policy** modeled with stochastic supplier lead times and reliability.  
- **CSV result logging** (per-rank and aggregated summaries).  
- **Scalable design** for running up to millions of scenarios.

---

## 🚀Running the Project
1. **Compile the C++ code:**
   ```bash
   python controller.py --compile
   ```
2. **Run the simulation (example with 4 MPI processes):**
```bash
python controller.py --np 4 --scenarios 1000000 --time_periods 52
```
3.** Aggregate results The script automatically collects per-rank results and prints the final KPIs**
The script automatically collects per-rank results and prints the final KPIs.
  ```bash
  ENHANCED SUPPLY CHAIN SIMULATION RESULTS
  Scenarios:          1000000
  Time periods:       52
  Products:           3
  Mean total cost:    XXXX
  Mean total revenue: XXXX
  Mean net profit:    XXXX
  Mean stockout rate: 0.XXXX
  Mean service level: 0.XXXX
  Mean inventory turnover: XX.XX
  Profit margin:      XX.XX%
  ```
## 🔧 Current Limitations
- **CPU-only implementation**: Runs efficiently on CPUs via MPI, but not yet accelerated for GPU execution.  
- **Correlation simplification**: Demand correlation is modeled with a single correlation coefficient for all product pairs.  
- **Fixed (s, Q) policy**: Inventory policy is not adaptive/optimized; could be extended with reinforcement learning or heuristic optimization.  
- **No visualization yet**: Results are aggregated numerically but not plotted for better insight.  
- **Single failure model**: Supplier reliability modeled probabilistically but does not yet include cascading disruptions or multi-tier supply chains.  

---

## 🌟 Future Improvements
1. **GPU Acceleration (CUDA / OpenCL)**  
   - Implement parallel random number generation and simulation kernels on GPUs.  
   - Expect speedups in the range of $10 \times$ – $100 \times$ for large-scale scenarios.  

2. **Hybrid CPU–GPU Execution**  
   - Use MPI for multi-node distribution and CUDA for intra-node acceleration.  

3. **Better Visualization**  
   - Add Python scripts (e.g., Matplotlib / Seaborn / Plotly) for plotting distributions of cost, profit, service levels, etc.  

4. **Configurable Policies**  
   - Support advanced inventory control policies: $(s, S)$, base-stock, dynamic reorder points.  

5. **Cloud & HPC Deployment**  
   - Docker / Singularity containerization for portability.  
   - SLURM or Kubernetes job submission for HPC clusters or cloud platforms.  

6. **Sensitivity & Optimization**  
   - Use optimization algorithms (Genetic Algorithms, Particle Swarm Optimization, Bayesian Optimization) to tune parameters like reorder points and order quantities.  

7. **Real-world Data Integration**  
   - Extend framework to load demand / lead time distributions from datasets instead of synthetic normal distributions.  

---

## 👥 Team Members
- **Karthick Ruban T** (CB.AI.U4AID23017)  
- **Sanjay Senthil** (CB.AI.U4AID23042)  
- **Hariprakash V** (CB.AI.U4AID23047)  
- **Vignesh V S** (CB.AI.U4AID23057)  

---

## 📚 References
- Monte Carlo Simulation techniques in supply chain systems.  
- MPI for high-performance distributed computing.  
- CUDA / OpenCL for GPU acceleration.  

