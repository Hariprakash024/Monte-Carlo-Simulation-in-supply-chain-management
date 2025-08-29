# Monte Carlo Simulation for Supply Chain Systems

## 📌 Overview
This project implements a **Monte Carlo Simulation (MCS)** framework for modeling stochastic supply chain systems under uncertainty.  
The simulation captures variability in:
- Demand fluctuations
- Lead times
- Supplier reliability
- Holding, ordering, and stockout costs

We parallelized the computation using **MPI (Message Passing Interface)**, enabling large-scale scenario simulation on multiple CPU cores.

---

## 🧑‍💻 Project Structure
- **`controller.py`** → Python script to compile, run, and aggregate results.  
- **`mc_mpi_cpu.cpp`** → C++ MPI implementation of the supply chain simulation.  
- **`results/`** → Folder where per-rank CSV and summary outputs are stored.  
- **`HPC.pdf`** → Report documenting methodology, problem statement, and results.

---

## ⚙️ Features Implemented
- **Parallel Monte Carlo simulation** using MPI across multiple processes.  
- **Correlated demand modeling** via Cholesky decomposition.  
- **Supply chain KPIs tracked:**  
  - Total cost, revenue, net profit  
  - Stockout rate, service level, inventory turnover, profit margin  
- **(s, Q) inventory policy** modeled with stochastic supplier lead times and reliability.  
- **CSV result logging** (per-rank and aggregated summaries).  
- **Scalable design** for running up to millions of scenarios.

---

## 🚀 Running the Project
1. **Compile the C++ code:**
   ```bash
   python controller.py --compile
