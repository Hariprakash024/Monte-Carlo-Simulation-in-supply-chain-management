import argparse
import os
import subprocess
import sys
from pathlib import Path

def compile_cpp(source="mc_mpi_cpu.cpp", output="mc_mpi_cpu", extra=[]):
    
    cmd = ["mpicxx", "-O3", "-std=c++17", source, "-o", output] + extra
    print("Compiling:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        print(f"Successfully built: ./{output}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print("Error: mpicxx compiler not found. Please install MPI (e.g., OpenMPI or MPICH).")
        return False

def run_job(binary="./mc_mpi_cpu",
            np=2,
            total_scenarios=1_000_000,
            time_periods=52,
            demand_means="500,300,200",
            demand_stds="100,50,30",
            n_products=3,
            initial_inventory="1000,800,600",
            reorder_points="200,150,100",
            order_quantities="500,400,300",
            lead_time_means="2.0,1.5,2.5",
            lead_time_stds="0.5,0.3,0.7",
            holding_costs="2.0,2.5,3.0",
            stockout_costs="5.0,6.0,7.0",
            ordering_costs="8.0,9.0,10.0",
            sales_prices="10.0,12.0,15.0",
            supplier_reliability="0.95,0.90,0.85",
            demand_correlation="0.3",
            write_results=False,
            csv_dir="results"):
    
    # Check if binary exists
    if not Path(binary).exists():
        print(f"Error: Binary '{binary}' not found. Use --compile to build it first.")
        return False
    
    args = [
        "mpirun", "-np", str(np), binary,
        "--scenarios", str(total_scenarios),
        "--time_periods", str(time_periods),
        "--n_products", str(n_products),
        "--demand_means", demand_means,
        "--demand_stds", demand_stds,
        "--initial_inventory", initial_inventory,
        "--reorder_points", reorder_points,
        "--order_quantities", order_quantities,
        "--lead_time_means", lead_time_means,
        "--lead_time_stds", lead_time_stds,
        "--holding_costs", holding_costs,
        "--stockout_costs", stockout_costs,
        "--ordering_costs", ordering_costs,
        "--sales_prices", sales_prices,
        "--supplier_reliability", supplier_reliability,
        "--demand_correlation", demand_correlation,
        "--write_results", str(int(write_results)),
        "--csv_dir", csv_dir,
    ]
    print("Running:", " ".join(args))
    try:
        proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"Process failed with return code {proc.returncode}")
            return False
        return True
    except Exception as e:
        print(f"Failed to run job: {e}")
        return False

def aggregate_results(np, csv_dir, scenarios, time_periods, n_products):
    total_n = 0
    total_sum_cost = 0.0
    total_sum_revenue = 0.0
    total_sum_profit = 0.0
    total_sum_stockout = 0.0
    total_sum_service = 0.0
    total_sum_turnover = 0.0
    
    for rank in range(np):
        fn = f"{csv_dir}/summary_rank{rank}.csv"
        if not Path(fn).exists():
            print(f"Error: Summary file {fn} not found.")
            return False
        with open(fn, 'r') as f:
            line = f.readline().strip()
            parts = line.split(',')
            if len(parts) != 7:
                print(f"Error: Invalid summary format in {fn}.")
                return False
            total_n += int(parts[0])
            total_sum_cost += float(parts[1])
            total_sum_revenue += float(parts[2])
            total_sum_profit += float(parts[3])
            total_sum_stockout += float(parts[4])
            total_sum_service += float(parts[5])
            total_sum_turnover += float(parts[6])
    
    if total_n == 0:
        print("Error: No scenarios simulated.")
        return False
    if total_n != scenarios:
        print(f"Warning: Total scenarios {total_n} does not match expected {scenarios}.")
    
    mean_cost = total_sum_cost / total_n
    mean_revenue = total_sum_revenue / total_n
    mean_profit = total_sum_profit / total_n
    mean_stockout = total_sum_stockout / total_n
    mean_service = total_sum_service / total_n
    mean_turnover = total_sum_turnover / total_n
    profit_margin = (mean_profit / mean_revenue * 100) if mean_revenue > 0 else 0.0
    
    print("ENHANCED SUPPLY CHAIN SIMULATION RESULTS")
    print(f"  Scenarios:          {total_n}")
    print(f"  Time periods:       {time_periods}")
    print(f"  Products:           {n_products}")
    print(f"  Mean total cost:    {mean_cost:.2f}")
    print(f"  Mean total revenue: {mean_revenue:.2f}")
    print(f"  Mean net profit:    {mean_profit:.2f}")
    print(f"  Mean stockout rate: {mean_stockout:.4f}")
    print(f"  Mean service level: {mean_service:.4f}")
    print(f"  Mean inventory turnover: {mean_turnover:.2f}")
    print(f"  Profit margin:      {profit_margin:.2f}%")
    return True

def main():
    p = argparse.ArgumentParser(description="Enhanced Supply Chain Monte Carlo Simulation")
    p.add_argument("--compile", action="store_true", help="Compile mc_mpi_cpu.cpp first")
    p.add_argument("--np", type=int, default=2, help="Number of MPI processes")
    p.add_argument("--binary", default="./mc_mpi_cpu", help="Path to compiled binary")

    # Model params
    p.add_argument("--scenarios", type=int, default=1_000_000, help="Total number of scenarios to simulate")
    p.add_argument("--time_periods", type=int, default=52, help="Number of time periods per scenario")
    p.add_argument("--n_products", type=int, default=3, help="Number of products in supply chain")
    
    # Product-specific parameters (comma-separated lists)
    p.add_argument("--demand_means", default="500,300,200", help="Mean demand for each product")
    p.add_argument("--demand_stds", default="100,50,30", help="Demand std deviation for each product")
    p.add_argument("--initial_inventory", default="1000,800,600", help="Initial inventory for each product")
    p.add_argument("--reorder_points", default="200,150,100", help="Reorder points for each product")
    p.add_argument("--order_quantities", default="500,400,300", help="Order quantities for each product")
    p.add_argument("--lead_time_means", default="2.0,1.5,2.5", help="Mean lead times for each product")
    p.add_argument("--lead_time_stds", default="0.5,0.3,0.7", help="Lead time std deviations for each product")
    p.add_argument("--holding_costs", default="2.0,2.5,3.0", help="Holding costs per unit for each product")
    p.add_argument("--stockout_costs", default="5.0,6.0,7.0", help="Stockout costs per unit for each product")
    p.add_argument("--ordering_costs", default="8.0,9.0,10.0", help="Ordering costs per unit for each product")
    p.add_argument("--sales_prices", default="10.0,12.0,15.0", help="Sales prices per unit for each product")
    p.add_argument("--supplier_reliability", default="0.95,0.90,0.85", help="Supplier reliability for each product")
    
    # System-wide parameters
    p.add_argument("--demand_correlation", default="0.3", help="Correlation between product demands")
    p.add_argument("--write_results", action="store_true", help="Write detailed results to CSV files")
    p.add_argument("--csv_dir", default="results", help="Directory for output CSV files")

    args = p.parse_args()

    success = True
    
    if args.compile:
        success = compile_cpp()
        if not success:
            print("Compilation failed. Exiting.")
            return

    Path(args.csv_dir).mkdir(parents=True, exist_ok=True)

    if success:
        success = run_job(
            binary=args.binary,
            np=args.np,
            total_scenarios=args.scenarios,
            time_periods=args.time_periods,
            demand_means=args.demand_means,
            demand_stds=args.demand_stds,
            n_products=args.n_products,
            initial_inventory=args.initial_inventory,
            reorder_points=args.reorder_points,
            order_quantities=args.order_quantities,
            lead_time_means=args.lead_time_means,
            lead_time_stds=args.lead_time_stds,
            holding_costs=args.holding_costs,
            stockout_costs=args.stockout_costs,
            ordering_costs=args.ordering_costs,
            sales_prices=args.sales_prices,
            supplier_reliability=args.supplier_reliability,
            demand_correlation=args.demand_correlation,
            write_results=args.write_results,
            csv_dir=args.csv_dir
        )
    
    if success:
        success = aggregate_results(
            args.np,
            args.csv_dir,
            args.scenarios,
            args.time_periods,
            args.n_products
        )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()