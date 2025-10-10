#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import List
import numpy as np

def parse_cpp_params(cpp_file: str) -> dict:
    """Parse default parameter values from the Params struct in the C++ file without regex."""
    try:
        with open(cpp_file, 'r') as f:
            content = f.read().splitlines()
        
        # Find the Params struct
        start_idx = -1
        for i, line in enumerate(content):
            if 'struct Params {' in line.strip():
                start_idx = i + 1
                break
        if start_idx == -1:
            raise ValueError(f"Could not find 'struct Params {{' in {cpp_file}")
        
        # Collect lines until the struct ends
        params_lines = []
        for line in content[start_idx:]:
            line = line.strip()
            if line == '};':
                break
            if line and not line.startswith('//'):
                params_lines.append(line)
        
        # Initialize defaults
        defaults = {
            'scenarios': 1_000_000,
            'time_periods': 55,
            'n_products': 4,
            'demand_means': '500,300,200,400',
            'demand_stds': '100,50,30,67',
            'initial_inventory': '1000,800,600,450',
            'reorder_points': '200,150,100,300',
            'order_quantities': '500,400,300,250',
            'lead_time_means': '2.0,1.5,2.5,4.6',
            'lead_time_stds': '0.5,0.3,0.7,0.6',
            'holding_costs': '2.0,2.5,3.0,4.6',
            'stockout_costs': '5.0,6.0,7.0,3.0',
            'ordering_costs': '8.0,9.0,10.0,9.0',
            'sales_prices': '10.0,12.0,15.0,13.0',
            'supplier_reliability': '0.95,0.90,0.85,0.54',
            'demand_correlation': 0.54,
            'write_results': False,
            'csv_dir': 'results'
        }
        
        # Parse each line in the Params struct
        for line in params_lines:
            # Skip empty or invalid lines
            if not line or '=' not in line:
                continue
            # Split on '=' and clean up
            key_value = line.split('=')
            if len(key_value) != 2:
                continue
            key = key_value[0].strip().split()[-1]  # Get variable name
            value = key_value[1].strip().rstrip(';').strip()
            
            # Handle different types
            if key == 'scenarios':
                # Remove thousands separator and convert to int
                defaults['scenarios'] = int(value.replace("'", ""))
            elif key == 'time_periods' or key == 'n_products':
                defaults[key] = int(value)
            elif key == 'demand_correlation':
                defaults[key] = float(value.rstrip('f'))
            elif key == 'write_results':
                defaults[key] = value.lower() == 'true'
            elif key == 'csv_dir':
                defaults[key] = value.strip('"')
            else:
                # Handle vector initializations like {500.f, 300.f, 200.f, 400.f}
                if value.startswith('{') and value.endswith('}'):
                    values = value[1:-1].split(',')
                    # Convert to comma-separated string, remove 'f' suffix
                    values = [v.strip().rstrip('f') for v in values]
                    defaults[key] = ','.join(values)
        
        # Set np default (not in C++ Params, so use a reasonable default)
        defaults['np'] = 2
        
        return defaults
    except FileNotFoundError:
        raise FileNotFoundError(f"C++ file {cpp_file} not found")
    except Exception as e:
        raise ValueError(f"Error parsing {cpp_file}: {e}")

def validate_csv_list(value: str, expected_length: int, type_cast=float) -> List[float]:
    """Validate and parse a comma-separated string into a list of values."""
    try:
        values = [type_cast(x) for x in value.split(',') if x.strip()]
        if len(values) != expected_length:
            raise ValueError(f"Expected {expected_length} values, got {len(values)}")
        return values
    except ValueError as e:
        raise ValueError(f"Invalid CSV list format for '{value}': {e}")

def compile_cpp(source: str = "mc_mpi_cpu.cpp", output: str = "mc_mpi_cpu", extra: List[str] = None) -> bool:
    """Compile the C++ source code using mpicxx."""
    extra = extra or []
    cmd = ["mpicxx", "-O3", "-std=c++17", source, "-o", output] + extra
    print(f"Compiling: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)
        print(f"Successfully built: ./{output}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print("Error: mpicxx compiler not found. Please install MPI (e.g., OpenMPI or MPICH).")
        return False

def run_job(
    binary: str,
    num_processes: int,
    total_scenarios: int,
    time_periods: int,
    demand_means: str,
    demand_stds: str,
    n_products: int,
    initial_inventory: str,
    reorder_points: str,
    order_quantities: str,
    lead_time_means: str,
    lead_time_stds: str,
    holding_costs: str,
    stockout_costs: str,
    ordering_costs: str,
    sales_prices: str,
    supplier_reliability: str,
    demand_correlation: float,
    write_results: bool,
    csv_dir: str
) -> bool:
    """Run the MPI simulation job with the specified parameters."""
    if not Path(binary).exists():
        print(f"Error: Binary '{binary}' not found. Use --compile to build it first.")
        return False

    args = [
        "mpirun", "-np", str(num_processes), binary,
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
        "--demand_correlation", str(demand_correlation),
        "--write_results", str(int(write_results)),
        "--csv_dir", csv_dir,
    ]

    print(f"Running: {' '.join(args)}")
    try:
        proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"Process failed with return code {proc.returncode}")
            return False
        return True
    except subprocess.SubprocessError as e:
        print(f"Failed to run job: {e}")
        return False

def aggregate_results(num_processes: int, csv_dir: str, scenarios: int, time_periods: int, n_products: int) -> bool:
    """Aggregate results from all MPI ranks and compute statistics."""
    totals = np.zeros(6)  # [cost, revenue, profit, stockout, service, turnover]
    total_n = 0

    for rank in range(num_processes):
        fn = Path(csv_dir) / f"summary_rank{rank}.csv"
        if not fn.exists():
            print(f"Error: Summary file {fn} not found.")
            return False
        try:
            with open(fn, 'r') as f:
                parts = f.readline().strip().split(',')
                if len(parts) != 7:
                    print(f"Error: Invalid summary format in {fn}.")
                    return False
                total_n += int(parts[0])
                totals += np.array([float(x) for x in parts[1:]])
        except (IOError, ValueError) as e:
            print(f"Error reading {fn}: {e}")
            return False

    if total_n == 0:
        print("Error: No scenarios simulated.")
        return False
    if total_n != scenarios:
        print(f"Warning: Total scenarios {total_n} does not match expected {scenarios}.")

    means = totals / total_n
    profit_margin = (means[2] / means[1] * 100) if means[1] > 0 else 0.0

    print("\nENHANCED SUPPLY CHAIN SIMULATION RESULTS")
    print(f"  Scenarios:          {total_n}")
    print(f"  Time periods:       {time_periods}")
    print(f"  Products:           {n_products}")
    print(f"  Mean total cost:    {means[0]:.2f}")
    print(f"  Mean total revenue: {means[1]:.2f}")
    print(f"  Mean net profit:    {means[2]:.2f}")
    print(f"  Mean stockout rate: {means[3]:.4f}")
    print(f"  Mean service level: {means[4]:.4f}")
    print(f"  Mean inventory turnover: {means[5]:.2f}")
    print(f"  Profit margin:      {profit_margin:.2f}%")
    return True

def main():
    """Main function to parse arguments and orchestrate simulation."""
    parser = argparse.ArgumentParser(description="Enhanced Supply Chain Monte Carlo Simulation")
    parser.add_argument("--compile", action="store_true", help="Compile mc_mpi_cpu.cpp first")
    parser.add_argument("--cpp_file", default="mc_mpi_cpu.cpp", help="Path to C++ source file")
    parser.add_argument("--binary", default="./mc_mpi_cpu", help="Path to compiled binary")
    parser.add_argument("--np", type=int, help="Number of MPI processes")
    parser.add_argument("--scenarios", type=int, help="Total number of scenarios to simulate")
    parser.add_argument("--time_periods", type=int, help="Number of time periods per scenario")
    parser.add_argument("--n_products", type=int, help="Number of products in supply chain")
    parser.add_argument("--demand_means", help="Mean demand for each product")
    parser.add_argument("--demand_stds", help="Demand std deviation for each product")
    parser.add_argument("--initial_inventory", help="Initial inventory for each product")
    parser.add_argument("--reorder_points", help="Reorder points for each product")
    parser.add_argument("--order_quantities", help="Order quantities for each product")
    parser.add_argument("--lead_time_means", help="Mean lead times for each product")
    parser.add_argument("--lead_time_stds", help="Lead time std deviations for each product")
    parser.add_argument("--holding_costs", help="Holding costs per unit for each product")
    parser.add_argument("--stockout_costs", help="Stockout costs per unit for each product")
    parser.add_argument("--ordering_costs", help="Ordering costs per unit for each product")
    parser.add_argument("--sales_prices", help="Sales prices per unit for each product")
    parser.add_argument("--supplier_reliability", help="Supplier reliability for each product")
    parser.add_argument("--demand_correlation", type=float, help="Correlation between product demands")
    parser.add_argument("--write_results", action="store_true", help="Write detailed results to CSV files")
    parser.add_argument("--csv_dir", help="Directory for output CSV files")

    # Parse defaults from C++ file
    try:
        defaults = parse_cpp_params(parser.parse_args().cpp_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error parsing C++ file: {e}")
        sys.exit(1)

    # Update parser with defaults
    parser.set_defaults(
        np=defaults['np'],
        scenarios=defaults['scenarios'],
        time_periods=defaults['time_periods'],
        n_products=defaults['n_products'],
        demand_means=defaults['demand_means'],
        demand_stds=defaults['demand_stds'],
        initial_inventory=defaults['initial_inventory'],
        reorder_points=defaults['reorder_points'],
        order_quantities=defaults['order_quantities'],
        lead_time_means=defaults['lead_time_means'],
        lead_time_stds=defaults['lead_time_stds'],
        holding_costs=defaults['holding_costs'],
        stockout_costs=defaults['stockout_costs'],
        ordering_costs=defaults['ordering_costs'],
        sales_prices=defaults['sales_prices'],
        supplier_reliability=defaults['supplier_reliability'],
        demand_correlation=defaults['demand_correlation'],
        write_results=defaults['write_results'],
        csv_dir=defaults['csv_dir']
    )

    args = parser.parse_args()

    # Validate product-specific parameters
    csv_params = [
        args.demand_means, args.demand_stds, args.initial_inventory, args.reorder_points,
        args.order_quantities, args.lead_time_means, args.lead_time_stds, args.holding_costs,
        args.stockout_costs, args.ordering_costs, args.sales_prices, args.supplier_reliability
    ]
    try:
        for param in csv_params:
            validate_csv_list(param, args.n_products)
        if args.demand_correlation < -1.0 or args.demand_correlation > 1.0:
            raise ValueError("Demand correlation must be between -1.0 and 1.0")
        if args.np < 1:
            raise ValueError("Number of processes (--np) must be positive")
        if args.scenarios < 1:
            raise ValueError("Number of scenarios must be positive")
        if args.time_periods < 1:
            raise ValueError("Number of time periods must be positive")
        if args.n_products < 1:
            raise ValueError("Number of products must be positive")
    except ValueError as e:
        print(f"Validation error: {e}")
        sys.exit(1)

    success = True
    if args.compile:
        success = compile_cpp(source=args.cpp_file, output=args.binary)
        if not success:
            print("Compilation failed. Exiting.")
            sys.exit(1)

    Path(args.csv_dir).mkdir(parents=True, exist_ok=True)

    if success:
        success = run_job(
            binary=args.binary,
            num_processes=args.np,
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
            args.np, args.csv_dir, args.scenarios, args.time_periods, args.n_products
        )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()