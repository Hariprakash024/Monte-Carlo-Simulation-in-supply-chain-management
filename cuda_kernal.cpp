#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cmath>

static inline void split_csv(const std::string& s, std::vector<float>& out) {
    out.clear();
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stof(item));
    }
}

static inline __host__ __device__ float clamp_nonneg(float x) { return fmaxf(0.0f, x); }

struct Params {
    long long scenarios = 1'000'000;
    int time_periods = 52;
    int n_products = 3;
    std::vector<float> demand_means;
    std::vector<float> demand_stds;
    std::vector<float> initial_inventory;
    std::vector<float> reorder_points;
    std::vector<float> order_quantities;
    std::vector<float> lead_time_means;
    std::vector<float> lead_time_stds;
    std::vector<float> holding_costs;
    std::vector<float> stockout_costs;
    std::vector<float> ordering_costs;
    std::vector<float> sales_prices;
    std::vector<float> supplier_reliability;
    float demand_correlation = 0.3f;
    int write_results = 0;
    std::string csv_dir = "results";

    Params() {
        demand_means.reserve(32);
        demand_stds.reserve(32);
        initial_inventory.reserve(32);
        reorder_points.reserve(32);
        order_quantities.reserve(32);
        lead_time_means.reserve(32);
        lead_time_stds.reserve(32);
        holding_costs.reserve(32);
        stockout_costs.reserve(32);
        ordering_costs.reserve(32);
        sales_prices.reserve(32);
        supplier_reliability.reserve(32);
    }
};

static inline void parse_args(int argc, char** argv, Params& p) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](float& dst) { if (i + 1 < argc) dst = std::stof(argv[++i]); };
        auto next_ll = [&](long long& dst) { if (i + 1 < argc) dst = std::stoll(argv[++i]); };
        auto next_i = [&](int& dst) { if (i + 1 < argc) dst = std::stoi(argv[++i]); };
        auto next_csv = [&](std::vector<float>& dst) { if (i + 1 < argc) split_csv(argv[++i], dst); };
        auto next_s = [&](std::string& dst) { if (i + 1 < argc) dst = argv[++i]; };

        if (k == "--scenarios") next_ll(p.scenarios);
        else if (k == "--time_periods") next_i(p.time_periods);
        else if (k == "--n_products") next_i(p.n_products);
        else if (k == "--demand_means") next_csv(p.demand_means);
        else if (k == "--demand_stds") next_csv(p.demand_stds);
        else if (k == "--initial_inventory") next_csv(p.initial_inventory);
        else if (k == "--reorder_points") next_csv(p.reorder_points);
        else if (k == "--order_quantities") next_csv(p.order_quantities);
        else if (k == "--lead_time_means") next_csv(p.lead_time_means);
        else if (k == "--lead_time_stds") next_csv(p.lead_time_stds);
        else if (k == "--holding_costs") next_csv(p.holding_costs);
        else if (k == "--stockout_costs") next_csv(p.stockout_costs);
        else if (k == "--ordering_costs") next_csv(p.ordering_costs);
        else if (k == "--sales_prices") next_csv(p.sales_prices);
        else if (k == "--supplier_reliability") next_csv(p.supplier_reliability);
        else if (k == "--demand_correlation") next(p.demand_correlation);
        else if (k == "--write_results") next_i(p.write_results);
        else if (k == "--csv_dir") next_s(p.csv_dir);
    }
}

// CUDA kernel for generating correlated normal random variables
__global__ void generate_correlated_normal(
    curandState* states,
    float* cholesky,
    float* output,
    int n_products,
    long long scenario_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = states[idx];
    
    float z[32];
    for (int i = 0; i < n_products; ++i) {
        z[i] = curand_normal(&local_state);
    }
    
    for (int i = 0; i < n_products; ++i) {
        float sum = 0.0f;
        for (int j = 0; j <= i; ++j) {
            sum += cholesky[i * n_products + j] * z[j];
        }
        output[idx * n_products + i] = sum;
    }
    
    states[idx] = local_state;
}

// CUDA kernel for supply chain simulation
__global__ void simulate_scenario_kernel(
    long long scenario_start,
    int time_periods,
    float* demand_means,
    float* demand_stds,
    int n_products,
    float* initial_inventory,
    float* reorder_points,
    float* order_quantities,
    float* lead_time_means,
    float* lead_time_stds,
    float* holding_costs,
    float* stockout_costs,
    float* ordering_costs,
    float* sales_prices,
    float* supplier_reliability,
    float* correlated_demand,
    curandState* states,
    float* out_total_cost,
    float* out_total_revenue,
    float* out_total_profit,
    float* out_stockout_rate,
    float* out_service_level,
    float* out_inventory_turnover
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = states[idx];
    
    constexpr int MAX_PRODUCTS = 32;
    if (n_products > MAX_PRODUCTS) return;

    float inventory[32];
    float on_order[32] = {0.0f};
    float arrival_time[32] = {-1.0f};
    float total_demand[32] = {0.0f};
    float total_sales[32] = {0.0f};
    float avg_inventory[32] = {0.0f};

    for (int p = 0; p < n_products; ++p) {
        inventory[p] = initial_inventory[p];
    }

    float total_cost = 0.0f, total_revenue = 0.0f, stockout_count = 0.0f;

    for (int t = 0; t < time_periods; ++t) {
        for (int p = 0; p < n_products; ++p) {
            avg_inventory[p] += inventory[p];

            if (on_order[p] > 0.0f && t >= arrival_time[p]) {
                if (curand_uniform(&local_state) <= supplier_reliability[p]) {
                    inventory[p] += on_order[p];
                }
                on_order[p] = 0.0f;
                arrival_time[p] = -1.0f;
            }

            float demand = clamp_nonneg(correlated_demand[idx * n_products + p] * demand_stds[p] + demand_means[p]);
            total_demand[p] += demand;

            float actual_sales = fminf(inventory[p], demand);
            inventory[p] -= actual_sales;
            total_sales[p] += actual_sales;

            total_revenue += actual_sales * sales_prices[p];
            total_cost += inventory[p] * holding_costs[p];

            if (actual_sales < demand) {
                total_cost += (demand - actual_sales) * stockout_costs[p];
                stockout_count += 1.0f;
            }

            if (inventory[p] + on_order[p] <= reorder_points[p]) {
                float order_size = order_quantities[p];
                on_order[p] += order_size;
                float lt = fmaxf(1.0f, curand_normal(&local_state) * lead_time_stds[p] + lead_time_means[p]);
                arrival_time[p] = t + lt;
                total_cost += order_size * ordering_costs[p];
            }
        }
    }

    float service_level = 0.0f, inventory_turnover = 0.0f;
    for (int p = 0; p < n_products; ++p) {
        service_level += total_demand[p] > 0 ? total_sales[p] / total_demand[p] : 0.0f;
        inventory_turnover += avg_inventory[p] > 0 ? total_sales[p] / (avg_inventory[p] / time_periods) : 0.0f;
    }

    service_level /= n_products;
    inventory_turnover /= n_products;
    out_total_cost[idx] = total_cost;
    out_total_revenue[idx] = total_revenue;
    out_total_profit[idx] = total_revenue - total_cost;
    out_stockout_rate[idx] = stockout_count / (time_periods * n_products);
    out_service_level[idx] = service_level;
    out_inventory_turnover[idx] = inventory_turnover;

    states[idx] = local_state;
}

__global__ void setup_curand(curandState* states, unsigned long long seed, long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed + idx, idx, 0, &states[idx]);
    }
}

int main(int argc, char** argv) {
    Params par;
    parse_args(argc, argv, par);

    if (par.demand_means.size() != par.n_products || par.demand_stds.size() != par.n_products ||
        par.initial_inventory.size() != par.n_products || par.reorder_points.size() != par.n_products ||
        par.order_quantities.size() != par.n_products || par.lead_time_means.size() != par.n_products ||
        par.lead_time_stds.size() != par.n_products || par.holding_costs.size() != par.n_products ||
        par.stockout_costs.size() != par.n_products || par.ordering_costs.size() != par.n_products ||
        par.sales_prices.size() != par.n_products || par.supplier_reliability.size() != par.n_products) {
        fprintf(stderr, "Error: All product-specific parameter lists must match n_products\n");
        return 1;
    }

    // Compute Cholesky decomposition for correlation matrix
    std::vector<float> cholesky(par.n_products * par.n_products, 0.0f);
    std::vector<std::vector<float>> correlation_matrix(par.n_products, std::vector<float>(par.n_products, par.demand_correlation));
    for (int i = 0; i < par.n_products; ++i) {
        correlation_matrix[i][i] = 1.0f;
    }
    for (int i = 0; i < par.n_products; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < j; ++k) {
                sum += cholesky[i * par.n_products + k] * cholesky[j * par.n_products + k];
            }
            cholesky[i * par.n_products + j] = (i == j) ? sqrtf(correlation_matrix[i][j] - sum) : (correlation_matrix[i][j] - sum) / cholesky[j * par.n_products + j];
        }
    }

    // Allocate device memory
    float *d_demand_means, *d_demand_stds, *d_initial_inventory, *d_reorder_points, *d_order_quantities;
    float *d_lead_time_means, *d_lead_time_stds, *d_holding_costs, *d_stockout_costs, *d_ordering_costs;
    float *d_sales_prices, *d_supplier_reliability, *d_cholesky, *d_correlated_demand;
    float *d_total_cost, *d_total_revenue, *d_total_profit, *d_stockout_rate, *d_service_level, *d_inventory_turnover;
    curandState *d_states;

    cudaMalloc(&d_demand_means, par.n_products * sizeof(float));
    cudaMalloc(&d_demand_stds, par.n_products * sizeof(float));
    cudaMalloc(&d_initial_inventory, par.n_products * sizeof(float));
    cudaMalloc(&d_reorder_points, par.n_products * sizeof(float));
    cudaMalloc(&d_order_quantities, par.n_products * sizeof(float));
    cudaMalloc(&d_lead_time_means, par.n_products * sizeof(float));
    cudaMalloc(&d_lead_time_stds, par.n_products * sizeof(float));
    cudaMalloc(&d_holding_costs, par.n_products * sizeof(float));
    cudaMalloc(&d_stockout_costs, par.n_products * sizeof(float));
    cudaMalloc(&d_ordering_costs, par.n_products * sizeof(float));
    cudaMalloc(&d_sales_prices, par.n_products * sizeof(float));
    cudaMalloc(&d_supplier_reliability, par.n_products * sizeof(float));
    cudaMalloc(&d_cholesky, par.n_products * par.n_products * sizeof(float));
    cudaMalloc(&d_correlated_demand, par.scenarios * par.n_products * sizeof(float));
    cudaMalloc(&d_total_cost, par.scenarios * sizeof(float));
    cudaMalloc(&d_total_revenue, par.scenarios * sizeof(float));
    cudaMalloc(&d_total_profit, par.scenarios * sizeof(float));
    cudaMalloc(&d_stockout_rate, par.scenarios * sizeof(float));
    cudaMalloc(&d_service_level, par.scenarios * sizeof(float));
    cudaMalloc(&d_inventory_turnover, par.scenarios * sizeof(float));
    cudaMalloc(&d_states, par.scenarios * sizeof(curandState));

    // Copy data to device
    cudaMemcpy(d_demand_means, par.demand_means.data(), par.n_products * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_demand_stds, par.demand_stds.data(), par.n_products * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_initial_inventory, par.initial_inventory.data(), par.n_products * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reorder_points, par.reorder_points.data(), par.n_products * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_order_quantities, par.order_quantities.data(), par.n_products * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lead_time_means, par.lead_time_means.data(), par.n_products * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lead_time_stds, par.lead_time_stds.data(), par.n_products * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_holding_costs, par.holding_costs.data(), par.n_products * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stockout_costs, par.stockout_costs.data(), par.n_products * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ordering_costs, par.ordering_costs.data(), par.n_products * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sales_prices, par.sales_prices.data(), par.n_products * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_supplier_reliability, par.supplier_reliability.data(), par.n_products * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cholesky, cholesky.data(), par.n_products * par.n_products * sizeof(float), cudaMemcpyHostToDevice);

    // Setup cuRAND states
    int threads_per_block = 256;
    int blocks = (par.scenarios + threads_per_block - 1) / threads_per_block;
    setup_curand<<<blocks, threads_per_block>>>(d_states, time(NULL), par.scenarios);
    cudaDeviceSynchronize();

    // Generate correlated demands
    generate_correlated_normal<<<blocks, threads_per_block>>>(d_states, d_cholesky, d_correlated_demand, par.n_products, par.scenarios);
    cudaDeviceSynchronize();

    // Run simulation
    simulate_scenario_kernel<<<blocks, threads_per_block>>>(
        0, par.time_periods, d_demand_means, d_demand_stds, par.n_products,
        d_initial_inventory, d_reorder_points, d_order_quantities,
        d_lead_time_means, d_lead_time_stds, d_holding_costs,
        d_stockout_costs, d_ordering_costs, d_sales_prices,
        d_supplier_reliability, d_correlated_demand, d_states,
        d_total_cost, d_total_revenue, d_total_profit,
        d_stockout_rate, d_service_level, d_inventory_turnover
    );
    cudaDeviceSynchronize();

    // Copy results back to host
    std::vector<float> total_cost(par.scenarios), total_revenue(par.scenarios), total_profit(par.scenarios);
    std::vector<float> stockout_rate(par.scenarios), service_level(par.scenarios), inventory_turnover(par.scenarios);
    cudaMemcpy(total_cost.data(), d_total_cost, par.scenarios * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(total_revenue.data(), d_total_revenue, par.scenarios * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(total_profit.data(), d_total_profit, par.scenarios * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(stockout_rate.data(), d_stockout_rate, par.scenarios * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(service_level.data(), d_service_level, par.scenarios * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(inventory_turnover.data(), d_inventory_turnover, par.scenarios * sizeof(float), cudaMemcpyDeviceToHost);

    // Aggregate results
    double sums[6] = {0.0};
    for (long long i = 0; i < par.scenarios; ++i) {
        sums[0] += total_cost[i];
        sums[1] += total_revenue[i];
        sums[2] += total_profit[i];
        sums[3] += stockout_rate[i];
        sums[4] += service_level[i];
        sums[5] += inventory_turnover[i];
    }

    // Write summary
    std::ostringstream sum_fn;
    sum_fn << par.csv_dir << "/summary.csv";
    std::ofstream sum_ofs(sum_fn.str());
    sum_ofs << par.scenarios << "," << sums[0] << "," << sums[1] << "," << sums[2] << ","
            << sums[3] << "," << sums[4] << "," << sums[5] << "\n";
    sum_ofs.close();

    // Write detailed results if requested
    if (par.write_results) {
        std::ostringstream fn;
        fn << par.csv_dir << "/results.csv";
        std::ofstream ofs(fn.str());
        ofs << "scenario,total_cost,total_revenue,profit,stockout_rate,service_level,inventory_turnover\n";
        for (long long i = 0; i < par.scenarios; ++i) {
            ofs << i << "," << total_cost[i] << "," << total_revenue[i] << ","
                << total_profit[i] << "," << stockout_rate[i] << ","
                << service_level[i] << "," << inventory_turnover[i] << "\n";
        }
        ofs.close();
    }

    // Free device memory
    cudaFree(d_demand_means);
    cudaFree(d_demand_stds);
    cudaFree(d_initial_inventory);
    cudaFree(d_reorder_points);
    cudaFree(d_order_quantities);
    cudaFree(d_lead_time_means);
    cudaFree(d_lead_time_stds);
    cudaFree(d_holding_costs);
    cudaFree(d_stockout_costs);
    cudaFree(d_ordering_costs);
    cudaFree(d_sales_prices);
    cudaFree(d_supplier_reliability);
    cudaFree(d_cholesky);
    cudaFree(d_correlated_demand);
    cudaFree(d_total_cost);
    cudaFree(d_total_revenue);
    cudaFree(d_total_profit);
    cudaFree(d_stockout_rate);
    cudaFree(d_service_level);
    cudaFree(d_inventory_turnover);
    cudaFree(d_states);

    return 0;
}