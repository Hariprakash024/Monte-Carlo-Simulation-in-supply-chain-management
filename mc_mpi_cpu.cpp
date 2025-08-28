#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <cmath>
#include <map>
#include <unordered_map>

static inline void split_csv(const std::string& s, std::vector<float>& out) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stof(item));
    }
}

static inline float clamp_nonneg(float x) { return x < 0.f ? 0.f : x; }

class CorrelatedNormalGenerator {
private:
    std::mt19937& rng;
    std::normal_distribution<float> normal_dist;
    std::vector<std::vector<float>> cholesky;
    int n_dim;
    
public:
    CorrelatedNormalGenerator(std::mt19937& generator, const std::vector<std::vector<float>>& correlation_matrix)
        : rng(generator), normal_dist(0.0f, 1.0f), n_dim(correlation_matrix.size()) {
        
        // Perform Cholesky decomposition
        cholesky.resize(n_dim, std::vector<float>(n_dim, 0.0f));
        for (int i = 0; i < n_dim; i++) {
            for (int j = 0; j <= i; j++) {
                float sum = 0.0f;
                for (int k = 0; k < j; k++) {
                    sum += cholesky[i][k] * cholesky[j][k];
                }
                
                if (i == j) {
                    cholesky[i][j] = std::sqrt(correlation_matrix[i][i] - sum);
                } else {
                    cholesky[i][j] = (correlation_matrix[i][j] - sum) / cholesky[j][j];
                }
            }
        }
    }
    
    void generate(std::vector<float>& output) {
        std::vector<float> z(n_dim);
        for (int i = 0; i < n_dim; i++) {
            z[i] = normal_dist(rng);
        }
        
        for (int i = 0; i < n_dim; i++) {
            output[i] = 0.0f;
            for (int j = 0; j <= i; j++) {
                output[i] += cholesky[i][j] * z[j];
            }
        }
    }
};

// Enhanced supply chain simulation with multiple products and correlated demand
void simulate_scenario(
    long long scenario_global,
    int time_periods,
    const std::vector<float>& demand_means,
    const std::vector<float>& demand_stds,
    int n_products,
    const std::vector<float>& initial_inventory,
    const std::vector<float>& reorder_points,
    const std::vector<float>& order_quantities,
    const std::vector<float>& lead_time_means,
    const std::vector<float>& lead_time_stds,
    const std::vector<float>& holding_costs,
    const std::vector<float>& stockout_costs,
    const std::vector<float>& ordering_costs,
    const std::vector<float>& sales_prices,
    const std::vector<float>& supplier_reliability,
    float demand_correlation,
    std::mt19937& rng,
    std::uniform_real_distribution<float>& uniform_dist,
    CorrelatedNormalGenerator& demand_generator,
    float& out_total_cost,
    float& out_total_revenue,
    float& out_total_profit,
    float& out_stockout_rate,
    float& out_service_level,
    float& out_inventory_turnover
) {
    const int MAX_PRODUCTS = 32;
    if (n_products > MAX_PRODUCTS) return;

    // Initialize product states
    std::vector<float> inventory(n_products);
    std::vector<float> on_order(n_products);
    std::vector<float> arrival_time(n_products);
    std::vector<float> total_demand(n_products, 0.0f);
    std::vector<float> total_sales(n_products, 0.0f);
    std::vector<float> avg_inventory(n_products, 0.0f);
    
    for (int p = 0; p < n_products; ++p) {
        inventory[p] = initial_inventory[p];
        on_order[p] = 0.0f;
        arrival_time[p] = -1.0f;
    }

    float total_cost = 0.0f;
    float total_revenue = 0.0f;
    float stockout_count = 0.0f;
    std::vector<float> correlated_demand(n_products);
    
    std::normal_distribution<float> lead_time_normal(0.0f, 1.0f);

    for (int t = 0; t < time_periods; ++t) {
        // Generate correlated demand
        demand_generator.generate(correlated_demand);
        
        for (int p = 0; p < n_products; ++p) {
            // Update average inventory tracking
            avg_inventory[p] += inventory[p];
            
            // Order arrivals (with supplier reliability check)
            if (on_order[p] > 0.0f && t >= arrival_time[p]) {
                // Check if supplier delivers (reliability factor)
                if (uniform_dist(rng) <= supplier_reliability[p]) {
                    inventory[p] += on_order[p];
                }
                on_order[p] = 0.0f;
                arrival_time[p] = -1.0f;
            }

            // Calculate demand with correlation
            float demand = clamp_nonneg(correlated_demand[p] * demand_stds[p] + demand_means[p]);
            total_demand[p] += demand;

            // Sales and inventory update
            float actual_sales = std::min(inventory[p], demand);
            inventory[p] -= actual_sales;
            total_sales[p] += actual_sales;

            // Revenue and costs
            total_revenue += actual_sales * sales_prices[p];
            total_cost += inventory[p] * holding_costs[p];  // Holding cost

            // Stockout penalty
            if (actual_sales < demand) {
                total_cost += (demand - actual_sales) * stockout_costs[p];
                stockout_count += 1.0f;
            }

            // Reorder decision with (s, Q) policy
            if (inventory[p] + on_order[p] <= reorder_points[p]) {
                float order_size = order_quantities[p];
                on_order[p] += order_size;
                
                // Calculate lead time with variability
                float lt = std::max(1.0f, lead_time_normal(rng) * lead_time_stds[p] + lead_time_means[p]);
                arrival_time[p] = t + lt;
                
                // Ordering cost
                total_cost += order_size * ordering_costs[p];
            }
        }
    }

    // Calculate performance metrics
    float service_level = 0.0f;
    float inventory_turnover = 0.0f;
    
    for (int p = 0; p < n_products; ++p) {
        service_level += total_sales[p] / total_demand[p];
        inventory_turnover += total_sales[p] / (avg_inventory[p] / time_periods);
    }
    
    service_level /= n_products;
    inventory_turnover /= n_products;
    
    float stockout_rate = stockout_count / (time_periods * n_products);
    float profit = total_revenue - total_cost;

    out_total_cost = total_cost;
    out_total_revenue = total_revenue;
    out_total_profit = profit;
    out_stockout_rate = stockout_rate;
    out_service_level = service_level;
    out_inventory_turnover = inventory_turnover;
}

struct Params {
    long long scenarios = 1000000;
    int time_periods = 52;
    int n_products = 3;
    std::vector<float> demand_means = {500.f, 300.f, 200.f};
    std::vector<float> demand_stds = {100.f, 50.f, 30.f};
    std::vector<float> initial_inventory = {1000.f, 800.f, 600.f};
    std::vector<float> reorder_points = {200.f, 150.f, 100.f};
    std::vector<float> order_quantities = {500.f, 400.f, 300.f};
    std::vector<float> lead_time_means = {2.0f, 1.5f, 2.5f};
    std::vector<float> lead_time_stds = {0.5f, 0.3f, 0.7f};
    std::vector<float> holding_costs = {2.0f, 2.5f, 3.0f};
    std::vector<float> stockout_costs = {5.0f, 6.0f, 7.0f};
    std::vector<float> ordering_costs = {8.0f, 9.0f, 10.0f};
    std::vector<float> sales_prices = {10.0f, 12.0f, 15.0f};
    std::vector<float> supplier_reliability = {0.95f, 0.90f, 0.85f};
    float demand_correlation = 0.3f;
    int write_results = 0;
    std::string csv_dir = "results";
};

static inline void parse_args(int argc, char** argv, Params& p) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](float& dst) { if (i + 1 < argc) dst = std::stof(argv[++i]); };
        auto next_ll = [&](long long& dst) { if (i + 1 < argc) dst = std::stoll(argv[++i]); };
        auto next_i = [&](int& dst) { if (i + 1 < argc) dst = std::stoi(argv[++i]); };
        auto next_csv = [&](std::vector<float>& dst) { if (i + 1 < argc) { dst.clear(); split_csv(argv[++i], dst); } };
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size = 1, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    Params par;
    parse_args(argc, argv, par);

    // Validate parameter sizes
    if (par.demand_means.size() != par.n_products || par.demand_stds.size() != par.n_products ||
        par.initial_inventory.size() != par.n_products || par.reorder_points.size() != par.n_products ||
        par.order_quantities.size() != par.n_products || par.lead_time_means.size() != par.n_products ||
        par.lead_time_stds.size() != par.n_products || par.holding_costs.size() != par.n_products ||
        par.stockout_costs.size() != par.n_products || par.ordering_costs.size() != par.n_products ||
        par.sales_prices.size() != par.n_products || par.supplier_reliability.size() != par.n_products) {
        
        if (world_rank == 0) {
            fprintf(stderr, "Error: All product-specific parameter lists must match n_products\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Partition scenarios
    long long base = par.scenarios / world_size;
    long long rem = par.scenarios % world_size;
    long long n_local = base + (world_rank < rem ? 1 : 0);
    long long start_global = world_rank * base + std::min<long long>(world_rank, rem);

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 rng(rd() + world_rank);
    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);

    // Create correlation matrix for demand
    std::vector<std::vector<float>> correlation_matrix(par.n_products, std::vector<float>(par.n_products, par.demand_correlation));
    for (int i = 0; i < par.n_products; i++) {
        correlation_matrix[i][i] = 1.0f;
    }
    
    CorrelatedNormalGenerator demand_generator(rng, correlation_matrix);

    // Local results
    std::vector<float> local_costs(n_local);
    std::vector<float> local_revenues(n_local);
    std::vector<float> local_profits(n_local);
    std::vector<float> local_stockout_rates(n_local);
    std::vector<float> local_service_levels(n_local);
    std::vector<float> local_inventory_turnovers(n_local);

    // Run simulations
    for (long long i = 0; i < n_local; ++i) {
        simulate_scenario(
            start_global + i,
            par.time_periods,
            par.demand_means,
            par.demand_stds,
            par.n_products,
            par.initial_inventory,
            par.reorder_points,
            par.order_quantities,
            par.lead_time_means,
            par.lead_time_stds,
            par.holding_costs,
            par.stockout_costs,
            par.ordering_costs,
            par.sales_prices,
            par.supplier_reliability,
            par.demand_correlation,
            rng,
            uniform_dist,
            demand_generator,
            local_costs[i],
            local_revenues[i],
            local_profits[i],
            local_stockout_rates[i],
            local_service_levels[i],
            local_inventory_turnovers[i]
        );
    }

    // Compute local sums
    double local_sum_cost = 0.0, local_sum_revenue = 0.0, local_sum_profit = 0.0;
    double local_sum_stockout = 0.0, local_sum_service = 0.0, local_sum_turnover = 0.0;
    
    for (long long i = 0; i < n_local; ++i) {
        local_sum_cost += local_costs[i];
        local_sum_revenue += local_revenues[i];
        local_sum_profit += local_profits[i];
        local_sum_stockout += local_stockout_rates[i];
        local_sum_service += local_service_levels[i];
        local_sum_turnover += local_inventory_turnovers[i];
    }

    // Write local summary
    std::ostringstream sum_fn;
    sum_fn << par.csv_dir << "/summary_rank" << world_rank << ".csv";
    std::ofstream sum_ofs(sum_fn.str());
    sum_ofs << n_local << "," << local_sum_cost << "," << local_sum_revenue << "," << local_sum_profit << ","
            << local_sum_stockout << "," << local_sum_service << "," << local_sum_turnover << "\n";

    // Optional: per-rank CSV output
    if (par.write_results) {
        std::ostringstream fn;
        fn << par.csv_dir << "/results_rank" << world_rank << ".csv";
        std::ofstream ofs(fn.str());
        ofs << "scenario,total_cost,total_revenue,profit,stockout_rate,service_level,inventory_turnover\n";
        for (long long i = 0; i < n_local; ++i) {
            ofs << (start_global + i) << "," << local_costs[i] << "," << local_revenues[i] << ","
                << local_profits[i] << "," << local_stockout_rates[i] << "," 
                << local_service_levels[i] << "," << local_inventory_turnovers[i] << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}