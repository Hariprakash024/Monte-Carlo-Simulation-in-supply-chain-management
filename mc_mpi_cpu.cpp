#include <mpi.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <random>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>

static inline void split_csv(const std::string& s, std::vector<float>& out) {
    out.clear();
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stof(item));
    }
}

static inline float clamp_nonneg(float x) { return std::max(0.0f, x); }

class CorrelatedNormalGenerator {
private:
    std::mt19937& rng;
    std::normal_distribution<float> normal_dist{0.0f, 1.0f};
    std::vector<std::vector<float>> cholesky;
    int n_dim;

public:
    CorrelatedNormalGenerator(std::mt19937& generator, const std::vector<std::vector<float>>& correlation_matrix)
        : rng(generator), n_dim(correlation_matrix.size()) {
        cholesky.assign(n_dim, std::vector<float>(n_dim, 0.0f));
        
        // Optimized Cholesky decomposition
        for (int i = 0; i < n_dim; ++i) {
            float sum = 0.0f;
            for (int k = 0; k < i; ++k) sum += cholesky[i][k] * cholesky[i][k];
            cholesky[i][i] = std::sqrt(std::max(correlation_matrix[i][i] - sum, 0.0f));
            
            for (int j = i + 1; j < n_dim; ++j) {
                sum = 0.0f;
                for (int k = 0; k < i; ++k) sum += cholesky[j][k] * cholesky[i][k];
                cholesky[j][i] = (correlation_matrix[j][i] - sum) / cholesky[i][i];
            }
        }
    }
    
    void generate(std::vector<float>& output) {
        std::vector<float> z(n_dim);
        for (int i = 0; i < n_dim; ++i) z[i] = normal_dist(rng);
        
        for (int i = 0; i < n_dim; ++i) {
            output[i] = 0.0f;
            for (int j = 0; j <= i; ++j) output[i] += cholesky[i][j] * z[j];
        }
    }
};

struct ProductState {
    float inventory = 0.0f;
    float on_order = 0.0f;
    float arrival_time = -1.0f;
    float total_demand = 0.0f;
    float total_sales = 0.0f;
    float avg_inventory = 0.0f;
};

struct ScenarioResult {
    float total_cost = 0.0f;
    float total_revenue = 0.0f;
    float total_profit = 0.0f;
    float stockout_rate = 0.0f;
    float service_level = 0.0f;
    float inventory_turnover = 0.0f;
};

void simulate_scenario(
    long long scenario_global,
    int time_periods,
    const std::vector<float>& demand_means,
    const std::vector<float>& demand_stds,
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
    CorrelatedNormalGenerator& demand_generator, // Removed const
    ScenarioResult& result
) {
    const int n_products = demand_means.size();
    if (n_products > 32) throw std::runtime_error("Too many products");

    std::vector<ProductState> products(n_products);
    for (int p = 0; p < n_products; ++p) products[p].inventory = initial_inventory[p];

    std::vector<float> correlated_demand(n_products);
    std::normal_distribution<float> lead_time_normal(0.0f, 1.0f);
    float stockout_count = 0.0f;

    for (int t = 0; t < time_periods; ++t) {
        demand_generator.generate(correlated_demand);

        for (int p = 0; p < n_products; ++p) {
            auto& prod = products[p];
            prod.avg_inventory += prod.inventory;

            // Order arrivals
            if (prod.on_order > 0.0f && t >= prod.arrival_time) {
                if (uniform_dist(rng) <= supplier_reliability[p]) {
                    prod.inventory += prod.on_order;
                }
                prod.on_order = 0.0f;
                prod.arrival_time = -1.0f;
            }

            // Calculate demand
            float demand = clamp_nonneg(correlated_demand[p] * demand_stds[p] + demand_means[p]);
            prod.total_demand += demand;

            // Sales and inventory update
            float actual_sales = std::min(prod.inventory, demand);
            prod.inventory -= actual_sales;
            prod.total_sales += actual_sales;

            // Costs and revenue
            result.total_revenue += actual_sales * sales_prices[p];
            result.total_cost += prod.inventory * holding_costs[p];

            // Stockout penalty
            if (actual_sales < demand) {
                result.total_cost += (demand - actual_sales) * stockout_costs[p];
                stockout_count += 1.0f;
            }

            // Reorder decision
            if (prod.inventory + prod.on_order <= reorder_points[p]) {
                prod.on_order += order_quantities[p];
                float lt = std::max(1.0f, lead_time_normal(rng) * lead_time_stds[p] + lead_time_means[p]);
                prod.arrival_time = t + lt;
                result.total_cost += order_quantities[p] * ordering_costs[p];
            }
        }
    }

    // Calculate metrics
    for (int p = 0; p < n_products; ++p) {
        const auto& prod = products[p];
        result.service_level += prod.total_sales / (prod.total_demand > 0 ? prod.total_demand : 1.0f); // Avoid division by zero
        result.inventory_turnover += prod.total_sales / (prod.avg_inventory / time_periods);
    }
    
    result.service_level /= n_products;
    result.inventory_turnover /= n_products;
    result.stockout_rate = stockout_count / (time_periods * n_products);
    result.total_profit = result.total_revenue - result.total_cost;
}

struct Params {
    long long scenarios = 1'000'00;
    int time_periods = 55;
    int n_products = 4;
    std::vector<float> demand_means = {500.f, 300.f, 200.f, 400.f};
    std::vector<float> demand_stds = {100.f, 50.f, 30.f, 67.f};
    std::vector<float> initial_inventory = {1000.f, 800.f, 600.f, 450.f};
    std::vector<float> reorder_points = {200.f, 150.f, 100.f, 300.f};
    std::vector<float> order_quantities = {500.f, 400.f, 300.f, 250.f};
    std::vector<float> lead_time_means = {2.0f, 1.5f, 2.5f, 4.6f};
    std::vector<float> lead_time_stds = {0.5f, 0.3f, 0.7f, 0.6f};
    std::vector<float> holding_costs = {2.0f, 2.5f, 3.0f, 4.6f};
    std::vector<float> stockout_costs = {5.0f, 6.0f, 7.0f, 3.0f};
    std::vector<float> ordering_costs = {8.0f, 9.0f, 10.0f, 9.0f};
    std::vector<float> sales_prices = {10.0f, 12.0f, 15.0f, 13.0f};
    std::vector<float> supplier_reliability = {0.95f, 0.90f, 0.85f, 0.54f};
    float demand_correlation = 0.54f;
    bool write_results = false;
    std::string csv_dir = "results";
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
        else if (k == "--write_results") p.write_results = (std::stoi(argv[++i]) != 0);
        else if (k == "--csv_dir") next_s(p.csv_dir);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    try {
        Params par;
        parse_args(argc, argv, par);

        // Validate parameters
        std::vector<size_t> sizes = {
            par.demand_means.size(), par.demand_stds.size(), par.initial_inventory.size(),
            par.reorder_points.size(), par.order_quantities.size(), par.lead_time_means.size(),
            par.lead_time_stds.size(), par.holding_costs.size(), par.stockout_costs.size(),
            par.ordering_costs.size(), par.sales_prices.size(), par.supplier_reliability.size()
        };
        if (std::any_of(sizes.begin(), sizes.end(), [&](size_t s) { return s != par.n_products; })) {
            throw std::runtime_error("All product-specific parameter lists must match n_products");
        }

        // Create output directory
        if (world_rank == 0) std::filesystem::create_directories(par.csv_dir);

        // Partition scenarios
        long long base = par.scenarios / world_size;
        long long rem = par.scenarios % world_size;
        long long n_local = base + (world_rank < rem ? 1 : 0);
        long long start_global = world_rank * base + std::min<long long>(world_rank, rem);

        // Initialize RNG
        std::mt19937 rng(std::random_device{}() + world_rank);
        std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);

        // Create correlation matrix
        std::vector<std::vector<float>> correlation_matrix(par.n_products, std::vector<float>(par.n_products, par.demand_correlation));
        for (int i = 0; i < par.n_products; ++i) correlation_matrix[i][i] = 1.0f;
        CorrelatedNormalGenerator demand_generator(rng, correlation_matrix);

        // Run simulations
        std::vector<ScenarioResult> results(n_local);
        for (long long i = 0; i < n_local; ++i) {
            simulate_scenario(
                start_global + i, par.time_periods, par.demand_means, par.demand_stds,
                par.initial_inventory, par.reorder_points, par.order_quantities,
                par.lead_time_means, par.lead_time_stds, par.holding_costs,
                par.stockout_costs, par.ordering_costs, par.sales_prices,
                par.supplier_reliability, par.demand_correlation, rng, uniform_dist,
                demand_generator, results[i]
            );
        }

        // Aggregate local results
        double sums[6] = {0.0};
        for (const auto& res : results) {
            sums[0] += res.total_cost;
            sums[1] += res.total_revenue;
            sums[2] += res.total_profit;
            sums[3] += res.stockout_rate;
            sums[4] += res.service_level;
            sums[5] += res.inventory_turnover;
        }

        // Write summary
        std::ofstream sum_ofs(par.csv_dir + "/summary_rank" + std::to_string(world_rank) + ".csv");
        if (!sum_ofs) throw std::runtime_error("Failed to open summary file");
        sum_ofs << n_local << "," << sums[0] << "," << sums[1] << "," << sums[2] << ","
                << sums[3] << "," << sums[4] << "," << sums[5] << "\n";

        // Write detailed results if requested
        if (par.write_results) {
            std::ofstream ofs(par.csv_dir + "/results_rank" + std::to_string(world_rank) + ".csv");
            if (!ofs) throw std::runtime_error("Failed to open results file");
            ofs << "scenario,total_cost,total_revenue,profit,stockout_rate,service_level,inventory_turnover\n";
            for (long long i = 0; i < n_local; ++i) {
                const auto& res = results[i];
                ofs << (start_global + i) << "," << res.total_cost << "," << res.total_revenue << ","
                    << res.total_profit << "," << res.stockout_rate << "," 
                    << res.service_level << "," << res.inventory_turnover << "\n";
            }
        }
    }
    catch (const std::exception& e) {
        if (world_rank == 0) std::cerr << "Error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}