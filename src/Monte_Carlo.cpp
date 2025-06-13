#include "Monte_Carlo.h"
#include <stdexcept>
#include <random>
#include <omp.h>
#include <algorithm>

using namespace std;

Monte_Carlo::Monte_Carlo() {}
Monte_Carlo::~Monte_Carlo() {}

double Monte_Carlo::option_price(double S, double K, double T, double r, double sigma, bool is_call, int num_sims) {
    if (S <= 0 || K <= 0 || T <= 0 || sigma <= 0 || num_sims <= 0) {
        throw invalid_argument("Invalid inputs: S, K, T, sigma, and num_sims must be positive.");
    }

    random_device rd;
    unsigned int base_seed = rd();
    normal_distribution<double> dist(0.0, 1.0);
    double sum_payoffs = 0.0;

    #pragma omp parallel reduction(+:sum_payoffs)
    {
        unsigned int seed = base_seed + omp_get_thread_num();
        mt19937 local_gen(seed);
        normal_distribution<double> local_dist(0.0, 1.0);
        #pragma omp for
        for (int i = 0; i < num_sims; ++i) {
            double Z = local_dist(local_gen);
            double ST = S * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
            double payoff = is_call ? max(ST - K, 0.0) : max(K - ST, 0.0);
            ST = S * exp((r - 0.5 * sigma * sigma) * T - sigma * sqrt(T) * Z);
            payoff += is_call ? max(ST - K, 0.0) : max(K - ST, 0.0);
            sum_payoffs += payoff / 2.0;
        }
    }

    return exp(-r * T) * sum_payoffs / num_sims;
}

double Monte_Carlo::Heston_option_price(double S0, double K, double T, double r, double v0, double kappa,
                                       double theta, double xi, double rho, bool is_call, int num_sims, int num_steps) {
    if (S0 <= 0 || K <= 0 || T <= 0 || v0 < 0 || kappa <= 0 || theta <= 0 || xi <= 0 || num_sims <= 0 || num_steps <= 0) {
        throw invalid_argument("Invalid inputs: S0, K, T, kappa, theta, xi, num_sims, and num_steps must be positive; v0 must be non-negative.");
    }
    if (rho < -1.0 || rho > 1.0) {
        throw invalid_argument("Invalid input: rho must be between -1 and 1.");
    }

    random_device rd;
    unsigned int base_seed = rd();
    normal_distribution<double> dist(0.0, 1.0);
    double dt = T / num_steps;
    double sqrt_dt = sqrt(dt);
    double sum_payoffs = 0.0;

    #pragma omp parallel reduction(+:sum_payoffs)
    {
        unsigned int seed = base_seed + omp_get_thread_num();
        mt19937 local_gen(seed);
        normal_distribution<double> local_dist(0.0, 1.0);
        #pragma omp for
        for (int i = 0; i < num_sims; i++) {
            double S = S0, S_anti = S0;
            double v = v0, v_anti = v0;
            for (int t = 0; t < num_steps; t++) {
                double z1 = local_dist(local_gen);
                double z2 = rho * z1 + sqrt(1 - rho * rho) * local_dist(local_gen);
                double sqrt_v = sqrt(max(v, 0.0));
                v = v + kappa * (theta - v) * dt + xi * sqrt_v * sqrt_dt * z2 + 0.25 * xi * xi * dt * (z2 * z2 - 1);
                v = max(v, 0.0);
                S *= exp((r - 0.5 * v) * dt + sqrt_v * sqrt_dt * z1);
                sqrt_v = sqrt(max(v_anti, 0.0));
                v_anti = v_anti + kappa * (theta - v_anti) * dt + xi * sqrt_v * sqrt_dt * (-z2) + 0.25 * xi * xi * dt * (z2 * z2 - 1);
                v_anti = max(v_anti, 0.0);
                S_anti *= exp((r - 0.5 * v_anti) * dt + sqrt_v * sqrt_dt * (-z1));
            }
            double payoff = is_call ? max(S - K, 0.0) : max(K - S, 0.0);
            payoff += is_call ? max(S_anti - K, 0.0) : max(K - S_anti, 0.0);
            sum_payoffs += payoff / 2.0;
        }
    }

    return exp(-r * T) * sum_payoffs / num_sims;
}