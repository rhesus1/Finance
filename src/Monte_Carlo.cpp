#include <stdexcept>
#include <random>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include "Monte_Carlo.h"

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

    #pragma omp parallel
    {
        std::mt19937 local_gen(std::random_device{}() + omp_get_thread_num());
        std::normal_distribution<double> local_dist(0.0, 1.0);
        double local_sum_payoffs = 0.0;

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
            local_sum_payoffs += payoff / 2.0;
        }

        #pragma omp critical
        sum_payoffs += local_sum_payoffs;
    }

    return exp(-r * T) * sum_payoffs / num_sims;
}

void Monte_Carlo::least_squares_regression(const std::vector<double>& S, const std::vector<double>& payoffs,
                                          std::vector<double>& continuation_values) {
    // Simple quadratic regression to estimate continuation value
    size_t n = S.size();
    continuation_values.resize(n, 0.0);
    vector<double> X(n * 3), Y(n);
    vector<size_t> in_the_money;

    // Identify in-the-money paths
    for (size_t i = 0; i < n; ++i) {
        if (payoffs[i] > 0) {
            in_the_money.push_back(i);
        }
    }

    if (in_the_money.empty()) {
        return; // No in-the-money paths, continuation value is 0
    }

    // Prepare regression data
    for (size_t idx : in_the_money) {
        double s = S[idx];
        X[idx * 3 + 0] = 1.0; // Constant term
        X[idx * 3 + 1] = s;    // Linear term
        X[idx * 3 + 2] = s * s; // Quadratic term
        Y[idx] = payoffs[idx];
    }

    // Solve least squares (simplified, assuming small number of in-the-money paths)
    // Using normal equations: (X^T X) beta = X^T Y
    double XT_X[3][3] = {{0}};
    double XT_Y[3] = {0};
    for (size_t idx : in_the_money) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                XT_X[i][j] += X[idx * 3 + i] * X[idx * 3 + j];
            }
            XT_Y[i] += X[idx * 3 + i] * Y[idx];
        }
    }

    // Solve 3x3 system (hard-coded Gaussian elimination for simplicity)
    double beta[3] = {0};
    if (abs(XT_X[0][0]) > 1e-10) {
        // Gaussian elimination
        for (int i = 0; i < 3; ++i) {
            for (int k = i + 1; k < 3; ++k) {
                double factor = XT_X[k][i] / XT_X[i][i];
                for (int j = i; j < 3; ++j) {
                    XT_X[k][j] -= factor * XT_X[i][j];
                }
                XT_Y[k] -= factor * XT_Y[i];
            }
        }
        // Back substitution
        for (int i = 2; i >= 0; --i) {
            double sum = XT_Y[i];
            for (int j = i + 1; j < 3; ++j) {
                sum -= XT_X[i][j] * beta[j];
            }
            beta[i] = sum / XT_X[i][i];
        }
    }

    // Compute continuation values
    for (size_t idx : in_the_money) {
        double s = S[idx];
        continuation_values[idx] = beta[0] + beta[1] * s + beta[2] * s * s;
        continuation_values[idx] = max(continuation_values[idx], 0.0);
    }
}

double Monte_Carlo::American_option_price(double S, double K, double T, double r, double sigma, bool is_call, int num_sims, int num_steps) {
    if (S <= 0 || K <= 0 || T <= 0 || sigma <= 0 || num_sims <= 0 || num_steps <= 0) {
        throw invalid_argument("Invalid inputs: S, K, T, sigma, num_sims, and num_steps must be positive.");
    }

    double dt = T / num_steps;
    double sqrt_dt = sqrt(dt);
    vector<vector<double>> paths(num_sims, vector<double>(num_steps + 1));
    vector<double> payoffs(num_sims);
    random_device rd;
    unsigned int base_seed = rd();

    // Simulate paths
    #pragma omp parallel
    {
        mt19937 local_gen(base_seed + omp_get_thread_num());
        normal_distribution<double> local_dist(0.0, 1.0);
        #pragma omp for
        for (int i = 0; i < num_sims; ++i) {
            paths[i][0] = S;
            for (int t = 1; t <= num_steps; ++t) {
                double Z = local_dist(local_gen);
                paths[i][t] = paths[i][t-1] * exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * Z);
            }
        }
    }

    // Backward induction using LSMC
    for (int i = 0; i < num_sims; ++i) {
        payoffs[i] = is_call ? max(paths[i][num_steps] - K, 0.0) : max(K - paths[i][num_steps], 0.0);
    }

    for (int t = num_steps - 1; t >= 0; --t) {
        vector<double> S_t(num_sims), continuation_values;
        for (int i = 0; i < num_sims; ++i) {
            S_t[i] = paths[i][t];
        }
        least_squares_regression(S_t, payoffs, continuation_values);

        for (int i = 0; i < num_sims; ++i) {
            double exercise_value = is_call ? max(paths[i][t] - K, 0.0) : max(K - paths[i][t], 0.0);
            double continuation_value = (t == num_steps - 1) ? 0.0 : continuation_values[i];
            if (exercise_value > continuation_value) {
                payoffs[i] = exercise_value;
            } else {
                payoffs[i] = payoffs[i] * exp(-r * dt);
            }
        }
    }

    double sum_payoffs = 0.0;
    for (int i = 0; i < num_sims; ++i) {
        sum_payoffs += payoffs[i];
    }

    return sum_payoffs / num_sims;
}

double Monte_Carlo::Heston_American_option_price(double S0, double K, double T, double r, double v0, double kappa,
                                                double theta, double xi, double rho, bool is_call, int num_sims, int num_steps) {
    if (S0 <= 0 || K <= 0 || T <= 0 || v0 < 0 || kappa <= 0 || theta <= 0 || xi <= 0 || num_sims <= 0 || num_steps <= 0) {
        throw invalid_argument("Invalid inputs: S0, K, T, kappa, theta, xi, num_sims, and num_steps must be positive; v0 must be non-negative.");
    }
    if (rho < -1.0 || rho > 1.0) {
        throw invalid_argument("Invalid input: rho must be between -1 and 1.");
    }

    double dt = T / num_steps;
    double sqrt_dt = sqrt(dt);
    vector<vector<double>> paths(num_sims, vector<double>(num_steps + 1));
    vector<vector<double>> variances(num_sims, vector<double>(num_steps + 1));
    vector<double> payoffs(num_sims);
    random_device rd;
    unsigned int base_seed = rd();

    // Simulate paths
    #pragma omp parallel
    {
        mt19937 local_gen(base_seed + omp_get_thread_num());
        normal_distribution<double> local_dist(0.0, 1.0);
        #pragma omp for
        for (int i = 0; i < num_sims; ++i) {
            paths[i][0] = S0;
            variances[i][0] = v0;
            for (int t = 1; t <= num_steps; ++t) {
                double z1 = local_dist(local_gen);
                double z2 = rho * z1 + sqrt(1 - rho * rho) * local_dist(local_gen);
                double v = variances[i][t-1];
                double sqrt_v = sqrt(max(v, 0.0));
                v = v + kappa * (theta - v) * dt + xi * sqrt_v * sqrt_dt * z2 + 0.25 * xi * xi * dt * (z2 * z2 - 1);
                variances[i][t] = max(v, 0.0);
                paths[i][t] = paths[i][t-1] * exp((r - 0.5 * variances[i][t]) * dt + sqrt_v * sqrt_dt * z1);
            }
        }
    }

    // Backward induction using LSMC
    for (int i = 0; i < num_sims; ++i) {
        payoffs[i] = is_call ? max(paths[i][num_steps] - K, 0.0) : max(K - paths[i][num_steps], 0.0);
    }

    for (int t = num_steps - 1; t >= 0; --t) {
        vector<double> S_t(num_sims), continuation_values;
        for (int i = 0; i < num_sims; ++i) {
            S_t[i] = paths[i][t];
        }
        least_squares_regression(S_t, payoffs, continuation_values);

        for (int i = 0; i < num_sims; ++i) {
            double exercise_value = is_call ? max(paths[i][t] - K, 0.0) : max(K - paths[i][t], 0.0);
            double continuation_value = (t == num_steps - 1) ? 0.0 : continuation_values[i];
            if (exercise_value > continuation_value) {
                payoffs[i] = exercise_value;
            } else {
                payoffs[i] = payoffs[i] * exp(-r * dt);
            }
        }
    }

    double sum_payoffs = 0.0;
    for (int i = 0; i < num_sims; ++i) {
        sum_payoffs += payoffs[i];
    }

    return sum_payoffs / num_sims;
}