#include <stdexcept>
#include <random>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include "Monte_Carlo.h"

using namespace std;

// Default constructor
Monte_Carlo::Monte_Carlo() {}
// Default destructor
Monte_Carlo::~Monte_Carlo() {}

// Computes option price using Black-Scholes Monte Carlo simulation with antithetic variates
double Monte_Carlo::option_price(double S, double K, double T, double r, double sigma, bool is_call, int num_sims) {
    // Input: S (spot price), K (strike price), T (time to maturity), r (risk-free rate),
    //        sigma (volatility), is_call (true for call, false for put), num_sims (number of simulations)
    // Output: Estimated option price
    // Logic: Simulates stock price paths using Black-Scholes model, applies antithetic variates
    //        to reduce variance, and computes average discounted payoff
    if (S <= 0 || K <= 0 || T <= 0 || sigma <= 0 || num_sims <= 0) {
        // Validate inputs to ensure positive values
        throw invalid_argument("Invalid inputs: S, K, T, sigma, and num_sims must be positive.");
    }

    random_device rd; // Random seed generator
    unsigned int base_seed = rd(); // Base seed for random number generation
    normal_distribution<double> dist(0.0, 1.0); // Standard normal distribution (mean 0, std 1)
    double sum_payoffs = 0.0; // Accumulator for option payoffs

    #pragma omp parallel reduction(+:sum_payoffs)
    {
        // Parallelize simulation with OpenMP, summing payoffs across threads
        unsigned int seed = base_seed + omp_get_thread_num(); // Unique seed per thread
        mt19937 local_gen(seed); // Mersenne Twister random number generator
        normal_distribution<double> local_dist(0.0, 1.0); // Thread-local normal distribution
        #pragma omp for
        for (int i = 0; i < num_sims; ++i) {
            // Simulate one path
            double Z = local_dist(local_gen); // Random standard normal variable
            // Compute stock price at maturity using Black-Scholes dynamics
            double ST = S * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
            // Compute payoff for call or put option
            double payoff = is_call ? max(ST - K, 0.0) : max(K - ST, 0.0);
            // Antithetic variate: compute stock price with opposite random variable
            ST = S * exp((r - 0.5 * sigma * sigma) * T - sigma * sqrt(T) * Z);
            // Add antithetic payoff to reduce variance
            payoff += is_call ? max(ST - K, 0.0) : max(K - ST, 0.0);
            // Average the payoffs and add to total
            sum_payoffs += payoff / 2.0;
        }
    }

    // Compute discounted average payoff to get option price
    return exp(-r * T) * sum_payoffs / num_sims;
}

// Computes European option price using Heston model Monte Carlo simulation with antithetic variates
double Monte_Carlo::Heston_option_price(double S0, double K, double T, double r, double v0, double kappa,
                                       double theta, double xi, double rho, bool is_call, int num_sims, int num_steps) {
    // Input: S0 (spot price), K (strike price), T (time to maturity), r (risk-free rate),
    //        v0 (initial variance), kappa (mean reversion rate), theta (long-term variance),
    //        xi (volatility of variance), rho (correlation), is_call (true for call, false for put),
    //        num_sims (number of simulations), num_steps (time steps)
    // Output: Estimated option price
    // Logic: Simulates stock and variance paths using Heston model with Euler discretization,
    //        applies antithetic variates, and computes average discounted payoff
    if (S0 <= 0 || K <= 0 || T <= 0 || v0 < 0 || kappa <= 0 || theta <= 0 || xi <= 0 || num_sims <= 0 || num_steps <= 0) {
        // Validate inputs for positive/non-negative values
        throw invalid_argument("Invalid inputs: S0, K, T, kappa, theta, xi, num_sims, and num_steps must be positive; v0 must be non-negative.");
    }
    if (rho < -1.0 || rho > 1.0) {
        // Validate correlation coefficient
        throw invalid_argument("Invalid input: rho must be between -1 and 1.");
    }

    random_device rd; // Random seed generator
    unsigned int base_seed = rd(); // Base seed for random number generation
    normal_distribution<double> dist(0.0, 1.0); // Standard normal distribution
    double dt = T / num_steps; // Time step size
    double sqrt_dt = sqrt(dt); // Square root of time step for volatility scaling
    double sum_payoffs = 0.0; // Accumulator for option payoffs

    #pragma omp parallel
    {
        // Parallelize simulation with OpenMP
        std::mt19937 local_gen(std::random_device{}() + omp_get_thread_num()); // Thread-local random number generator
        std::normal_distribution<double> local_dist(0.0, 1.0); // Thread-local normal distribution
        double local_sum_payoffs = 0.0; // Thread-local payoff accumulator

        #pragma omp for
        for (int i = 0; i < num_sims; i++) {
            // Simulate one path
            double S = S0, S_anti = S0; // Initial stock prices (regular and antithetic)
            double v = v0, v_anti = v0; // Initial variances (regular and antithetic)
            for (int t = 0; t < num_steps; t++) {
                // Generate correlated random variables
                double z1 = local_dist(local_gen); // Random normal for stock price
                double z2 = rho * z1 + sqrt(1 - rho * rho) * local_dist(local_gen); // Correlated random normal for variance
                double sqrt_v = sqrt(max(v, 0.0)); // Square root of variance, ensure non-negative
                // Update variance using Euler discretization with Milstein correction
                v = v + kappa * (theta - v) * dt + xi * sqrt_v * sqrt_dt * z2 + 0.25 * xi * xi * dt * (z2 * z2 - 1);
                v = max(v, 0.0); // Ensure non-negative variance
                // Update stock price using Euler discretization
                S *= exp((r - 0.5 * v) * dt + sqrt_v * sqrt_dt * z1);
                sqrt_v = sqrt(max(v_anti, 0.0)); // Square root of antithetic variance
                // Update antithetic variance with opposite random variable
                v_anti = v_anti + kappa * (theta - v_anti) * dt + xi * sqrt_v * sqrt_dt * (-z2) + 0.25 * xi * xi * dt * (z2 * z2 - 1);
                v_anti = max(v_anti, 0.0); // Ensure non-negative antithetic variance
                // Update antithetic stock price
                S_anti *= exp((r - 0.5 * v_anti) * dt + sqrt_v * sqrt_dt * (-z1));
            }
            // Compute payoffs for regular and antithetic paths
            double payoff = is_call ? max(S - K, 0.0) : max(K - S, 0.0);
            payoff += is_call ? max(S_anti - K, 0.0) : max(K - S_anti, 0.0);
            // Average payoffs and add to thread-local sum
            local_sum_payoffs += payoff / 2.0;
        }

        #pragma omp critical
        // Combine thread-local payoffs into global sum
        sum_payoffs += local_sum_payoffs;
    }

    // Compute discounted average payoff to get option price
    return exp(-r * T) * sum_payoffs / num_sims;
}

// Performs least squares regression to estimate continuation values for American options
void Monte_Carlo::least_squares_regression(const std::vector<double>& S, const std::vector<double>& payoffs,
                                          std::vector<double>& continuation_values) {
    // Input: S (stock prices), payoffs (option payoffs), continuation_values (output vector for continuation values)
    // Output: None (populates continuation_values with regression results)
    // Logic: Uses quadratic regression on in-the-money paths to estimate continuation values
    //        via least squares, solving normal equations with Gaussian elimination
    size_t n = S.size(); // Number of paths
    continuation_values.resize(n, 0.0); // Initialize output vector
    vector<double> X(n * 3), Y(n); // Regression design matrix and response vector
    vector<size_t> in_the_money; // Indices of in-the-money paths

    // Identify in-the-money paths (where payoff > 0)
    for (size_t i = 0; i < n; ++i) {
        if (payoffs[i] > 0) {
            in_the_money.push_back(i);
        }
    }

    if (in_the_money.empty()) {
        // No in-the-money paths, return zero continuation values
        return;
    }

    // Prepare regression data: constant, linear, and quadratic terms
    for (size_t idx : in_the_money) {
        double s = S[idx]; // Stock price
        X[idx * 3 + 0] = 1.0; // Constant term
        X[idx * 3 + 1] = s;    // Linear term
        X[idx * 3 + 2] = s * s; // Quadratic term
        Y[idx] = payoffs[idx]; // Payoff as response
    }

    // Compute normal equations: (X^T X) beta = X^T Y
    double XT_X[3][3] = {{0}}; // Normal matrix
    double XT_Y[3] = {0}; // Right-hand side vector
    for (size_t idx : in_the_money) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                // Accumulate X^T X
                XT_X[i][j] += X[idx * 3 + i] * X[idx * 3 + j];
            }
            // Accumulate X^T Y
            XT_Y[i] += X[idx * 3 + i] * Y[idx];
        }
    }

    // Solve 3x3 system using Gaussian elimination
    double beta[3] = {0}; // Regression coefficients
    if (abs(XT_X[0][0]) > 1e-10) {
        // Gaussian elimination to solve for beta
        for (int i = 0; i < 3; ++i) {
            for (int k = i + 1; k < 3; ++k) {
                double factor = XT_X[k][i] / XT_X[i][i]; // Pivot factor
                for (int j = i; j < 3; ++j) {
                    // Eliminate column i in row k
                    XT_X[k][j] -= factor * XT_X[i][j];
                }
                // Update right-hand side
                XT_Y[k] -= factor * XT_Y[i];
            }
        }
        // Back substitution to compute coefficients
        for (int i = 2; i >= 0; --i) {
            double sum = XT_Y[i];
            for (int j = i + 1; j < 3; ++j) {
                sum -= XT_X[i][j] * beta[j];
            }
            beta[i] = sum / XT_X[i][i]; // Solve for beta[i]
        }
    }

    // Compute continuation values for in-the-money paths
    for (size_t idx : in_the_money) {
        double s = S[idx]; // Stock price
        // Apply quadratic regression: beta0 + beta1*s + beta2*s^2
        continuation_values[idx] = beta[0] + beta[1] * s + beta[2] * s * s;
        // Ensure non-negative continuation values
        continuation_values[idx] = max(continuation_values[idx], 0.0);
    }
}

// Computes American option price using Black-Scholes model with Longstaff-Schwartz Monte Carlo
double Monte_Carlo::American_option_price(double S, double K, double T, double r, double sigma, bool is_call, int num_sims, int num_steps) {
    // Input: S (spot price), K (strike price), T (time to maturity), r (risk-free rate),
    //        sigma (volatility), is_call (true for call, false for put),
    //        num_sims (number of simulations), num_steps (time steps)
    // Output: Estimated American option price
    // Logic: Simulates stock price paths, uses least squares regression to estimate
    //        continuation values, and applies backward induction to determine optimal exercise
    if (S <= 0 || K <= 0 || T <= 0 || sigma <= 0 || num_sims <= 0 || num_steps <= 0) {
        // Validate inputs for positive values
        throw invalid_argument("Invalid inputs: S, K, T, sigma, num_sims, and num_steps must be positive.");
    }

    double dt = T / num_steps; // Time step size
    double sqrt_dt = sqrt(dt); // Square root of time step for volatility scaling
    vector<vector<double>> paths(num_sims, vector<double>(num_steps + 1)); // Stock price paths
    vector<double> payoffs(num_sims); // Option payoffs
    random_device rd; // Random seed generator
    unsigned int base_seed = rd(); // Base seed for random number generation

    // Simulate stock price paths
    #pragma omp parallel
    {
        mt19937 local_gen(base_seed + omp_get_thread_num()); // Thread-local random number generator
        normal_distribution<double> local_dist(0.0, 1.0); // Thread-local normal distribution
        #pragma omp for
        for (int i = 0; i < num_sims; ++i) {
            paths[i][0] = S; // Initialize path with spot price
            for (int t = 1; t <= num_steps; ++t) {
                double Z = local_dist(local_gen); // Random normal variable
                // Update stock price using Black-Scholes dynamics
                paths[i][t] = paths[i][t-1] * exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * Z);
            }
        }
    }

    // Initialize payoffs at maturity
    for (int i = 0; i < num_sims; ++i) {
        payoffs[i] = is_call ? max(paths[i][num_steps] - K, 0.0) : max(K - paths[i][num_steps], 0.0);
    }

    // Backward induction using Longstaff-Schwartz method
    for (int t = num_steps - 1; t >= 0; --t) {
        vector<double> S_t(num_sims), continuation_values; // Stock prices and continuation values at time t
        for (int i = 0; i < num_sims; ++i) {
            S_t[i] = paths[i][t]; // Extract stock prices at time t
        }
        // Estimate continuation values via regression
        least_squares_regression(S_t, payoffs, continuation_values);

        for (int i = 0; i < num_sims; ++i) {
            // Compute exercise value at time t
            double exercise_value = is_call ? max(paths[i][t] - K, 0.0) : max(K - paths[i][t], 0.0);
            // Get continuation value (0 at maturity)
            double continuation_value = (t == num_steps - 1) ? 0.0 : continuation_values[i];
            if (exercise_value > continuation_value) {
                // Exercise if exercise value exceeds continuation value
                payoffs[i] = exercise_value;
            } else {
                // Continue holding, discount future payoff
                payoffs[i] = payoffs[i] * exp(-r * dt);
            }
        }
    }

    // Compute average payoff to get option price
    double sum_payoffs = 0.0;
    for (int i = 0; i < num_sims; ++i) {
        sum_payoffs += payoffs[i];
    }

    return sum_payoffs / num_sims;
}

// Computes American option price using Heston model with Longstaff-Schwartz Monte Carlo
double Monte_Carlo::Heston_American_option_price(double S0, double K, double T, double r, double v0, double kappa,
                                                double theta, double xi, double rho, bool is_call, int num_sims, int num_steps) {
    // Input: S0 (spot price), K (strike price), T (time to maturity), r (risk-free rate),
    //        v0 (initial variance), kappa (mean reversion rate), theta (long-term variance),
    //        xi (volatility of variance), rho (correlation), is_call (true for call, false for put),
    //        num_sims (number of simulations), num_steps (time steps)
    // Output: Estimated American option price
    // Logic: Simulates stock and variance paths using Heston model, uses least squares regression
    //        to estimate continuation values, and applies backward induction
    if (S0 <= 0 || K <= 0 || T <= 0 || v0 < 0 || kappa <= 0 || theta <= 0 || xi <= 0 || num_sims <= 0 || num_steps <= 0) {
        // Validate inputs for positive/non-negative values
        throw invalid_argument("Invalid inputs: S0, K, T, kappa, theta, xi, num_sims, and num_steps must be positive; v0 must be non-negative.");
    }
    if (rho < -1.0 || rho > 1.0) {
        // Validate correlation coefficient
        throw invalid_argument("Invalid input: rho must be between -1 and 1.");
    }

    double dt = T / num_steps; // Time step size
    double sqrt_dt = sqrt(dt); // Square root of time step for volatility scaling
    vector<vector<double>> paths(num_sims, vector<double>(num_steps + 1)); // Stock price paths
    vector<vector<double>> variances(num_sims, vector<double>(num_steps + 1)); // Variance paths
    vector<double> payoffs(num_sims); // Option payoffs
    random_device rd; // Random seed generator
    unsigned int base_seed = rd(); // Base seed for random number generation

    // Simulate stock and variance paths
    #pragma omp parallel
    {
        mt19937 local_gen(base_seed + omp_get_thread_num()); // Thread-local random number generator
        normal_distribution<double> local_dist(0.0, 1.0); // Thread-local normal distribution
        #pragma omp for
        for (int i = 0; i < num_sims; ++i) {
            paths[i][0] = S0; // Initialize stock price
            variances[i][0] = v0; // Initialize variance
            for (int t = 1; t <= num_steps; ++t) {
                // Generate correlated random variables
                double z1 = local_dist(local_gen); // Random normal for stock price
                double z2 = rho * z1 + sqrt(1 - rho * rho) * local_dist(local_gen); // Correlated random normal for variance
                double v = variances[i][t-1]; // Current variance
                double sqrt_v = sqrt(max(v, 0.0)); // Square root of variance
                // Update variance with Euler discretization and Milstein correction
                v = v + kappa * (theta - v) * dt + xi * sqrt_v * sqrt_dt * z2 + 0.25 * xi * xi * dt * (z2 * z2 - 1);
                variances[i][t] = max(v, 0.0); // Ensure non-negative variance
                // Update stock price using Heston dynamics
                paths[i][t] = paths[i][t-1] * exp((r - 0.5 * variances[i][t]) * dt + sqrt_v * sqrt_dt * z1);
            }
        }
    }

    // Initialize payoffs at maturity
    for (int i = 0; i < num_sims; ++i) {
        payoffs[i] = is_call ? max(paths[i][num_steps] - K, 0.0) : max(K - paths[i][num_steps], 0.0);
    }

    // Backward induction using Longstaff-Schwartz method
    for (int t = num_steps - 1; t >= 0; --t) {
        vector<double> S_t(num_sims), continuation_values; // Stock prices and continuation values at time t
        for (int i = 0; i < num_sims; ++i) {
            S_t[i] = paths[i][t]; // Extract stock prices at time t
        }
        // Estimate continuation values via regression
        least_squares_regression(S_t, payoffs, continuation_values);

        for (int i = 0; i < num_sims; ++i) {
            // Compute exercise value at time t
            double exercise_value = is_call ? max(paths[i][t] - K, 0.0) : max(K - paths[i][t], 0.0);
            // Get continuation value (0 at maturity)
            double continuation_value = (t == num_steps - 1) ? 0.0 : continuation_values[i];
            if (exercise_value > continuation_value) {
                // Exercise if exercise value exceeds continuation value
                payoffs[i] = exercise_value;
            } else {
                // Continue holding, discount future payoff
                payoffs[i] = payoffs[i] * exp(-r * dt);
            }
        }
    }

    // Compute average payoff to get option price
    double sum_payoffs = 0.0;
    for (int i = 0; i < num_sims; ++i) {
        sum_payoffs += payoffs[i];
    }

    return sum_payoffs / num_sims;
}