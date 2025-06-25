#include <cmath>
#include <algorithm>
#include <vector>

#ifndef BLACK_SCHOLES_FD_H
#define BLACK_SCHOLES_FD_H

// Class for pricing options using Finite Difference method
class Black_Scholes_FD {
private:
    // Model parameters
    double S0;        // Initial stock price
    double S_max;     // Maximum stock price for grid
    double T;         // Time to maturity
    double r;         // Risk-free rate
    double sigma;     // Volatility
    double K;         // Strike price
    double dS;        // Stock price step size
    double dt;        // Time step size
    double dtau;      // Backward time step
    double sigma2;    // Volatility squared
    double dS2;       // Stock price step size squared
    double dtau_sixth;// One-sixth of backward time step
    int N_S;          // Number of stock price steps
    int N_t;          // Number of time steps
    int max_iter;     // Maximum iterations for solver
    bool is_call;     // True for call option, false for put
    bool use_arrested_newton; // Flag for using arrested Newton method
    std::vector<std::vector<double>> V, V_temp, k_temp, res, res_prev; // 2D arrays for option prices and temporary storage

    // Calculate first derivative w.r.t. stock price
    double deriv1_S(const std::vector<std::vector<double>>& f, int i, int j);
    // Calculate second derivative w.r.t. stock price
    double deriv2_S(const std::vector<std::vector<double>>& f, int i, int j);
    // Calculate first derivative w.r.t. time
    double deriv1_t(const std::vector<std::vector<double>>& f, int i, int j);
    // Compute equation of motion for one grid point
    void calc_eom(int i, int j, double& res, const std::vector<std::vector<double>>& V_in);
    // Apply boundary conditions to option prices
    void apply_boundary_conditions(std::vector<std::vector<double>>& V_out);
    // Compute one Runge-Kutta stage
    void compute_rk_stage(const std::vector<std::vector<double>>& V_in, std::vector<std::vector<double>>& V_out, double scale);

public:
    // Constructor to initialize model parameters
    Black_Scholes_FD(double S0, double K, double T, double r, double sigma, int N_S, int N_t, int max_iter, bool is_call, bool use_arrested_newton = false);

    // Solve for European option price
    void solve();
    // Solve for American option price
    void solve_american();
    // Get option price at given stock price
    double get_option_price(double S0);
    // Return number of iterations for convergence
    int get_convergence_iterations() const { return last_iter; }
private:
    int last_iter = 0; // Store last iteration count
};

#endif