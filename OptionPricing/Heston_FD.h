#ifndef HESTON_FD_H
#define HESTON_FD_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include "Black_Scholes.h"

using namespace std;

// Class for pricing options using Heston model with Finite Difference method
class Heston_FD {
private:
    // Model parameters
    double S0;        // Initial stock price
    double K;         // Strike price
    double T;         // Time to maturity
    double r;         // Risk-free rate
    double v0;        // Initial variance
    double kappa;     // Mean reversion rate
    double theta;     // Long-term variance
    double xi;        // Volatility of variance
    double rho;       // Correlation between stock and variance
    double S_max;     // Maximum stock price for grid
    double v_max;     // Maximum variance for grid
    double dS;        // Stock price step size
    double dv;        // Variance step size
    double dt;        // Time step size
    double dtau;      // Backward time step
    int N_S;          // Number of stock price steps
    int N_v;          // Number of variance steps
    int N_t;          // Number of time steps
    int max_iter;     // Maximum iterations for solver
    bool is_call;     // True for call option, false for put
    bool is_solved;   // Tracks if PDE is solved
    bool solved_is_call; // Tracks option type of solved PDE
    bool is_eu;       // True for European, false for American
    vector<vector<vector<double>>> V;       // 3D array for option prices
    vector<vector<vector<double>>> V_temp;  // Temporary storage for option prices
    vector<vector<vector<double>>> k1;      // Runge-Kutta stage 1
    vector<vector<vector<double>>> k2;      // Runge-Kutta stage 2
    vector<vector<vector<double>>> k3;      // Runge-Kutta stage 3
    vector<vector<vector<double>>> k4;      // Runge-Kutta stage 4
    vector<vector<vector<double>>> res;     // Residuals
    vector<vector<vector<double>>> res_prev; // Previous residuals

    // Calculate first derivative w.r.t. stock price
    double deriv1_S(const vector<vector<vector<double>>>& f, int i, int k, int j);
    // Calculate second derivative w.r.t. stock price
    double deriv2_S(const vector<vector<vector<double>>>& f, int i, int k, int j);
    // Calculate first derivative w.r.t. variance
    double deriv1_v(const vector<vector<vector<double>>>& f, int i, int k, int j);
    // Calculate second derivative w.r.t. variance
    double deriv2_v(const vector<vector<vector<double>>>& f, int i, int k, int j);
    // Calculate mixed derivative w.r.t. stock price and variance
    double deriv2_Sv(const vector<vector<vector<double>>>& f, int i, int k, int j);
    // Calculate first derivative w.r.t. time
    double deriv1_t(const vector<vector<vector<double>>>& f, int i, int k, int j);
    // Compute equation of motion for one grid point
    void calc_eom(int i, int k, int j, double& res, const vector<vector<vector<double>>>& V_in);
    // Apply boundary conditions to option prices
    void apply_boundary_conditions(vector<vector<vector<double>>>& V_out);
    // Compute one Runge-Kutta stage
    void compute_rk_stage(const vector<vector<vector<double>>>& V_in, vector<vector<vector<double>>>& V_out, double scale, vector<vector<vector<double>>>& k_stage);

public:
    // Constructor to initialize model parameters
    Heston_FD(double S0, double K, double T, double r, double v0, double kappa, double theta, double xi, double rho,
              int N_S, int N_v, int N_t, int max_iter, bool is_call, bool is_eu = true);

    // Solve Heston PDE for European option price
    void solve(bool is_call);
    // Solve Heston PDE for American option price
    void solve_american();
    // Get option price at given stock price and variance
    double get_option_price(double S, double v, bool is_call);
    // Validate computed price
    double check_price(double S, double v, bool is_call);
};

#endif // HESTON_FD_H