#include <stdexcept>
#include <random>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <cmath>

#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

// Class for pricing options using Monte Carlo simulation
class Monte_Carlo {
public:
    // Constructor
    Monte_Carlo();
    // Destructor
    ~Monte_Carlo();

    // Price European option using Monte Carlo
    double option_price(double S, double K, double T, double r, double sigma, bool is_call, int num_sims);

    // Price Heston model option using Monte Carlo
    double Heston_option_price(double S0, double K, double T, double r, double v0, double kappa,
                              double theta, double xi, double rho, bool is_call, int num_sims, int num_steps);

    // Price American option using Monte Carlo (Longstaff-Schwartz)
    double American_option_price(double S, double K, double T, double r, double sigma, bool is_call, int num_sims, int num_steps);

    // Price Heston American option using Monte Carlo
    double Heston_American_option_price(double S0, double K, double T, double r, double v0, double kappa,
                                        double theta, double xi, double rho, bool is_call, int num_sims, int num_steps);

private:
    // Perform least squares regression for Longstaff-Schwartz method
    void least_squares_regression(const std::vector<double>& S, const std::vector<double>& payoffs,
                                  std::vector<double>& continuation_values);
};

#endif