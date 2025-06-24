#include <stdexcept>
#include <random>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <cmath>

#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

class Monte_Carlo {
public:
    Monte_Carlo();
    ~Monte_Carlo();

    double option_price(double S, double K, double T, double r, double sigma, bool is_call, int num_sims);

    double Heston_option_price(double S0, double K, double T, double r, double v0, double kappa,
                              double theta, double xi, double rho, bool is_call, int num_sims, int num_steps);

    // New functions for American option pricing
    double American_option_price(double S, double K, double T, double r, double sigma, bool is_call, int num_sims, int num_steps);
    double Heston_American_option_price(double S0, double K, double T, double r, double v0, double kappa,
                                        double theta, double xi, double rho, bool is_call, int num_sims, int num_steps);

private:
    // Helper function for regression in LSMC
    void least_squares_regression(const std::vector<double>& S, const std::vector<double>& payoffs,
                                  std::vector<double>& continuation_values);
};

#endif