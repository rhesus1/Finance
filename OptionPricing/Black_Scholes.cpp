#include <boost/math/distributions/normal.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include "Black_Scholes.h"

using namespace std;

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Default constructor
Black_Scholes::Black_Scholes() {
}

// Approximates the cumulative distribution function for standard normal
double Black_Scholes::NormCDF(double x) {
    // Compute intermediate terms for approximation
    double x1 = 7.0 * exp(-0.5 * x * x);
    double x2 = 16.0 * exp(-x * x * (2.0 - sqrt(2.0)));
    double x3 = (7.0 + 0.25 * M_PI * x * x) * exp(-x * x);
    // Calculate approximation factor Q
    double Q = 0.5 * sqrt(1 - (x1 + x2 + x3) / 30.0);
    // Return CDF based on sign of x
    return x > 0 ? 0.5 + Q : 0.5 - Q;
}

// Calculates European call option price using Black-Scholes formula
double Black_Scholes::call(double S, double K, double T, double r, double q, double v) {
    // Handle edge cases
    if (T <= 0 || S <= 0 || K <= 0 || v <= 0) {
        return max(S - K, 0.0);
    }
    // Compute d1 and d2 terms
    double d1 = (log(S / K) + (r - q + 0.5 * pow(v, 2)) * T) / (v * sqrt(T));
    double d2 = d1 - v * sqrt(T);
    // Return call price
    return S * exp(-q * T) * NormCDF(d1) - K * exp(-r * T) * NormCDF(d2);
}

// Calculates European put option price using Black-Scholes formula
double Black_Scholes::put(double S, double K, double T, double r, double q, double v) {
    // Handle edge cases
    if (T <= 0 || S <= 0 || K <= 0 || v <= 0) {
        return max(K - S, 0.0);
    }
    // Compute d1 and d2 terms
    double d1 = (log(S / K) + (r - q + 0.5 * pow(v, 2)) * T) / (v * sqrt(T));
    double d2 = d1 - v * sqrt(T);
    // Return put price
    return K * exp(-r * T) * NormCDF(-d2) - S * exp(-q * T) * NormCDF(-d1);
}

// Prices American call option using binomial model
double Black_Scholes::option_price_call_american_binomial(double S, double K, double T, double r, double q, double v, int steps) {
    // Handle edge cases
    if (T <= 0 || S <= 0 || K <= 0 || v <= 0) {
        return max(S - K, 0.0);
    }
    // Calculate time step and discount factors
    double dt = T / steps;
    double R = exp(r * dt);
    double Rinv = 1.0 / R;
    // Compute up and down factors
    double u = exp(v * sqrt(dt));
    double d = 1.0 / u;
    // Calculate risk-neutral probabilities
    double p_up = (exp((r - q) * dt) - d) / (u - d);
    double p_down = 1.0 - p_up;
    // Initialize stock price and option value vectors
    vector<double> prices(steps + 1);
    prices[0] = S * pow(d, steps);
    double uu = u * u;
    // Populate stock prices at final time step
    for (int i = 1; i <= steps; ++i) prices[i] = uu * prices[i - 1];
    vector<double> call_values(steps + 1);
    // Set option payoffs at expiration
    for (int i = 0; i <= steps; ++i) call_values[i] = max(0.0, prices[i] - K);
    // Backward induction to compute option price
    for (int step = steps - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            // Discounted expected value
            call_values[i] = (p_up * call_values[i + 1] + p_down * call_values[i]) * Rinv;
            // Update stock price
            prices[i] = d * prices[i + 1];
            // Check early exercise
            call_values[i] = max(call_values[i], prices[i] - K);
        }
    }
    // Return option price at t=0
    return call_values[0];
}

// Prices American put option using binomial model
double Black_Scholes::option_price_put_american_binomial(double S, double K, double T, double r, double q, double v, int steps) {
    // Handle edge cases
    if (T <= 0 || S <= 0 || K <= 0 || v <= 0) {
        return max(K - S, 0.0);
    }
    // Calculate time step and discount factors
    double dt = T / steps;
    double R = exp(r * dt);
    double Rinv = 1.0 / R;
    // Compute up and down factors
    double u = exp(v * sqrt(dt));
    double d = 1.0 / u;
    // Calculate risk-neutral probabilities
    double p_up = (exp((r - q) * dt) - d) / (u - d);
    double p_down = 1.0 - p_up;
    // Initialize stock price and option value vectors
    vector<double> prices(steps + 1);
    prices[0] = S * pow(d, steps);
    double uu = u * u;
    // Populate stock prices at final time step
    for (int i = 1; i <= steps; ++i) prices[i] = uu * prices[i - 1];
    vector<double> put_values(steps + 1);
    // Set option payoffs at expiration
    for (int i = 0; i <= steps; ++i) put_values[i] = max(0.0, K - prices[i]);
    // Backward induction to compute option price
    for (int step = steps - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            // Discounted expected value
            put_values[i] = Rinv * (p_up * put_values[i + 1] + p_down * put_values[i]);
            // Update stock price
            prices[i] = u * prices[i];
            // Check early exercise
            put_values[i] = max(put_values[i], K - prices[i]);
        }
    }
    // Return option price at t=0
    return put_values[0];
}

// Wrapper for American call option pricing
double Black_Scholes::AmericanCall(double S, double K, double T, double r, double q, double v) {
    const int steps = 100; // Fixed number of binomial steps
    return option_price_call_american_binomial(S, K, T, r, q, v, steps);
}

// Wrapper for American put option pricing
double Black_Scholes::AmericanPut(double S, double K, double T, double r, double q, double v) {
    const int steps = 100; // Fixed number of binomial steps
    return option_price_put_american_binomial(S, K, T, r, q, v, steps);
}