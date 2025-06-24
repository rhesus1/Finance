#include <boost/math/distributions/normal.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include "Black_Scholes.h"

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Black_Scholes::Black_Scholes() {
}

double Black_Scholes::NormCDF(double x) {
    double x1 = 7.0 * exp(-0.5 * x * x);
    double x2 = 16.0 * exp(-x * x * (2.0 - sqrt(2.0)));
    double x3 = (7.0 + 0.25 * M_PI * x * x) * exp(-x * x);
    double Q = 0.5 * sqrt(1 - (x1 + x2 + x3) / 30.0);
    if (x > 0) {
        return 0.5 + Q;
    } else {
        return 0.5 - Q;
    }
}

double Black_Scholes::call(double S, double K, double T, double r, double q, double v) {
    if (T <= 0 || S <= 0 || K <= 0 || v <= 0) {
        return max(S - K, 0.0);
    }
    double d1 = (log(S / K) + (r - q + 0.5 * pow(v, 2)) * T) / (v * sqrt(T));
    double d2 = d1 - v * sqrt(T);
    return S * exp(-q * T) * NormCDF(d1) - K * exp(-r * T) * NormCDF(d2);
}

double Black_Scholes::put(double S, double K, double T, double r, double q, double v) {
    if (T <= 0 || S <= 0 || K <= 0 || v <= 0) {
        return max(K - S, 0.0);
    }
    double d1 = (log(S / K) + (r - q + 0.5 * pow(v, 2)) * T) / (v * sqrt(T));
    double d2 = d1 - v * sqrt(T);
    return K * exp(-r * T) * NormCDF(-d2) - S * exp(-q * T) * NormCDF(-d1);
}

double Black_Scholes::option_price_call_american_binomial(double S, double K, double T, double r, double q, double v, int steps) {
    if (T <= 0 || S <= 0 || K <= 0 || v <= 0) {
        return max(S - K, 0.0);
    }
    double dt = T / steps;
    double R = exp(r * dt);
    double Rinv = 1.0 / R;
    double u = exp(v * sqrt(dt));
    double d = 1.0 / u;
    double p_up = (exp((r - q) * dt) - d) / (u - d);
    double p_down = 1.0 - p_up;
    vector<double> prices(steps + 1);
    prices[0] = S * pow(d, steps);
    double uu = u * u;
    for (int i = 1; i <= steps; ++i) prices[i] = uu * prices[i - 1];
    vector<double> call_values(steps + 1);
    for (int i = 0; i <= steps; ++i) call_values[i] = max(0.0, prices[i] - K);
    for (int step = steps - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            call_values[i] = (p_up * call_values[i + 1] + p_down * call_values[i]) * Rinv;
            prices[i] = d * prices[i + 1];
            call_values[i] = max(call_values[i], prices[i] - K);
        }
    }
    return call_values[0];
}

double Black_Scholes::option_price_put_american_binomial(double S, double K, double T, double r, double q, double v, int steps) {
    if (T <= 0 || S <= 0 || K <= 0 || v <= 0) {
        return max(K - S, 0.0);
    }
    double dt = T / steps;
    double R = exp(r * dt);
    double Rinv = 1.0 / R;
    double u = exp(v * sqrt(dt));
    double d = 1.0 / u;
    double p_up = (exp((r - q) * dt) - d) / (u - d);
    double p_down = 1.0 - p_up;
    vector<double> prices(steps + 1);
    prices[0] = S * pow(d, steps);
    double uu = u * u;
    for (int i = 1; i <= steps; ++i) prices[i] = uu * prices[i - 1];
    vector<double> put_values(steps + 1);
    for (int i = 0; i <= steps; ++i) put_values[i] = max(0.0, K - prices[i]);
    for (int step = steps - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            put_values[i] = Rinv * (p_up * put_values[i + 1] + p_down * put_values[i]);
            prices[i] = u * prices[i];
            put_values[i] = max(put_values[i], K - prices[i]);
        }
    }
    return put_values[0];
}

double Black_Scholes::AmericanCall(double S, double K, double T, double r, double q, double v) {
    const int steps = 100; // Reasonable number of steps for binomial model
    return option_price_call_american_binomial(S, K, T, r, q, v, steps);
}

double Black_Scholes::AmericanPut(double S, double K, double T, double r, double q, double v) {
    const int steps = 100; // Reasonable number of steps for binomial model
    return option_price_put_american_binomial(S, K, T, r, q, v, steps);
}