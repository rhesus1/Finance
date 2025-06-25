#include <boost/math/distributions/normal.hpp>
#include <cmath>
#include <algorithm>

#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

// Class for pricing options using Black-Scholes model
class Black_Scholes {
public:
    // Constructor
    Black_Scholes();

    // Compute cumulative distribution function for normal distribution
    double NormCDF(double x);

    // Price European call option
    double call(double S, double K, double T, double r, double q, double v);

    // Price European put option
    double put(double S, double K, double T, double r, double q, double v);

    // Price American call option
    double AmericanCall(double S, double K, double T, double r, double q, double v);

    // Price American put option
    double AmericanPut(double S, double K, double T, double r, double q, double v);

private:
    // Price American call using binomial tree
    double option_price_call_american_binomial(double S, double K, double T, double r, double q, double v, int steps);

    // Price American put using binomial tree
    double option_price_put_american_binomial(double S, double K, double T, double r, double q, double v, int steps);
};

#endif