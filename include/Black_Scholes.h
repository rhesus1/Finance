#include <boost/math/distributions/normal.hpp>
#include <cmath>
#include <algorithm>

#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

class Black_Scholes {
public:
    Black_Scholes();
    double NormCDF(double x);
    double call(double S, double K, double T, double r, double q, double v);
    double put(double S, double K, double T, double r, double q, double v);
    double AmericanCall(double S, double K, double T, double r, double q, double v);
    double AmericanPut(double S, double K, double T, double r, double q, double v);
private:
    double option_price_call_american_binomial(double S, double K, double T, double r, double q, double v, int steps);
    double option_price_put_american_binomial(double S, double K, double T, double r, double q, double v, int steps);
};

#endif