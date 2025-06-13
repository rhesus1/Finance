#include "Black_Scholes.h"
#include <boost/math/distributions/normal.hpp>
#include <cmath>
#include <algorithm>

using namespace std;

Black_Scholes::Black_Scholes() {
}

double Black_Scholes::call(double S, double K, double T, double r, double sigma) {
    using boost::math::normal;
    normal dist;
    if (T <= 0 || S <= 0 || K <= 0 || sigma <= 0) {
        return max(S - K, 0.0);
    }
    double d1 = (log(S/K) + (r + 0.5 * pow(sigma, 2)) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return S * cdf(dist, d1) - K * exp(-r * T) * cdf(dist, d2);
}

double Black_Scholes::put(double S, double K, double T, double r, double sigma) {
    using boost::math::normal;
    normal dist;
    if (T <= 0 || S <= 0 || K <= 0 || sigma <= 0) {
        return max(K - S, 0.0);
    }
    double d1 = (log(S/K) + (r + 0.5 * pow(sigma, 2)) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return K * exp(-r * T) * cdf(dist, -d2) - S * cdf(dist, -d1);
}