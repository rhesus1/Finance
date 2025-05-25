#include "Black_Scholes.h"
#include <cmath>

using namespace std;

Black_Scholes::Black_Scholes(){
}

double Black_Scholes::call(double S, double K, double T, double r, double sigma){
    using boost::math::normal;
    normal dist;
    double d1 = (log(S/K) + (r + 0.5 * pow(sigma,2) * T) / sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return S * cdf(dist, d1) - K * exp(-r * T) * cdf(dist,d2);
}

double Black_Scholes::put(double S, double K, double T, double r, double sigma){
    using boost::math::normal;
    normal dist;
    double d1 = (log(S/K) + (r + 0.5 * pow(sigma,2) * T) / sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return -S * cdf(dist, -d1) + K * exp(-r * T) * cdf(dist,-d2);
}