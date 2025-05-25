#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

#include <cmath>
#include <boost/math/distributions/normal.hpp>

using namespace std;

class Black_Scholes
{
    public:
        Black_Scholes();
        double call(double S, double K, double T, double r, double sigma);
        double put(double S, double K, double T, double r, double sigma);
    protected:

    private:
};

#endif // BLACK_SCHOLES_H

