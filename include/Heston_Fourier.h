#ifndef HESTON_FOURIER_H
#define HESTON_FOURIER_H


#include <iostream>
#include <complex>
#include <cmath>
using namespace std;

class Heston_Fourier
{
    public:
        Heston_Fourier(double S0, double K, double T, double r, double v0, double kappa, double theta, double xi, double rho);
        virtual ~Heston_Fourier();
    double price_call();

    protected:

    private:
        double S0, K, T, r, v0, kappa, theta, xi, rho;
        const double pi = 3.141592653589793;

        complex<double> characteristic_function(complex<double> u);
        double integrate_Pj(int j);
};

#endif // HESTON_FOURIER_H



