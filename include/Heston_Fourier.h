#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <stdexcept>

#ifndef HESTON_FOURIER_H
#define HESTON_FOURIER_H

class HestonFourier {
private:
    double S0, K, v0, T, r, q, kappa, theta, sigma, rho, Lphi, Uphi, dphi;
    bool debug;
    const double M_PI = 3.14159265358979323846;

public:
    HestonFourier(double S0_, double K_, double v0_, double T_, double r_, double q_,
                  double kappa_, double theta_, double xi_, double rho_,
                  double Lphi_, double Uphi_, double dphi_, bool debug_);

    void SetDebug(bool d);

    double trapz(const std::vector<double>& y, double dx);

    double HestonProb(std::complex<double> phi, double tau, int j, int Trap);

    void HestonPrice(std::string PutCall, int trap, double& HestonC, double& HestonP);

    double HestonProbConsol(std::complex<double> phi, double tau, int Trap);

    void PriceConsol(std::string PutCall, int trap, double& HestonC, double& HestonP);

    // New function for American exercise boundary approximation
    double AmericanExerciseBoundary(double tau, bool isCall);

    // New function for American option pricing
    void HestonAmericanPrice(std::string PutCall, int trap, double& HestonC, double& HestonP);
};

#endif