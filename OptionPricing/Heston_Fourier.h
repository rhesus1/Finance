#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <stdexcept>

#ifndef HESTON_FOURIER_H
#define HESTON_FOURIER_H

// Class for pricing options using Heston model with Fourier transform
class HestonFourier {
private:
    // Model parameters
    double S0;    // Initial stock price
    double K;     // Strike price
    double v0;    // Initial variance
    double T;     // Time to maturity
    double r;     // Risk-free rate
    double q;     // Dividend yield
    double kappa; // Mean reversion rate
    double theta; // Long-term variance
    double sigma; // Volatility of variance
    double rho;   // Correlation between stock and variance
    double Lphi, Uphi, dphi; // Integration bounds and step size for Fourier transform
    bool debug;   // Flag for debug output
    const double M_PI = 3.14159265358979323846; // Pi constant

public:
    // Constructor to initialize Heston model parameters
    HestonFourier(double S0_, double K_, double v0_, double T_, double r_, double q_,
                  double kappa_, double theta_, double xi_, double rho_,
                  double Lphi_, double Uphi_, double dphi_, bool debug_);

    // Enable/disable debug mode
    void SetDebug(bool d);

    // Perform trapezoidal integration
    double trapz(const std::vector<double>& y, double dx);

    // Compute Heston probability for Fourier transform
    double HestonProb(std::complex<double> phi, double tau, int j, int Trap);

    // Compute European call and put prices
    void HestonPrice(std::string PutCall, int trap, double& HestonC, double& HestonP);

    // Compute consolidated Heston probability
    double HestonProbConsol(std::complex<double> phi, double tau, int Trap);

    // Compute consolidated call and put prices
    void HestonPriceConsol(std::string PutCall, int trap, double& HestonC, double& HestonP);

    // Approximate American exercise boundary
    double AmericanExerciseBoundary(double tau, bool isCall);

    // Compute American option prices
    void HestonAmericanPrice(std::string PutCall, int trap, double& HestonC, double& HestonP);
};

#endif