#ifndef HESTON_FOURIER_H
#define HESTON_FOURIER_H

#include <complex>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class HestonFourier {
private:
    double S0, K, v0, T, r, q;
    double kappa, theta, sigma, rho;
    double Lphi, Uphi, dphi;
    bool debug;

    double HestonProb(std::complex<double> phi, double tau, int j, int Trap);
    double trapz(const std::vector<double>& y, double dx);

public:
    HestonFourier(double S0_, double K_, double v0_, double T_, double r_, double q_,
                  double kappa_, double theta_, double sigma_, double rho_,
                  double Lphi_, double Uphi_, double dphi_, bool debug_ = false);
    void SetDebug(bool d);
    void HestonPrice(std::string PutCall, int trap, double& HestonC, double& HestonP);
};

#endif