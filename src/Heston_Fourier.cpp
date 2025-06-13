#include "Heston_Fourier.h"
#include <cmath>
#include <iostream>
#include <complex>
#include <vector>
#include <string>
#include <stdexcept>

using namespace std;

HestonFourier::HestonFourier(double S0_, double K_, double v0_, double T_, double r_, double q_,
                             double kappa_, double theta_, double xi_, double rho_,
                            double Lphi_, double Uphi_, double dphi_, bool debug_)
    : S0(S0_), K(K_), v0(v0_), T(T_), r(r_), q(q_),
      kappa(kappa_), theta(theta_), sigma(xi_), rho(rho_), Lphi(Lphi_), Uphi(Uphi_), dphi(dphi_), debug(debug_) {
}

void HestonFourier::SetDebug(bool d) {
    debug = d;
}

double HestonFourier::HestonProb(complex<double> phi, double tau, int j, int Trap) {
    complex<double> i(0.0, 1.0);
    double x = log(S0);
    double a = kappa * theta;
    double u = (j == 1) ? 0.5 : -0.5;
    complex<double> b = (j == 1) ? kappa + i * rho * sigma : kappa;

    complex<double> d = sqrt(pow(rho * sigma * i * phi - b, 2) - pow(sigma, 2) * (2.0 * u * i * phi - pow(phi, 2)));
    complex<double> g = (b - rho * sigma * i * phi + d) / (b - rho * sigma * i * phi - d);
    complex<double> C, D, G;

    if (Trap == 1) {
        complex<double> c = 1.0 / g;
        G = (1.0 - c * exp(-d * tau)) / (1.0 - c);
        C = (r - q) * i * phi * tau + a / pow(sigma, 2) * ((b - rho * sigma * i * phi - d) * tau - 2.0 * log(G));
        D = (b - rho * sigma * i * phi - d) / pow(sigma, 2) * ((1.0 - exp(-d * tau)) / (1.0 - c * exp(-d * tau)));
    } else {
        G = (1.0 - g * exp(d * tau)) / (1.0 - g);
        C = (r - q) * i * phi * tau + a / pow(sigma, 2) * ((b - rho * sigma * i * phi + d) * tau - 2.0 * log(G));
        D = (b - rho * sigma * i * phi + d) / pow(sigma, 2) * ((1.0 - exp(d * tau)) / (1.0 - g * exp(d * tau)));
    }

    complex<double> f = exp(C + D * v0 + i * phi * x);
    complex<double> y = exp(-i * phi * log(K)) * f / (i * phi);
    return y.real();
}

double HestonFourier::trapz(const vector<double>& y, double dx) {
    double sum = 0.0;
    for (size_t i = 1; i < y.size(); ++i) {
        sum += (y[i-1] + y[i]) * 0.5 * dx;
    }
    return sum;
}

void HestonFourier::HestonPrice(string PutCall, int trap, double& HestonC, double& HestonP) {
    vector<double> phi;
    int N = static_cast<int>((Uphi - Lphi) / dphi) + 1;
    for (int k = 0; k < N; ++k) {
        phi.push_back(Lphi + k * dphi);
    }

    vector<double> int1(N), int2(N);
    for (int k = 0; k < N; ++k) {
        int1[k] = HestonProb(complex<double>(phi[k], 0.0), T, 1, trap);
        int2[k] = HestonProb(complex<double>(phi[k], 0.0), T, 2, trap);
    }

    double I1 = trapz(int1, dphi);
    double I2 = trapz(int2, dphi);

    double P1 = 0.5 + I1 / M_PI;
    double P2 = 0.5 + I2 / M_PI;

    HestonC = S0 * exp(-q * T) * P1 - K * exp(-r * T) * P2;
    HestonP = HestonC - S0 * exp(-q * T) + K * exp(-r * T);

}


