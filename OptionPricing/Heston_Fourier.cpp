#include <cmath>
#include <iostream>
#include <complex>
#include <vector>
#include <string>
#include <stdexcept>
#include "Heston_Fourier.h"

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

double HestonFourier::trapz(const vector<double>& y, double dx) {
    double sum = 0.0;
    for (size_t i = 1; i < y.size(); ++i) {
        sum += (y[i-1] + y[i]) * 0.5 * dx;
    }
    return sum;
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

void HestonFourier::HestonPrice(string PutCall, int trap, double& HestonC, double& HestonP) {
    vector<double> phi;
    int N = static_cast<int>((Uphi - Lphi) / dphi) + 1;
    N = max(N, 1000); // Increase integration points for better accuracy
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

double HestonFourier::HestonProbConsol(complex<double> phi, double tau, int Trap) {
    complex<double> i(0.0, 1.0);
    double x = log(S0);
    double a = kappa * theta;

    // First characteristic function f1
    double u1 = 0.5;
    complex<double> b1 = kappa + i * rho * sigma;
    complex<double> d1 = sqrt(pow(rho * sigma * i * phi - b1, 2) - pow(sigma, 2) * (2.0 * u1 * i * phi - pow(phi, 2)));
    complex<double> g1 = (b1 - rho * sigma * i * phi + d1) / (b1 - rho * sigma * i * phi - d1);
    complex<double> C1, D1, G1;

    if (Trap == 1) {
        complex<double> c1 = 1.0 / g1;
        G1 = (1.0 - c1 * exp(-d1 * tau)) / (1.0 - c1);
        C1 = (r - q) * i * phi * tau + a / pow(sigma, 2) * ((b1 - rho * sigma * i * phi - d1) * tau - 2.0 * log(G1));
        D1 = (b1 - rho * sigma * i * phi - d1) / pow(sigma, 2) * ((1.0 - exp(-d1 * tau)) / (1.0 - c1 * exp(-d1 * tau)));
    } else {
        G1 = (1.0 - g1 * exp(d1 * tau)) / (1.0 - g1);
        C1 = (r - q) * i * phi * tau + a / pow(sigma, 2) * ((b1 - rho * sigma * i * phi + d1) * tau - 2.0 * log(G1));
        D1 = (b1 - rho * sigma * i * phi + d1) / pow(sigma, 2) * ((1.0 - exp(d1 * tau)) / (1.0 - g1 * exp(d1 * tau)));
    }
    complex<double> f1 = exp(C1 + D1 * v0 + i * phi * x);

    // Second characteristic function f2
    double u2 = -0.5;
    complex<double> b2 = kappa;
    complex<double> d2 = sqrt(pow(rho * sigma * i * phi - b2, 2) - pow(sigma, 2) * (2.0 * u2 * i * phi - pow(phi, 2)));
    complex<double> g2 = (b2 - rho * sigma * i * phi + d2) / (b2 - rho * sigma * i * phi - d2);
    complex<double> C2, D2, G2;

    if (Trap == 1) {
        complex<double> c2 = 1.0 / g2;
        G2 = (1.0 - c2 * exp(-d2 * tau)) / (1.0 - c2);
        C2 = (r - q) * i * phi * tau + a / pow(sigma, 2) * ((b2 - rho * sigma * i * phi - d2) * tau - 2.0 * log(G2));
        D2 = (b2 - rho * sigma * i * phi - d2) / pow(sigma, 2) * ((1.0 - exp(-d2 * tau)) / (1.0 - c2 * exp(-d2 * tau)));
    } else {
        G2 = (1.0 - g2 * exp(d2 * tau)) / (1.0 - g2);
        C2 = (r - q) * i * phi * tau + a / pow(sigma, 2) * ((b2 - rho * sigma * i * phi + d2) * tau - 2.0 * log(G2));
        D2 = (b2 - rho * sigma * i * phi + d2) / pow(sigma, 2) * ((1.0 - exp(d2 * tau)) / (1.0 - g2 * exp(d2 * tau)));
    }
    complex<double> f2 = exp(C2 + D2 * v0 + i * phi * x);

    complex<double> y = exp(-i * phi * log(K)) / (i * phi) * (S0 * exp(-q * tau) * f1 - K * exp(-r * tau) * f2);
    return y.real();
}

void HestonFourier::PriceConsol(string PutCall, int trap, double& HestonC, double& HestonP) {
    vector<double> phi;
    int N = static_cast<int>((Uphi - Lphi) / dphi) + 1;
    N = max(N, 1000); // Increase integration points
    for (int k = 0; k < N; ++k) {
        phi.push_back(Lphi + k * dphi);
    }

    vector<double> inte(N);
    for (int k = 0; k < N; ++k) {
        inte[k] = HestonProbConsol(complex<double>(phi[k], 0.0), T, trap);
    }

    double I = trapz(inte, dphi);

    HestonC = 0.5 * S0 * exp(-q * T) - 0.5 * K * exp(-r * T) + I / M_PI;
    HestonP = HestonC - S0 * exp(-q * T) + K * exp(-r * T);
}

double HestonFourier::AmericanExerciseBoundary(double tau, bool isCall) {
    // Improved approximation for early exercise boundary
    double b = r - q; // Adjusted drift
    double vol = sqrt(v0 + theta * (1.0 - exp(-kappa * tau))); // Effective volatility
    double boundary = K;

    if (isCall) {
        if (q > 1e-6) { // Only significant for non-zero dividends
            // Use Barone-Adesi-Whaley-like approximation
            boundary *= (1.0 + (r - q) * tau + vol * sqrt(tau) * (1.0 + sigma * sqrt(tau)));
            boundary = min(boundary, 2.0 * K); // Cap to prevent unrealistic boundaries
        } else {
            boundary = K * 1e6; // Effectively no early exercise for q ≈ 0
        }
    } else {
        // Boundary below strike for puts
        boundary *= max(0.5, 1.0 - (b * tau + vol * sqrt(tau) * (1.0 + sigma * sqrt(tau))));
        boundary = min(boundary, 0.95 * K); // Cap to prevent overly small S_star
    }
    return max(boundary, 0.0); // Ensure non-negative
}

void HestonFourier::HestonAmericanPrice(string PutCall, int trap, double& HestonC, double& HestonP) {
    // Compute European prices as the base
    HestonPrice(PutCall, trap, HestonC, HestonP);

    // Add early exercise premium using quadrature
    const int Nt = 500; // Time steps
    double dt = T / Nt;
    double earlyExercisePremiumC = 0.0;
    double earlyExercisePremiumP = 0.0;

    for (int i = 1; i <= Nt; ++i) {
        double tau = i * dt;
        double S_star = AmericanExerciseBoundary(tau, PutCall == "Call");

        // Compute risk-neutral probability using HestonProb
        vector<double> phi;
        int N = static_cast<int>((Uphi - Lphi) / dphi) + 1;
        N = max(N, 1000); // Increase integration points
        for (int k = 0; k < N; ++k) {
            phi.push_back(Lphi + k * dphi);
        }

        vector<double> inte(N);
        for (int k = 0; k < N; ++k) {
            complex<double> phi_c = complex<double>(phi[k], 0.0);
            double prob = HestonProb(phi_c, tau, 2, trap);
            double payoff = (PutCall == "Call") ? max(S_star - K, 0.0) : max(K - S_star, 0.0);
            inte[k] = prob * payoff;
        }

        double premium = trapz(inte, dphi) / M_PI;

        // Adjust premium with cash flows
        if (PutCall == "Call" && q > 1e-6) {
            earlyExercisePremiumC += (q * S_star * exp(-q * tau) - r * K * exp(-r * tau)) * premium * dt;
        } else if (PutCall == "Put") {
            earlyExercisePremiumP += (r * K * exp(-r * tau) - q * S_star * exp(-q * tau)) * premium * dt;
        }

        // Debug output
        if (debug) {
            cout << (PutCall == "Call" ? "Call" : "Put") << " - tau: " << tau
                 << ", S_star: " << S_star << ", premium: " << premium
                 << ", accumulated premium: " << (PutCall == "Call" ? earlyExercisePremiumC : earlyExercisePremiumP) << endl;
        }
    }

    // Add early exercise premium to European prices
    HestonC += max(earlyExercisePremiumC, 0.0); // Ensure non-negative premium
    HestonP += max(earlyExercisePremiumP, 0.0);

    // Ensure non-negative prices
    HestonC = max(HestonC, 0.0);
    HestonP = max(HestonP, 0.0);
}