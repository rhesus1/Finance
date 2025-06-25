#include <cmath>
#include <iostream>
#include <complex>
#include <vector>
#include <string>
#include <stdexcept>
#include "Heston_Fourier.h"

using namespace std;

// Constructor initializes Heston model parameters and integration bounds
HestonFourier::HestonFourier(double S0_, double K_, double v0_, double T_, double r_, double q_,
                             double kappa_, double theta_, double xi_, double rho_,
                             double Lphi_, double Uphi_, double dphi_, bool debug_)
    : S0(S0_), K(K_), v0(v0_), T(T_), r(r_), q(q_),
      kappa(kappa_), theta(theta_), sigma(xi_), rho(rho_), Lphi(Lphi_), Uphi(Uphi_), dphi(dphi_), debug(debug_) {
    // Input: S0_ (spot price), K_ (strike price), v0_ (initial variance), T_ (time to maturity),
    //        r_ (risk-free rate), q_ (dividend yield), kappa_ (mean reversion rate),
    //        theta_ (long-term variance), xi_ (volatility of variance), rho_ (correlation),
    //        Lphi_ (lower integration bound), Uphi_ (upper integration bound),
    //        dphi_ (integration step size), debug_ (debug flag)
    // Output: None (initializes member variables)
    // Logic: Assigns input parameters to class members for use in pricing functions
}

// Sets debug flag for diagnostic output
void HestonFourier::SetDebug(bool d) {
    // Input: d (boolean debug flag)
    // Output: None (updates debug member)
    // Logic: Sets the debug flag to enable/disable diagnostic output
    debug = d;
}

// Performs trapezoidal integration
double HestonFourier::trapz(const vector<double>& y, double dx) {
    // Input: y (vector of integrand values), dx (integration step size)
    // Output: Approximate integral value
    // Logic: Applies trapezoidal rule to compute integral of y over evenly spaced points
    //        using formula: sum((y[i-1] + y[i]) * dx / 2) for each interval
    double sum = 0.0;
    for (size_t i = 1; i < y.size(); ++i) {
        // Sum (y[i-1] + y[i]) * dx / 2 for each interval
        sum += (y[i-1] + y[i]) * 0.5 * dx;
    }
    return sum;
}

// Computes Heston model probability function for Fourier integration
double HestonFourier::HestonProb(complex<double> phi, double tau, int j, int Trap) {
    // Input: phi (Fourier variable, complex), tau (time to maturity), j (probability index: 1 or 2),
    //        Trap (trapezoidal integration flag: 1 for modified, 0 for standard)
    // Output: Real part of characteristic function for pricing
    // Logic: Computes characteristic function for probability P1 (j=1) or P2 (j=2)
    //        using Heston model dynamics, with optional trapezoidal modification
    complex<double> i(0.0, 1.0); // Imaginary unit
    double x = log(S0); // Log of spot price
    double a = kappa * theta; // Mean reversion term (kappa * long-term variance)
    double u = (j == 1) ? 0.5 : -0.5; // Adjusts characteristic function for P1 or P2
    complex<double> b = (j == 1) ? kappa + i * rho * sigma : kappa; // Drift adjustment for stochastic volatility

    // Compute discriminant for characteristic function
    complex<double> d = sqrt(pow(rho * sigma * i * phi - b, 2) - pow(sigma, 2) * (2.0 * u * i * phi - pow(phi, 2)));
    // Compute g term for characteristic function
    complex<double> g = (b - rho * sigma * i * phi + d) / (b - rho * sigma * i * phi - d);
    complex<double> C, D, G;

    if (Trap == 1) {
        // Modified trapezoidal integration for numerical stability
        complex<double> c = 1.0 / g; // Inverse of g
        G = (1.0 - c * exp(-d * tau)) / (1.0 - c); // G term for modified form
        // Compute C term (drift and variance effects)
        C = (r - q) * i * phi * tau + a / pow(sigma, 2) * ((b - rho * sigma * i * phi - d) * tau - 2.0 * log(G));
        // Compute D term (variance contribution)
        D = (b - rho * sigma * i * phi - d) / pow(sigma, 2) * ((1.0 - exp(-d * tau)) / (1.0 - c * exp(-d * tau)));
    } else {
        // Standard integration form
        G = (1.0 - g * exp(d * tau)) / (1.0 - g); // G term for standard form
        C = (r - q) * i * phi * tau + a / pow(sigma, 2) * ((b - rho * sigma * i * phi + d) * tau - 2.0 * log(G));
        D = (b - rho * sigma * i * phi + d) / pow(sigma, 2) * ((1.0 - exp(d * tau)) / (1.0 - g * exp(d * tau)));
    }

    // Compute characteristic function: exp(C + D * v0 + i * phi * x)
    complex<double> f = exp(C + D * v0 + i * phi * x);
    // Adjust for pricing integral: divide by i*phi and adjust by strike
    complex<double> y = exp(-i * phi * log(K)) * f / (i * phi);
    return y.real(); // Return real part for integration
}

// Computes European option prices using Heston model Fourier integration
void HestonFourier::HestonPrice(string PutCall, int trap, double& HestonC, double& HestonP) {
    // Input: PutCall ("Call" or "Put"), trap (trapezoidal integration flag: 1 for modified, 0 for standard),
    //        HestonC (reference to call price), HestonP (reference to put price)
    // Output: None (updates HestonC and HestonP with computed prices)
    // Logic: Computes call and put prices using Fourier integration of probabilities P1 and P2,
    //        applies put-call parity for consistency
    vector<double> phi;
    int N = static_cast<int>((Uphi - Lphi) / dphi) + 1; // Number of integration points
    N = max(N, 1000); // Ensure at least 1000 points for accuracy
    for (int k = 0; k < N; ++k) {
        phi.push_back(Lphi + k * dphi); // Initialize Fourier variable grid (integration points)
    }

    vector<double> int1(N), int2(N);
    for (int k = 0; k < N; ++k) {
        // Compute integrands for probabilities P1 and P2
        int1[k] = HestonProb(complex<double>(phi[k], 0.0), T, 1, trap); // Integrand for P1
        int2[k] = HestonProb(complex<double>(phi[k], 0.0), T, 2, trap); // Integrand for P2
    }

    // Integrate using trapezoidal rule
    double I1 = trapz(int1, dphi); // Integral for P1
    double I2 = trapz(int2, dphi); // Integral for P2

    // Compute probabilities P1 and P2
    double P1 = 0.5 + I1 / M_PI; // Probability P1
    double P2 = 0.5 + I2 / M_PI; // Probability P2

    // Compute call price using Heston formula: S0 * e^(-qT) * P1 - K * e^(-rT) * P2
    HestonC = S0 * exp(-q * T) * P1 - K * exp(-r * T) * P2;
    // Compute put price using put-call parity: HestonC - S0 * e^(-qT) + K * e^(-rT)
    HestonP = HestonC - S0 * exp(-q * T) + K * exp(-r * T);
}

// Computes consolidated probability function for Heston model pricing
double HestonFourier::HestonProbConsol(complex<double> phi, double tau, int Trap) {
    // Input: phi (Fourier variable, complex), tau (time to maturity),
    //        Trap (trapezoidal integration flag: 1 for modified, 0 for standard)
    // Output: Real part of consolidated characteristic function for pricing
    // Logic: Combines characteristic functions f1 and f2 into a single integrand
    //        for efficient call/put price computation
    complex<double> i(0.0, 1.0); // Imaginary unit
    double x = log(S0); // Log of spot price
    double a = kappa * theta; // Mean reversion term (kappa * long-term variance)

    // First characteristic function f1 (for stock price probability)
    double u1 = 0.5; // Adjusts for P1
    complex<double> b1 = kappa + i * rho * sigma; // Drift adjustment for f1
    complex<double> d1 = sqrt(pow(rho * sigma * i * phi - b1, 2) - pow(sigma, 2) * (2.0 * u1 * i * phi - pow(phi, 2))); // Discriminant for f1
    complex<double> g1 = (b1 - rho * sigma * i * phi + d1) / (b1 - rho * sigma * i * phi - d1); // g term for f1
    complex<double> C1, D1, G1;

    if (Trap == 1) {
        // Modified trapezoidal integration for numerical stability
        complex<double> c1 = 1.0 / g1; // Inverse of g1
        G1 = (1.0 - c1 * exp(-d1 * tau)) / (1.0 - c1); // G term for f1
        // Compute C1 term (drift and variance effects for f1)
        C1 = (r - q) * i * phi * tau + a / pow(sigma, 2) * ((b1 - rho * sigma * i * phi - d1) * tau - 2.0 * log(G1));
        // Compute D1 term (variance contribution for f1)
        D1 = (b1 - rho * sigma * i * phi - d1) / pow(sigma, 2) * ((1.0 - exp(-d1 * tau)) / (1.0 - c1 * exp(-d1 * tau)));
    } else {
        // Standard integration
        G1 = (1.0 - g1 * exp(d1 * tau)) / (1.0 - g1); // G term for standard form
        C1 = (r - q) * i * phi * tau + a / pow(sigma, 2) * ((b1 - rho * sigma * i * phi + d1) * tau - 2.0 * log(G1));
        D1 = (b1 - rho * sigma * i * phi + d1) / pow(sigma, 2) * ((1.0 - exp(d1 * tau)) / (1.0 - g1 * exp(d1 * tau)));
    }
    // Compute characteristic function f1
    complex<double> f1 = exp(C1 + D1 * v0 + i * phi * x);

    // Second characteristic function f2 (for risk-neutral probability)
    double u2 = -0.5; // Adjusts for P2
    complex<double> b2 = kappa; // Drift adjustment for f2
    complex<double> d2 = sqrt(pow(rho * sigma * i * phi - b2, 2) - pow(sigma, 2) * (2.0 * u2 * i * phi - pow(phi, 2))); // Discriminant for f2
    complex<double> g2 = (b2 - rho * sigma * i * phi + d2) / (b2 - rho * sigma * i * phi - d2); // g term for f2
    complex<double> C2, D2, G2;

    if (Trap == 1) {
        // Modified trapezoidal integration
        complex<double> c2 = 1.0 / g2; // Inverse of g2
        G2 = (1.0 - c2 * exp(-d2 * tau)) / (1.0 - c2); // G term for f2
        // Compute C2 term (drift and variance effects for f2)
        C2 = (r - q) * i * phi * tau + a / pow(sigma, 2) * ((b2 - rho * sigma * i * phi - d2) * tau - 2.0 * log(G2));
        // Compute D2 term (variance contribution for f2)
        D2 = (b2 - rho * sigma * i * phi - d2) / pow(sigma, 2) * ((1.0 - exp(-d2 * tau)) / (1.0 - c2 * exp(-d2 * tau)));
    } else {
        // Standard integration
        G2 = (1.0 - g2 * exp(d2 * tau)) / (1.0 - g2); // G term for standard form
        C2 = (r - q) * i * phi * tau + a / pow(sigma, 2) * ((b2 - rho * sigma * i * phi + d2) * tau - 2.0 * log(G2));
        D2 = (b2 - rho * sigma * i * phi + d2) / pow(sigma, 2) * ((1.0 - exp(d2 * tau)) / (1.0 - g2 * exp(d2 * tau)));
    }
    // Compute characteristic function f2
    complex<double> f2 = exp(C2 + D2 * v0 + i * phi * x);

    // Combine f1 and f2 for consolidated pricing integrand
    complex<double> y = exp(-i * phi * log(K)) / (i * phi) * (S0 * exp(-q * tau) * f1 - K * exp(-r * tau) * f2);
    return y.real(); // Return real part for integration
}

// Computes European option prices using consolidated Heston probability function
void HestonFourier::HestonPriceConsol(string PutCall, int trap, double& HestonC, double& HestonP) {
    // Input: PutCall ("Put" or "Call"), trap (trapezoidal integration flag: 1 for modified, 0 for standard),
    //        HestonC (reference to call price), HestonP (reference to put price)
    // Output: None (updates HestonC and HestonP with computed prices)
    // Logic: Integrates consolidated probability function to compute call price directly,
    //        then uses put-call parity to compute put price
    vector<double> phi;
    int N = static_cast<int>((Uphi - Lphi) / dphi) + 1; // Number of integration points
    N = max(N, 1000); // Ensure at least 1000 points for accuracy
    for (int k = 0; k < N; ++k) {
        phi.push_back(Lphi + k * dphi); // Initialize Fourier variable grid
    }

    vector<double> inte(N);
    for (int k = 0; k < N; ++k) {
        // Compute consolidated integrand
        inte[k] = HestonProbConsol(complex<double>(phi[k], 0.0), T, trap);
    }

    // Integrate using trapezoidal rule
    double I = trapz(inte, dphi);

    // Compute call price: 0.5 * S0 * e^(-qT) - 0.5 * K * e^(-rT) + I/π
    HestonC = 0.5 * S0 * exp(-q * T) - 0.5 * K * exp(-r * T) + I / M_PI;
    // Compute put price using put-call parity: HestonC - S0 * e^(-qT) + K * e^(-rT)
    HestonP = HestonC - S0 * exp(-q * T) + K * exp(-r * T);
}

// Approximates the early exercise boundary for American options
double HestonFourier::AmericanExerciseBoundary(double tau, bool isCall) {
    // Input: tau (time to maturity), isCall (true for call, false for put)
    // Output: Approximate early exercise boundary (stock price S_star)
    // Logic: Uses an approximation inspired by Barone-Adesi-Whaley to estimate
    //        the stock price at which early exercise is optimal
    double b = r - q; // Adjusted drift (risk-free rate minus dividend yield)
    double vol = sqrt(v0 + theta * (1.0 - exp(-kappa * tau))); // Effective volatility, blending initial and long-term variance
    double boundary = K; // Initialize boundary at strike price

    if (isCall) {
        if (q > 1e-6) { // Only significant for non-zero dividends
            // Barone-Adesi-Whaley-like approximation for call boundary
            boundary *= (1.0 + (r - q) * tau + vol * sqrt(tau) * (1.0 + sigma * sqrt(tau)));
            boundary = min(boundary, 2.0 * K); // Cap to prevent unrealistic boundaries
        } else {
            boundary = K * 1e6; // Effectively no early exercise for near-zero dividends
        }
    } else {
        // Boundary below strike for puts
        boundary *= max(0.5, 1.0 - (b * tau + vol * sqrt(tau) * (1.0 + sigma * sqrt(tau))));
        boundary = min(boundary, 0.95 * K); // Cap to prevent overly small S_star
    }
    return max(boundary, 0.0); // Ensure non-negative boundary
}

// Computes American option prices using Heston model with early exercise premium
void HestonFourier::HestonAmericanPrice(string PutCall, int trap, double& HestonC, double& HestonP) {
    // Input: PutCall ("Call" or "Put"), trap (trapezoidal integration flag: 1 for modified, 0 for standard),
    //        HestonC (reference to call price), HestonP (reference to put price)
    // Output: None (updates HestonC and HestonP with American option prices)
    // Logic: Starts with European prices, adds early exercise premium via quadrature
    //        over time steps, using risk-neutral probabilities and exercise boundaries
    // Compute European prices as the base
    HestonPrice(PutCall, trap, HestonC, HestonP);

    // Add early exercise premium using quadrature
    const int Nt = 500; // Number of time steps for quadrature
    double dt = T / Nt; // Time step size
    double earlyExercisePremiumC = 0.0; // Accumulator for call premium
    double earlyExercisePremiumP = 0.0; // Accumulator for put premium

    for (int i = 1; i <= Nt; ++i) {
        double tau = i * dt; // Time to maturity at current step
        double S_star = AmericanExerciseBoundary(tau, PutCall == "Call"); // Early exercise boundary

        // Compute risk-neutral probability using HestonProb
        vector<double> phi;
        int N = static_cast<int>((Uphi - Lphi) / dphi) + 1; // Number of integration points
        N = max(N, 1000); // Ensure at least 1000 points
        for (int k = 0; k < N; ++k) {
            phi.push_back(Lphi + k * dphi); // Initialize Fourier variable grid
        }

        vector<double> inte(N);
        for (int k = 0; k < N; ++k) {
            complex<double> phi_c = complex<double>(phi[k], 0.0); // Fourier variable
            double prob = HestonProb(phi_c, tau, 2, trap); // Risk-neutral probability (P2)
            double payoff = (PutCall == "Call") ? max(S_star - K, 0.0) : max(K - S_star, 0.0); // Immediate exercise payoff
            inte[k] = prob * payoff; // Integrand for premium
        }

        // Compute premium contribution at current time step
        double premium = trapz(inte, dphi) / M_PI;

        // Adjust premium with cash flows
        if (PutCall == "Call" && q > 1e-6) {
            // Accumulate call premium with dividend and interest rate effects
            earlyExercisePremiumC += (q * S_star * exp(-q * tau) - r * K * exp(-r * tau)) * premium * dt;
        } else if (PutCall == "Put") {
            // Accumulate put premium with interest rate and dividend effects
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