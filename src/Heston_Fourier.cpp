#include "Heston_Fourier.h"

using namespace std;

Heston_Fourier::Heston_Fourier(double S0, double K, double T, double r, double v0, double kappa, double theta, double xi, double rho): S0(S0), K(K), T(T), r(r), v0(v0), kappa(kappa), theta(theta), xi(xi), rho(rho) {}

Heston_Fourier::~Heston_Fourier() {}

complex<double> Heston_Fourier::characteristic_function(complex<double> u) {
    const complex<double> i(0, 1);
        complex<double> b = kappa - i * rho * xi * u;
        complex<double> d = sqrt(b * b + xi * xi * (u * u + i * u));
        complex<double> g = (b - d) / (b + d);
        complex<double> C = r * i * u * T + (kappa * theta / (xi * xi)) *
                           ((b - d) * T - 2.0 * log((1.0 - g * exp(-d * T)) / (1.0 - g)));
        complex<double> D = (b - d) / (xi * xi) * (1.0 - exp(-d * T)) / (1.0 - g * exp(-d * T));
        complex<double> result = exp(C + D * v0 + i * u * log(S0));
        return isnan(real(result)) || isinf(real(result)) ? 0.0 : result;
}

double Heston_Fourier::integrate_Pj(int j) {
const complex<double> i(0, 1);
        double alpha = (j == 1) ? 1.1 : 0.1;
        double u_max = 50.0, du = 0.005;
        double sum = 0.0;
        for (double u = du; u < u_max; u += du) {
            complex<double> u_complex = u - i * alpha;
            complex<double> phi = characteristic_function(u_complex);
            complex<double> denominator = i * u;
            complex<double> integrand = exp(-i * u_complex * log(K)) * phi / denominator;
            double term = real(integrand);
            if (!isnan(term) && !isinf(term)) {
                sum += term * du;
            }
        }
        double result = 0.5 + sum / pi;
        return max(0.0, min(1.0, result));
}

double Heston_Fourier::price_call() {
    double P1 = integrate_Pj(1);
    double P2 = integrate_Pj(2);
    return max(0.0, S0 * P1 - K * exp(-r * T) * P2);
}
