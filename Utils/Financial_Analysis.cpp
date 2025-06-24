#include "Financial_Analysis.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

Financial_Analysis::Financial_Analysis()
{
    //ctor
}

//Annualised Log Return
double Financial_Analysis::calc_volatility(vector<double>& close, int duration) {
    vector<double> returns;
    for (size_t i = 1; i < min(close.size(), size_t(duration)); ++i) {
        returns.push_back(log(close[i] / close[i-1]));
    }
    double mean = computeMean(returns);
    double variance = computeVariance(returns);
    return sqrt(variance * 252); // Annualized volatility
}

//Daily Log Return (Variance)
vector<double> Financial_Analysis::computeDailyVariance(vector<double>& close, int duration) {
    vector<double> variances;
    for (size_t i = 1; i < min(close.size(), size_t(duration)); ++i) {
        double logReturn = log(close[i] / close[i-1]);
        variances.push_back(logReturn * logReturn); // Daily variance proxy
    }
    return variances;
}

double Financial_Analysis::blackScholesCall(double S, double K, double r, double T, double sigma) {
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    // Using normal CDF approximation (simplified)
    double normCDF_d1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)));
    double normCDF_d2 = 0.5 * (1.0 + erf(d2 / sqrt(2.0)));
    return S * normCDF_d1 - K * exp(-r * T) * normCDF_d2;
}

// Newton-Raphson to find implied volatility (sqrt(v0))
double Financial_Analysis::impliedVolatility(double S, double K, double r, double T, double optionPrice){
double tol = 1e-6;
 int maxIter = 100;
    double sigma = 0.2; // Initial guess
    for (int i = 0; i < maxIter; ++i) {
        double price = blackScholesCall(S, K, r, T, sigma);
        double vega = S * sqrt(T) * exp(-0.5 * pow((log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T)), 2)) / sqrt(2.0 * M_PI); // Vega
        double diff = price - optionPrice;
        if (abs(diff) < tol) return sigma;
        sigma -= diff / vega;
        if (sigma < 0.01) sigma = 0.01;
    }
    return sigma;
}

double Financial_Analysis::computeMean(const vector<double>& data) {
    return accumulate(data.begin(), data.end(), 0.0) / data.size();
}

//Variance
double Financial_Analysis::computeVariance(const vector<double>& data) {
    double mean = computeMean(data);
    double sumSquaredDiff = 0.0;
    for (double x : data) {
        sumSquaredDiff += (x - mean) * (x - mean);
    }
    return sumSquaredDiff / (data.size() - 1);
}

//Covariance
double Financial_Analysis::computeCovariance(const vector<double>& x, const vector<double>& y) {
    /*if (x.size() != y.size()){
        cout << x.size() << ", " << y.size() << endl;
        return 0.0;
    }*/

    double meanX = computeMean(x);
    double meanY = computeMean(y);
    double sumCov = 0.0;
    for (size_t i = 0; i < min(x.size(),y.size()); ++i) {
        sumCov += (x[i] - meanX) * (y[i] - meanY);
    }
    return sumCov / (x.size() - 1);
}

//Pearson Correlation (rho)
double Financial_Analysis::computeCorrelation(const vector<double>& x, const vector<double>& y) {
    double cov = computeCovariance(x, y);
    double varX = computeVariance(x);
    double varY = computeVariance(y);
    if (varX == 0 || varY == 0) return 0.0;
    return cov / sqrt(varX * varY);
}


double Financial_Analysis::hestonOptionPrice(double S0, double K, double r, double T,
                                            double kappa, double theta, double sigma, double rho, double v0) {
    // Simplified Fourier integration (pseudo-code)
    double price = 0.0;
    // Implement Fourier transform using Cui et al. (2017) characteristic function
    // Placeholder: return Black-Scholes price as fallback
    double vol = std::sqrt(v0);
    return blackScholesCall(S0, K, r, T, vol);
}

//kappa (Speed of mean reversion), theta (Long term variance), sigma (volatility of variance)
void Financial_Analysis::estimateHestonParameters(const vector<double>& impliedVols, double S0, double K, double r, double T, double dt,
                             double& kappa, double& theta, double& sigma, double& rho, double& v0) {

    // Step 1: Compute market option prices from implied volatilities
    std::vector<double> marketPrices(impliedVols.size());
    for (size_t i = 0; i < impliedVols.size(); ++i) {
        // Use Black-Scholes to convert implied volatility to option price
        double t = T - i * dt; // Time to maturity for each day
        if (t <= 0) t = 1e-4; // Avoid zero maturity
        marketPrices[i] = blackScholesCall(S0, K, r, t, impliedVols[i]);
        if (!isfinite(marketPrices[i]) || marketPrices[i] <= 0) {
            cerr << "Invalid market price at index " << i << ": " << marketPrices[i] << endl;
            marketPrices[i] = marketPrices[i > 0 ? i - 1 : i + 1]; // Forward/backward fill
        }
    }

    // Step 2: Initialize parameters
    kappa = 2.0; // Initial guess
    theta = 0.1; // Initial guess
    sigma = 0.3; // Initial guess
    rho = -0.5;  // Initial guess
    v0 = impliedVols[0] * impliedVols[0]; // Initial variance

    // Gradient descent parameters
    double learningRate = 0.01;
    int maxIterations = 1000;
    double tolerance = 1e-6;
    double loss = 0.0;

    // Step 3: Gradient descent
    for (int iter = 0; iter < maxIterations; ++iter) {
        double gradKappa = 0.0, gradTheta = 0.0, gradSigma = 0.0, gradRho = 0.0, gradV0 = 0.0;
        loss = 0.0;

        // Compute gradients and loss
        for (size_t i = 0; i < marketPrices.size(); ++i) {
            double t = T - i * dt;
            if (t <= 0) continue;

            // Current Heston price
            double modelPrice = hestonOptionPrice(S0, K, r, t, kappa, theta, sigma, rho, v0);
            if (!isfinite(modelPrice)) {
                cerr << "Invalid model price at iter " << iter << ", index " << i << endl;
                modelPrice = marketPrices[i]; // Fallback
            }

            double error = modelPrice - marketPrices[i];
            loss += error * error;

            // Numerical gradients (finite differences)
            double eps = 1e-5;
            gradKappa += 2 * error * (hestonOptionPrice(S0, K, r, t, kappa + eps, theta, sigma, rho, v0) - modelPrice) / eps;
            gradTheta += 2 * error * (hestonOptionPrice(S0, K, r, t, kappa, theta + eps, sigma, rho, v0) - modelPrice) / eps;
            gradSigma += 2 * error * (hestonOptionPrice(S0, K, r, t, kappa, theta, sigma + eps, rho, v0) - modelPrice) / eps;
            gradRho   += 2 * error * (hestonOptionPrice(S0, K, r, t, kappa, theta, sigma, rho + eps, v0) - modelPrice) / eps;
            gradV0    += 2 * error * (hestonOptionPrice(S0, K, r, t, kappa, theta, sigma, rho, v0 + eps) - modelPrice) / eps;
        }
        loss /= marketPrices.size();
        if (iter % 100 == 0) {
            std::cout << "Iteration " << iter << ", Loss: " << loss << std::endl;
        }

        // Update parameters
        kappa -= learningRate * gradKappa;
        theta -= learningRate * gradTheta;
        sigma -= learningRate * gradSigma;
        rho   -= learningRate * gradRho;
        v0    -= learningRate * gradV0;

        // Apply bounds
        kappa = max(0.1, std::min(kappa, 10.0));
        theta = max(0.01, std::min(theta, 0.5));
        sigma = max(0.01, std::min(sigma, 1.0));
        rho   = max(-0.99, std::min(rho, 0.99));
        v0    = max(0.01, std::min(v0, 0.5));

        // Check Feller condition
        if (2 * kappa * theta <= sigma * sigma) {
            sigma = std::sqrt(1.9 * kappa * theta); // Adjust sigma
            cout << "Adjusted sigma to satisfy Feller condition: " << sigma << endl;
        }

        // Convergence check
        if (loss < tolerance) {
            cout << "Converged at iteration " << iter << endl;
            break;
        }
    }

    // Step 4: Validate parameters
    if (kappa <= 0 || theta <= 0 || sigma <= 0 || !std::isfinite(loss)) {
        cerr << "Invalid parameters, using defaults\n";
        kappa = 2.0;
        theta = v0;
        sigma = 0.3;
        rho = -0.5;
    }
}