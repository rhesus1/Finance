#include "Financial_Analysis.h"

using namespace std;

Financial_Analysis::Financial_Analysis()
{
    //ctor
}
Financial_Analysis::Financial_Analysis() {}

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
double Financial_Analysis::impliedVolatility(double S, double K, double r, double T, double optionPrice, double tol = 1e-6, int maxIter = 100) {
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
    if (x.size() != y.size()) return 0.0;
    double meanX = computeMean(x);
    double meanY = computeMean(y);
    double sumCov = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
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

//kappa (Speed of mean reversion), theta (Long term variance), sigma (volatility of variance)
void Financial_Analysis::estimateHestonParameters(const vector<double>& variance, double dt, double& kappa, double& theta, double& sigma) {
    vector<double> y(variance.size() - 1);
    vector<double> x(variance.size() - 1);
    for (size_t i = 0; i < variance.size() - 1; ++i) {
        double sqrtVt = sqrt(variance[i]);
        y[i] = (variance[i + 1] - variance[i]) / sqrtVt;
        x[i] = -sqrtVt * dt;
    }
    double meanX = computeMean(x);
    double meanY = computeMean(y);
    double sumXY = 0.0, sumXX = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sumXY += (x[i] - meanX) * (y[i] - meanY);
        sumXX += (x[i] - meanX) * (x[i] - meanX);
    }
    kappa = -sumXY / sumXX;
    theta = meanY / (kappa * dt);
    double sumSquaredResiduals = 0.0;
    for (size_t i = 0; i < variance.size() - 1; ++i) {
        double expected = kappa * (theta - variance[i]) * dt;
        double residual = (variance[i + 1] - variance[i]) - expected;
        sumSquaredResiduals += residual * residual / variance[i];
    }
    sigma = sqrt(sumSquaredResiduals / (variance.size() - 1)) / sqrt(dt);
}