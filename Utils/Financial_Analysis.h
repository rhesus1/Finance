#ifndef FINANCIAL_ANALYSIS_H
#define FINANCIAL_ANALYSIS_H
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>

using namespace std;

// Class for financial data analysis and option pricing
class Financial_Analysis {
public:
    // Constructor
    Financial_Analysis();

    // Calculate historical volatility from closing prices
    double calc_volatility(vector<double>& close, int duration);

    // Compute daily variance from closing prices
    vector<double> computeDailyVariance(vector<double>& close, int duration);

    // Estimate implied volatility from option price
    double impliedVolatility(double S, double K, double r, double T, double optionPrice);

    // Compute variance of a dataset
    double computeVariance(const vector<double>& data);

    // Compute covariance between two datasets
    double computeCovariance(const vector<double>& x, const vector<double>& y);

    // Compute correlation between two datasets
    double computeCorrelation(const vector<double>& x, const vector<double>& y);

    // Estimate Heston model parameters from implied volatilities
    void estimateHestonParameters(const vector<double>& impliedVols, double S0, double K, double r, double T, double dt,
                                 double& kappa, double& theta, double& sigma, double& rho, double& v0);

private:
    // Compute mean of a dataset
    double computeMean(const vector<double>& data);

    // Calculate Black-Scholes call option price
    double blackScholesCall(double S, double K, double r, double T, double sigma);

    // Calculate Heston model option price
    double hestonOptionPrice(double S0, double K, double r, double T,
                             double kappa, double theta, double sigma, double rho, double v0);
};

#endif // FINANCIAL_ANALYSIS_H