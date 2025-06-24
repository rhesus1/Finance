#ifndef FINANCIAL_ANALYSIS_H
#define FINANCIAL_ANALYSIS_H
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>


using namespace std;

class Financial_Analysis
{
    public:
        Financial_Analysis();
        double calc_volatility(vector<double>& close, int duration);
        vector<double> computeDailyVariance(vector<double>& close, int duration);
        double impliedVolatility(double S, double K, double r, double T, double optionPrice);
        double computeVariance(const vector<double>& data);
        double computeCovariance(const vector<double>& x, const vector<double>& y);
        double computeCorrelation(const vector<double>& x, const vector<double>& y);
        void estimateHestonParameters(const vector<double>& impliedVols, double S0, double K, double r, double T, double dt,
                             double& kappa, double& theta, double& sigma, double& rho, double& v0);
    protected:

    private:
        double computeMean(const vector<double>& data);
        double blackScholesCall(double S, double K, double r, double T, double sigma);
        double hestonOptionPrice(double S0, double K, double r, double T,
                                            double kappa, double theta, double sigma, double rho, double v0);

};

#endif // FINANCIAL_ANALYSIS_H


