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
        double impliedVolatility(double S, double K, double r, double T, double optionPrice, double tol = 1e-6, int maxIter = 100);
        double computeVariance(const vector<double>& data);
        double computeCovariance(const vector<double>& x, const vector<double>& y);
        double computeCorrelation(const vector<double>& x, const vector<double>& y);
        void estimateHestonParameters(const vector<double>& variance, double dt, double& kappa, double& theta, double& sigma);
    protected:

    private:
        double computeMean(const vector<double>& data);
        double blackScholesCall(double S, double K, double r, double T, double sigma);
};

#endif // FINANCIAL_ANALYSIS_H


