#ifndef FINANCIAL_ANALYSIS_H
#define FINANCIAL_ANALYSIS_H
#include <vector>
#include <cmath>

using namespace std;

class Financial_Analysis
{
    public:
        Financial_Analysis();
        double calc_volatility(vector<double>& close);
    protected:

    private:
};

#endif // FINANCIAL_ANALYSIS_H
