#include "Financial_Analysis.h"

using namespace std;

Financial_Analysis::Financial_Analysis()
{
    //ctor
}

double Financial_Analysis::calc_volatility(vector<double>& close){
    vector<double> returns;
    for (size_t i = 1; i < min(close.size(), size_t(30)); ++i) {
        returns.push_back(log(close[i-1] / close[i]));
    }

    double mean = 0.0;
    for (double r : returns) mean += r;
    mean /= returns.size();

    double variance = 0.0;
    for (double r : returns) variance += pow((r - mean),2);
    variance /= (returns.size() - 1);

    return sqrt(variance * 252);
}