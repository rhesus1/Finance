#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H
#include <vector>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <algorithm>

using namespace std;

class Linear_Regression
{
    public:
        Linear_Regression(double lr, int iter);
        void train(const vector<double>& x, const vector<double>& y);
        double predict(double x);
    protected:

    private:
        vector<double> weights;
        double dt;
        int max_iter;
        double min_x, max_x, min_y, max_y;
};

#endif // LINEAR_REGRESSION_H
