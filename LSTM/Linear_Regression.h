#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H
#include <vector>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <algorithm>

using namespace std;

// Class for linear regression model
class Linear_Regression {
public:
    // Constructor to initialize learning rate and max iterations
    Linear_Regression(double lr, int iter);

    // Train the model on input data
    void train(const vector<double>& x, const vector<double>& y);

    // Predict output for given input
    double predict(double x);

private:
    vector<double> weights; // Model weights
    double dt;             // Learning rate
    int max_iter;          // Maximum training iterations
    double min_x, max_x;   // Min/max input values for normalization
    double min_y, max_y;   // Min/max output values for normalization
};

#endif // LINEAR_REGRESSION_H