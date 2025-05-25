#include "Linear_Regression.h"

Linear_Regression::Linear_Regression(double lr, int iter)
    : dt(lr), max_iter(iter){
    weights = {0.0, 0.0};
    min_x = max_x = min_y = max_y = 0.0;
}

void Linear_Regression::train(const vector<double>& x, const vector<double>& y){
    int n = x.size();

    min_x = *min_element(x.begin(), x.end());
    max_x = *max_element(x.begin(), x.end());
    min_y = *min_element(y.begin(), y.end());
    max_y = *max_element(y.begin(), y.end());

    vector<double> x_norm(n), y_norm(n);
    for (int i = 0; i < n; ++i) {
        x_norm[i] = (x[i] - min_x) / (max_x - min_x);
        y_norm[i] = (y[i] - min_y) / (max_y - min_y);
    }

    double grad0 = 0.0;
    double grad1 = 0.0;


    for (int i = 0; i < max_iter; i++){
        double mse = 0.0;
        grad0 = 0.0;
        grad1 = 0.0;

        #pragma omp parallel for num_threads(4) reduction(+:mse,grad0,grad1)
        for (int j = 0; j < n; j ++){
            double pred = weights[0] + weights[1] * x_norm[j];
            double err = pred - y_norm[j];
            mse += err * err;
            grad0 += err;
            grad1 += err * x_norm[j];
        }
        mse /= n;
        grad0 /= n;
        grad1 /= n;
        weights[0] -= dt * grad0;
        weights[1] -= dt * grad1;
    }
}

double Linear_Regression::predict(double x){
    double x_norm = (x - min_x) / (max_x - min_x);
    double y_norm = weights[0] + weights[1] * x_norm;
    return y_norm * (max_y - min_y) + min_y;
}

