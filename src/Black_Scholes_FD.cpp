#include "Black_Scholes_FD.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <vector>
#include <iostream>

using namespace std;

Black_Scholes_FD::Black_Scholes_FD(double S0, double K, double T, double r, double sigma, int N_S, int N_t, int max_iter, bool is_call, bool use_arrested_newton):
    S0(S0), S_max(2*S0), T(T), r(r), sigma(sigma), K(K), N_S(N_S), N_t(N_t), max_iter(max_iter), is_call(is_call) {
    dS = S_max / (N_S - 1);
    dt = T / (N_t - 1);
    dtau = 1e-5; // Slightly larger step size for RK4
    V.resize(N_S, vector<double>(N_t, 0.0));
    res_prev.resize(N_S, vector<double>(N_t, 0.0));
    V_temp.resize(N_S, vector<double>(N_t, 0.0));
    k1.resize(N_S, vector<double>(N_t, 0.0));
    k2.resize(N_S, vector<double>(N_t, 0.0));
    k3.resize(N_S, vector<double>(N_t, 0.0));
    k4.resize(N_S, vector<double>(N_t, 0.0));

    if (is_call) {
        // Initialise grid at t = T
        for (int i = 0; i < N_S; ++i) {
            double S = i * dS;
            V[i][N_t - 1] = max(S - K, 0.0);
        }
        // Initialise boundaries
        for (int j = 0; j < N_t; ++j) {
            V[0][j] = 0.0; // S = 0
            double t = j * dt;
            V[N_S - 1][j] = S_max - K * exp(-r * (T - t)); // S = S_max
        }
        // Initial guess: Linear interpolation
        for (int j = 0; j < N_t - 1; ++j) {
            for (int i = 1; i < N_S - 1; ++i) {
                double S = i * dS;
                V[i][j] = max(S - K * exp(-r * (T - j * dt)), 0.0);
            }
        }
    } else {
        // Initialise grid at t = T
        for (int i = 0; i < N_S; ++i) {
            double S = i * dS;
            V[i][N_t - 1] = max(K - S, 0.0);
        }
        // Initialise boundaries
        for (int j = 0; j < N_t; ++j) {
            double t = j * dt;
            V[0][j] = K * exp(-r * T * (1.0 - t));
            V[N_S - 1][j] = 0;
        }
        // Initial guess: Linear interpolation
        for (int j = 0; j < N_t - 1; ++j) {
            for (int i = 1; i < N_S - 1; ++i) {
                double S = i * dS;
                V[i][j] = max(K - S, 0.0);
            }
        }
    }
}

double Black_Scholes_FD::deriv1_S(const vector<vector<double>>& f, int i, int j, bool forward) {
    if (forward) {
        return (f[i+1][j] - f[i][j]) / dS;
    }
    return (f[i + 1][j] - f[i - 1][j]) / (2 * dS);
}

double Black_Scholes_FD::deriv2_S(const vector<vector<double>>& f, int i, int j, bool forward) {
    if (forward) {
        return (f[i+2][j] - 2.0 * f[i+1][j] + f[i][j]) / (dS * dS);
    }
    return (f[i + 1][j] - 2 * f[i][j] + f[i - 1][j]) / (dS * dS);
}

double Black_Scholes_FD::deriv1_t(const vector<vector<double>>& f, int i, int j, bool forward) {
    if (forward) {
        return (f[i][j + 1] - f[i][j]) / dt;
    }
    return (f[i][j + 1] - f[i][j - 1]) / (2 * dt);
}

void Black_Scholes_FD::calc_eom(int i, int j, double& res, bool forward_t, bool forward_S, const vector<vector<double>>& V_in) {
    double S = i * dS;
    res = deriv1_t(V_in, i, j, forward_t) + 0.5 * sigma * sigma * S * S * deriv2_S(V_in, i, j, forward_S) + r * S * deriv1_S(V_in, i, j, forward_S) - r * V_in[i][j];
}

void Black_Scholes_FD::solve() {
   vector<vector<double>> res(N_S, vector<double>(N_t, 0.0));
   double mean_res = 0.0;
   int iter = 0;
    #pragma omp parallel num_threads(6) shared(V, res, res_prev, V_temp, k1, k2, k3, k4) private(iter)
    {
        for (iter = 0; iter < max_iter; iter++) {
            #pragma omp single
            mean_res = 0.0;
            // Compute residuals (k1)
            #pragma omp for collapse(2) reduction(+:mean_res)
            for (int i = 0; i < N_S - 1; i++) {
                for (int j = 0; j < N_t - 1; j++) {
                    bool forward_t = (j == 0);
                    bool forward_S = (i == 0);
                    calc_eom(i, j, k1[i][j], forward_t, forward_S, V);
                    res[i][j] = k1[i][j]; // Store for mean residual and arresting condition
                    mean_res += abs(res[i][j]);
                }
            }

            // RK4 Steps
            // Compute k2: V_temp = V + (dtau/2) * k1
            #pragma omp for collapse(2)
            for (int i = 0; i < N_S - 1; i++) {
                for (int j = 0; j < N_t - 1; j++) {
                    V_temp[i][j] = V[i][j] + (dtau / 2.0) * k1[i][j];
                }
            }
            // Apply boundary conditions to V_temp
            #pragma omp for
            for (int j = 0; j < N_t; ++j) {
                V_temp[0][j] = V[0][j]; // S = 0
                V_temp[N_S - 1][j] = V[N_S - 1][j]; // S = S_max
            }
            #pragma omp for
            for (int i = 0; i < N_S; ++i) {
                V_temp[i][N_t - 1] = V[i][N_t - 1]; // t = T
            }
            // Compute k2
            #pragma omp for collapse(2)
            for (int i = 0; i < N_S - 1; i++) {
                for (int j = 0; j < N_t - 1; j++) {
                    bool forward_t = (j == 0);
                    bool forward_S = (i == 0);
                    calc_eom(i, j, k2[i][j], forward_t, forward_S, V_temp);
                }
            }
            // Compute k3: V_temp = V + (dtau/2) * k2
            #pragma omp for collapse(2)
            for (int i = 0; i < N_S - 1; i++) {
                for (int j = 0; j < N_t - 1; j++) {
                    V_temp[i][j] = V[i][j] + (dtau / 2.0) * k2[i][j];
                }
            }
            // Apply boundary conditions to V_temp
            #pragma omp for
            for (int j = 0; j < N_t; ++j) {
                V_temp[0][j] = V[0][j];
                V_temp[N_S - 1][j] = V[N_S - 1][j];
            }
            #pragma omp for
            for (int i = 0; i < N_S; ++i) {
                V_temp[i][N_t - 1] = V[i][N_t - 1];
            }
            // Compute k3
            #pragma omp for collapse(2)
            for (int i = 0; i < N_S - 1; i++) {
                for (int j = 0; j < N_t - 1; j++) {
                    bool forward_t = (j == 0);
                    bool forward_S = (i == 0);
                    calc_eom(i, j, k3[i][j], forward_t, forward_S, V_temp);
                }
            }
            // Compute k4: V_temp = V + dtau * k3
            #pragma omp for collapse(2)
            for (int i = 0; i < N_S - 1; i++) {
                for (int j = 0; j < N_t - 1; j++) {
                    V_temp[i][j] = V[i][j] + dtau * k3[i][j];
                }
            }
            // Apply boundary conditions to V_temp
            #pragma omp for
            for (int j = 0; j < N_t; ++j) {
                V_temp[0][j] = V[0][j];
                V_temp[N_S - 1][j] = V[N_S - 1][j];
            }
            #pragma omp for
            for (int i = 0; i < N_S; ++i) {
                V_temp[i][N_t - 1] = V[i][N_t - 1];
            }
            // Compute k4
            #pragma omp for collapse(2)
            for (int i = 0; i < N_S - 1; i++) {
                for (int j = 0; j < N_t - 1; j++) {
                    bool forward_t = (j == 0);
                    bool forward_S = (i == 0);
                    calc_eom(i, j, k4[i][j], forward_t, forward_S, V_temp);
                }
            }
            // Apply RK4 update with arresting condition
            #pragma omp for collapse(2)
            for (int i = 0; i < N_S - 1; i++) {
                for (int j = 0; j < N_t - 1; j++) {
                    double update = (dtau / 6.0) * (k1[i][j] + 2.0 * k2[i][j] + 2.0 * k3[i][j] + k4[i][j]);
                    double accel = res[i][j] - res_prev[i][j];
                    if (iter > 0 && update * accel < 0) {
                        res_prev[i][j] = res[i][j]; // Update previous residual
                        continue; // Skip update if arresting condition is met
                    }
                    V[i][j] += update;
                    res_prev[i][j] = res[i][j]; // Update previous residual
                }
            }

            #pragma omp single
            {
                if (iter % 1000 == 0) {
                    cout << "Iteration: " << iter << ", Mean Residual: " << mean_res / ((N_S - 1) * (N_t - 1)) << endl;
                }
            }
        }
    }
}

double Black_Scholes_FD::get_option_price(double S0) {
    int i = (int)floor(S0 / dS);
    if (i < 0 || i >= N_S - 1) {
        return 0;
    }
    double S_i = i * dS;
    double S_ip1 = (i + 1) * dS;
    return V[i][0] + ((V[i + 1][0] - V[i][0]) / (S_ip1 - S_i)) * (S0 - S_i);
}