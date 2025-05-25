#include "Black_Scholes_FD.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <vector>
#include <iostream>

using namespace std;

Black_Scholes_FD::Black_Scholes_FD(double S0, double K, double T, double r, double sigma, int N_S, int N_t, int max_iter, bool is_call):
    S0(S0), S_max(2*S0), T(T), r(r), sigma(sigma), K(K), N_S(N_S), N_t(N_t), max_iter(max_iter), is_call(is_call){
    dS = S_max / (N_S - 1);
    dt = T / (N_t - 1);
    dtau = 1e-5; // Gradient flow step size
    V.resize(N_S, vector<double>(N_t, 0.0));

    if(is_call){
        // Initialize grid at t = T
        for (int i = 0; i < N_S; ++i) {
            double S = i * dS;
            V[i][N_t - 1] = max(S - K, 0.0);
        }
        // Initialize boundaries
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
    }else{
        // Initialize grid at t = T
        for (int i = 0; i < N_S; ++i) {
            double S = i * dS;
            V[i][N_t - 1] = max(K - S, 0.0);
        }
        // Initialize boundaries
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


double Black_Scholes_FD::deriv1_S(const vector<vector<double>>& f, int i, int j) {
    return (f[i + 1][j] - f[i - 1][j]) / (2 * dS);
}

double Black_Scholes_FD::deriv2_S(const vector<vector<double>>& f, int i, int j) {
    return (f[i + 1][j] - 2 * f[i][j] + f[i - 1][j]) / (dS * dS);
}

double Black_Scholes_FD::deriv1_t(const vector<vector<double>>& f, int i, int j, bool forward) {
    if (forward) {
        return (f[i][j + 1] - f[i][j]) / dt;
    }
    return (f[i][j + 1] - f[i][j - 1]) / (2 * dt);
}

void Black_Scholes_FD::calc_eom(int i, int j, double& res, bool forward_t){
    double S = i * dS;
    res = deriv1_t(V, i, j, forward_t) + 0.5 * sigma * sigma * S * S * deriv2_S(V, i, j) + r * S * deriv1_S(V,i,j) - r*V[i][j];
}

void Black_Scholes_FD::solve(){
   vector<vector<double>> res(N_S, vector<double>(N_t, 0.0));
   double max_res = 0.0;
   #pragma omp parallel num_threads(4) shared(V, res)
   {
        for(int iter = 0; iter < max_iter; iter++){
            max_res = 0.0;
            #pragma omp for collapse(2)
            for (int i = 1; i < N_S - 1; i++){
                for(int j = 0; j < N_t - 1; j++){
                    bool forward_t = (j == 0);
                    calc_eom(i,j,res[i][j],forward_t);
                    #pragma omp critical
                    max_res = max(max_res, abs(res[i][j]));
                }
            }
            #pragma omp for collapse(2)
            for(int i = 1; i < N_S - 1; i++){
                for(int j = 0; j < N_t - 1; j++){
                    V[i][j] += dtau * res[i][j];
                }
            }
        }
        #pragma omp single
        cout << "Max Residual: " << max_res << endl << flush;
   }
}

double Black_Scholes_FD::get_option_price(double S0){
    int i = (int)floor(S0 / dS);
    if(i < 0 || i >= N_S - 1){
        return 0;
    }
    cout << i << endl;
    double S_i = i * dS;
    double S_ip1 = (i+1)*dS;
    return V[i][0] + ((V[i+1][0] - V[i][0]) / (S_ip1 - S_i)) * (S0 - S_i);
}