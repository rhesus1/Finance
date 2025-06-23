#include "Black_Scholes_FD.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <vector>
#include <memory>

using namespace std;

Black_Scholes_FD::Black_Scholes_FD(double S0, double K, double T, double r, double sigma, int N_S, int N_t, int max_iter, bool is_call, bool use_arrested_newton)
    : S0(S0), S_max(2 * S0), T(T), r(r), sigma(sigma), K(K), N_S(N_S), N_t(N_t), max_iter(max_iter), is_call(is_call), use_arrested_newton(use_arrested_newton) {
    dS = S_max / (N_S - 1);
    dt = T / (N_t - 1);
    dtau = 1e-4;
    sigma2 = sigma * sigma;
    dS2 = dS * dS;
    dtau_sixth = dtau / 6.0;
    V.resize(N_S, vector<double>(N_t, 0.0));
    V_temp.resize(N_S, vector<double>(N_t, 0.0));
    k_temp.resize(N_S, vector<double>(N_t, 0.0));
    res.resize(N_S, vector<double>(N_t, 0.0));
    res_prev.resize(N_S, vector<double>(N_t, 0.0));

    // Initialize grid
    if (is_call) {
        for (int i = 0; i < N_S; ++i) {
            double S = i * dS;
            V[i][N_t - 1] = max(S - K, 0.0);
        }
        for (int j = 0; j < N_t; ++j) {
            V[0][j] = 0.0;
            double t = j * dt;
            V[N_S - 1][j] = S_max - K * exp(-r * (T - t));
        }
        for (int j = 0; j < N_t - 1; ++j) {
            for (int i = 1; i < N_S - 1; ++i) {
                double S = i * dS;
                V[i][j] = max(S - K, 0.0);
            }
        }
    } else {
        for (int i = 0; i < N_S; ++i) {
            double S = i * dS;
            V[i][N_t - 1] = max(K - S, 0.0);
        }
        for (int j = 0; j < N_t; ++j) {
            double t = j * dt;
            V[0][j] = K * exp(-r * (T - t));
            V[N_S - 1][j] = 0.0;
        }
        for (int j = 0; j < N_t - 1; ++j) {
            for (int i = 1; i < N_S - 1; ++i) {
                double S = i * dS;
                V[i][j] = max(K - S, 0.0);
            }
        }
    }
}

double Black_Scholes_FD::deriv1_S(const vector<vector<double>>& f, int i, int j) {
    return i == 0 ? (f[i + 1][j] - f[i][j]) / dS :
           i == N_S - 1 ? (f[i][j] - f[i - 1][j]) / dS :
           (f[i + 1][j] - f[i - 1][j]) / (2 * dS);
}

double Black_Scholes_FD::deriv2_S(const vector<vector<double>>& f, int i, int j) {
    return i == 0 ? (f[i + 2][j] - 2.0 * f[i + 1][j] + f[i][j]) / dS2 :
           i == N_S - 1 ? (f[i][j] - 2.0 * f[i - 1][j] + f[i - 2][j]) / dS2 :
           (f[i + 1][j] - 2.0 * f[i][j] + f[i - 1][j]) / dS2;
}

double Black_Scholes_FD::deriv1_t(const vector<vector<double>>& f, int i, int j) {
    return j == 0 ? (f[i][j + 1] - f[i][j]) / dt :
           j == N_t - 1 ? (f[i][j] - f[i][j - 1]) / dt :
           (f[i][j + 1] - f[i][j - 1]) / (2 * dt);
}

void Black_Scholes_FD::calc_eom(int i, int j, double& res, const vector<vector<double>>& V_in) {
    double S = i * dS;
    double d1S = deriv1_S(V_in, i, j);
    double d2S = deriv2_S(V_in, i, j);
    double d1t = deriv1_t(V_in, i, j);
    res = d1t + 0.5 * sigma2 * S * S * d2S + r * S * d1S - r * V_in[i][j];
}

void Black_Scholes_FD::apply_boundary_conditions(vector<vector<double>>& V_out) {
    #pragma omp parallel for num_threads(omp_get_num_procs())
    for (int j = 0; j < N_t; ++j) {
        V_out[0][j] = V[0][j];
        V_out[N_S - 1][j] = V[N_S - 1][j];
    }
    #pragma omp parallel for num_threads(omp_get_num_procs())
    for (int i = 0; i < N_S; ++i) {
        V_out[i][N_t - 1] = V[i][N_t - 1];
    }
}

void Black_Scholes_FD::compute_rk_stage(const vector<vector<double>>& V_in, vector<vector<double>>& V_out, double scale) {
    #pragma omp parallel for collapse(2) num_threads(omp_get_num_procs()) schedule(static)
    for (int i = 0; i < N_S - 1; ++i) {
        for (int j = 0; j < N_t - 1; ++j) {
            calc_eom(i, j, k_temp[i][j], V_in);
            V_out[i][j] = V[i][j] + scale * k_temp[i][j];
        }
    }
    apply_boundary_conditions(V_out);
}

void Black_Scholes_FD::solve() {
    const double error_tol = 1e-6;
    for (int iter = 0; iter < max_iter; ++iter) {
        double mean_res = 0.0;
        #pragma omp parallel for collapse(2) num_threads(omp_get_num_procs()) reduction(+:mean_res) schedule(static)
        for (int i = 0; i < N_S - 1; ++i) {
            for (int j = 0; j < N_t - 1; ++j) {
                calc_eom(i, j, k_temp[i][j], V);
                res[i][j] = k_temp[i][j];
                mean_res += abs(res[i][j]);
            }
        }
        mean_res /= (N_S - 1) * (N_t - 1);
        if (mean_res < error_tol) break;

        compute_rk_stage(V, V_temp, dtau / 2.0);
        compute_rk_stage(V_temp, V_temp, dtau / 2.0);
        compute_rk_stage(V_temp, V_temp, dtau);
        compute_rk_stage(V_temp, V_temp, dtau);

        #pragma omp parallel for collapse(2) num_threads(omp_get_num_procs()) schedule(static)
        for (int i = 0; i < N_S - 1; ++i) {
            for (int j = 0; j < N_t - 1; ++j) {
                double update = dtau_sixth * (k_temp[i][j] + 2.0 * k_temp[i][j] + 2.0 * k_temp[i][j] + k_temp[i][j]);
                double accel = res[i][j] - res_prev[i][j];
                double apply_update = (use_arrested_newton && iter > 0 && update * accel < 0) ? 0.0 : update;
                V[i][j] += apply_update;
                res_prev[i][j] = res[i][j];
            }
        }
    }
}

void Black_Scholes_FD::solve_american() {
    const double error_tol = 1e-6;
    for (int iter = 0; iter < max_iter; ++iter) {
        double mean_res = 0.0;
        #pragma omp parallel for collapse(2) num_threads(omp_get_num_procs()) reduction(+:mean_res) schedule(static)
        for (int i = 0; i < N_S - 1; ++i) {
            for (int j = 0; j < N_t - 1; ++j) {
                calc_eom(i, j, k_temp[i][j], V);
                res[i][j] = k_temp[i][j];
                mean_res += abs(res[i][j]);
            }
        }
        mean_res /= (N_S - 1) * (N_t - 1);
        if (mean_res < error_tol) break;

        compute_rk_stage(V, V_temp, dtau / 2.0);
        compute_rk_stage(V_temp, V_temp, dtau / 2.0);
        compute_rk_stage(V_temp, V_temp, dtau);
        compute_rk_stage(V_temp, V_temp, dtau);

        #pragma omp parallel for collapse(2) num_threads(omp_get_num_procs()) schedule(static)
        for (int i = 0; i < N_S - 1; ++i) {
            for (int j = 0; j < N_t - 1; ++j) {
                double update = dtau_sixth * (k_temp[i][j] + 2.0 * k_temp[i][j] + 2.0 * k_temp[i][j] + k_temp[i][j]);
                double accel = res[i][j] - res_prev[i][j];
                double apply_update = (use_arrested_newton && iter > 0 && update * accel < 0) ? 0.0 : update;
                V[i][j] += apply_update;
                double S = i * dS;
                double intrinsic = is_call ? max(S - K, 0.0) : max(K - S, 0.0);
                V[i][j] = max(V[i][j], intrinsic);
                res_prev[i][j] = res[i][j];
            }
        }
    }
}

double Black_Scholes_FD::get_option_price(double S0) {
    int i = static_cast<int>(floor(S0 / dS));
    if (i < 0 || i >= N_S - 1) return 0.0;
    double S_i = i * dS;
    double S_ip1 = (i + 1) * dS;
    return V[i][0] + ((V[i + 1][0] - V[i][0]) / (S_ip1 - S_i)) * (S0 - S_i);
}