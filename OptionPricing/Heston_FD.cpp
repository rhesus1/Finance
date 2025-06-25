#include "Heston_FD.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <vector>
#include <iostream>
#include <limits>

using namespace std;

// Constructor initializes finite difference grid for Heston model
Heston_FD::Heston_FD(double S0, double K, double T, double r, double v0, double kappa, double theta, double xi, double rho,
                     int N_S, int N_v, int N_t, int max_iter, bool is_call, bool is_eu)
    : S0(S0), K(K), T(T), r(r), v0(v0), kappa(kappa), theta(theta), xi(xi), rho(rho), N_S(N_S), N_v(N_v), N_t(N_t), max_iter(max_iter), is_call(is_call), is_solved(false), solved_is_call(is_call), is_eu(is_eu) {
    // Validate Feller condition
    if (2 * kappa * theta <= xi * xi) {
        cout << "Warning: Feller condition (2 * kappa * theta > xi^2) not satisfied. Numerical instability possible." << endl;
    }

    // Set grid parameters
    S_max = 2 * S0; // Increased for stability
    v_max = 4 * theta; // Increased for better variance coverage
    dS = S_max / (N_S - 1);
    dv = v_max / (N_v - 1);
    dt = T / (N_t - 1);
    dtau = 1e-4; // Reduced for stability

    // Resize grids
    V.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    V_temp.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    k1.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    k2.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    k3.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    k4.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    res.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    res_prev.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));

    // Initialize grid for call or put
    Black_Scholes BS;
    for (int i = 0; i < N_S; ++i) {
        double S = i * dS;
        for (int k = 0; k < N_v; ++k) {
            double v = k * dv;
            for (int j = 0; j < N_t; ++j) {
                double t = j * dt;
                double time_to_maturity = T - t;
                if (j == N_t - 1 || time_to_maturity <= 0 || S <= 0) {
                    V[i][k][j] = is_call ? max(S - K, 0.0) : max(K - S, 0.0);
                } else {
                    double vol = sqrt(max(v, 0.01));
                    V[i][k][j] = is_call ? BS.call(S, K, time_to_maturity, r, 0.0, vol)
                                         : BS.put(S, K, time_to_maturity, r, 0.0, vol);
                    if (isnan(V[i][k][j]) || V[i][k][j] < 0) {
                        V[i][k][j] = is_call ? max(S - K * exp(-r * time_to_maturity), 0.0)
                                             : max(K * exp(-r * time_to_maturity) - S, 0.0);
                    }
                }
            }
        }
    }

    // Smooth interior points
    #pragma omp parallel for collapse(3) num_threads(omp_get_num_procs()) schedule(static)
    for (int j = 0; j < N_t; ++j) {
        for (int i = 1; i < N_S - 1; ++i) {
            for (int k = 1; k < N_v - 1; ++k) {
                V[i][k][j] = 0.25 * (V[i][k][j] + V[i+1][k][j] + V[i-1][k][j] + V[i][k+1][j]);
            }
        }
    }

    // Apply boundary conditions
    apply_boundary_conditions(V);
}

// Computes first derivative with respect to stock price
double Heston_FD::deriv1_S(const vector<vector<vector<double>>>& f, int i, int k, int j) {
    if (i == 0) return (f[i + 1][k][j] - f[i][k][j]) / dS;
    if (i == N_S - 1) return (f[i][k][j] - f[i - 1][k][j]) / dS;
    return (f[i + 1][k][j] - f[i - 1][k][j]) / (2 * dS);
}

// Computes second derivative with respect to stock price
double Heston_FD::deriv2_S(const vector<vector<vector<double>>>& f, int i, int k, int j) {
    if (i == 0) return (f[i + 2][k][j] - 2 * f[i + 1][k][j] + f[i][k][j]) / (dS * dS);
    if (i == N_S - 1) return (f[i][k][j] - 2 * f[i - 1][k][j] + f[i - 2][k][j]) / (dS * dS);
    return (f[i + 1][k][j] - 2 * f[i][k][j] + f[i - 1][k][j]) / (dS * dS);
}

// Computes first derivative with respect to variance
double Heston_FD::deriv1_v(const vector<vector<vector<double>>>& f, int i, int k, int j) {
    if (k == 0) return (f[i][k + 1][j] - f[i][k][j]) / dv;
    if (k == N_v - 1) return (f[i][k][j] - f[i][k - 1][j]) / dv;
    return (f[i][k + 1][j] - f[i][k - 1][j]) / (2 * dv);
}

// Computes second derivative with respect to variance
double Heston_FD::deriv2_v(const vector<vector<vector<double>>>& f, int i, int k, int j) {
    if (k == 0) return (f[i][k + 2][j] - 2 * f[i][k + 1][j] + f[i][k][j]) / (dv * dv);
    if (k == N_v - 1) return (f[i][k][j] - 2 * f[i][k - 1][j] + f[i][k - 2][j]) / (dv * dv);
    return (f[i][k + 1][j] - 2 * f[i][k][j] + f[i][k - 1][j]) / (dv * dv);
}

// Computes mixed derivative with respect to stock price and variance
double Heston_FD::deriv2_Sv(const vector<vector<vector<double>>>& f, int i, int k, int j) {
    if (i == 0 || i == N_S - 1 || k == 0 || k == N_v - 1) {
        double di = (i == 0) ? 1 : (i == N_S - 1) ? -1 : 0;
        double dk = (k == 0) ? 1 : (k == N_v - 1) ? -1 : 0;
        return (f[i + di][k + dk][j] - f[i + di][k][j] - f[i][k + dk][j] + f[i][k][j]) / (dS * dv);
    }
    return (f[i + 1][k + 1][j] - f[i + 1][k - 1][j] - f[i - 1][k + 1][j] + f[i - 1][k - 1][j]) / (4 * dS * dv);
}

// Computes first derivative with respect to time
double Heston_FD::deriv1_t(const vector<vector<vector<double>>>& f, int i, int k, int j) {
    if (j == 0) return (f[i][k][j + 1] - f[i][k][j]) / dt;
    if (j == N_t - 1) return (f[i][k][j] - f[i][k][j - 1]) / dt;
    return (f[i][k][j + 1] - f[i][k][j - 1]) / (2 * dt);
}

// Computes Heston PDE residual
void Heston_FD::calc_eom(int i, int k, int j, double& res, const vector<vector<vector<double>>>& f) {
    double S = i * dS;
    double v = max(k * dv, 1e-6); // Stronger variance clipping
    const double max_res = 1e10; // Prevent overflow
    res = deriv1_t(f, i, k, j) +
          0.5 * v * S * S * deriv2_S(f, i, k, j) +
          rho * xi * v * S * deriv2_Sv(f, i, k, j) +
          0.5 * xi * xi * v * deriv2_v(f, i, k, j) +
          r * S * deriv1_S(f, i, k, j) +
          kappa * (theta - v) * deriv1_v(f, i, k, j) -
          r * f[i][k][j];
    if (isnan(res) || abs(res) > max_res) res = (res > 0 ? max_res : -max_res);
}

// Applies boundary conditions to the grid
void Heston_FD::apply_boundary_conditions(vector<vector<vector<double>>>& V_out) {
    #pragma omp parallel for collapse(2) num_threads(omp_get_num_procs()) schedule(static)
    for (int j = 0; j < N_t; ++j) {
        double t = j * dt;
        for (int k = 0; k < N_v; ++k) {
            V_out[0][k][j] = is_call ? 0.0 : max(K * exp(-r * (T - t)), 0.0);
            V_out[N_S - 1][k][j] = is_call ? max(S_max - K * exp(-r * (T - t)), 0.0) : 0.0;
        }
    }
    #pragma omp parallel for collapse(2) num_threads(omp_get_num_procs()) schedule(static)
    for (int j = 0; j < N_t; ++j) {
        for (int i = 0; i < N_S; ++i) {
            V_out[i][0][j] = V_out[i][1][j];
            V_out[i][N_v - 1][j] = is_call ? max(i * dS - K * exp(-r * (T - j * dt)), 0.0) : max(K * exp(-r * (T - j * dt)) - i * dS, 0.0);
        }
    }
    #pragma omp parallel for collapse(2) num_threads(omp_get_num_procs()) schedule(static)
    for (int i = 0; i < N_S; ++i) {
        for (int k = 0; k < N_v; ++k) {
            V_out[i][k][N_t - 1] = is_call ? max(i * dS - K, 0.0) : max(K - i * dS, 0.0);
        }
    }
}

// Computes one Runge-Kutta stage
void Heston_FD::compute_rk_stage(const vector<vector<vector<double>>>& V_in, vector<vector<vector<double>>>& V_out, double scale, vector<vector<vector<double>>>& k_stage) {
    #pragma omp parallel for collapse(3) num_threads(omp_get_num_procs()) schedule(static)
    for (int i = 0; i < N_S - 1; ++i) {
        for (int k = 0; k < N_v - 1; ++k) {
            for (int j = 0; j < N_t - 1; ++j) {
                calc_eom(i, k, j, k_stage[i][k][j], V_in);
                V_out[i][k][j] = V[i][k][j] + scale * k_stage[i][k][j]; // Backward PDE
                if (V_out[i][k][j] < 0) V_out[i][k][j] = 0.0;
            }
        }
    }
}
void Heston_FD::solve(bool is_call) {
    const double error_tol = 1e-6;
    const double dtau_sixth_init = dtau / 6.0;
    bool use_arrested_newton = true;
    double prev_mean_res = 1e10;
    is_solved = true;
    solved_is_call = is_call;
    this->is_call = is_call;

    // Adaptive time step parameters
    double dtau = this->dtau; // Current time step
    const double dtau_min = dtau * 0.01; // Minimum time step
    const double dtau_max = dtau * 2.0; // Maximum time step
    const double res_change_tol = 1e-8; // Threshold for small residual change
    double prev_prev_mean_res = prev_mean_res;
    double initial_mean_res = 0.0; // To store initial residual for scaling

    // Compute initial residual to set adaptive threshold
    double mean_res = 0.0;
    #pragma omp parallel for collapse(3) num_threads(omp_get_num_procs()) reduction(+:mean_res) schedule(static)
    for (int i = 0; i < N_S - 1; ++i) {
        for (int k = 0; k < N_v - 1; ++k) {
            for (int j = 0; j < N_t - 1; ++j) {
                calc_eom(i, k, j, k1[i][k][j], V);
                res[i][k][j] = k1[i][k][j];
                mean_res += abs(res[i][k][j]);
            }
        }
    }
    mean_res /= (N_S - 1) * (N_v - 1) * (N_t - 1);
    initial_mean_res = mean_res;

    for (int iter = 0; iter < max_iter; ++iter) {
        mean_res = 0.0;
        double dtau_sixth = dtau / 6.0; // Update dtau_sixth based on current dtau

        // Compute residuals
        #pragma omp parallel for collapse(3) num_threads(omp_get_num_procs()) reduction(+:mean_res) schedule(static)
        for (int i = 0; i < N_S - 1; ++i) {
            for (int k = 0; k < N_v - 1; ++k) {
                for (int j = 0; j < N_t - 1; ++j) {
                    calc_eom(i, k, j, k1[i][k][j], V);
                    res[i][k][j] = k1[i][k][j];
                    mean_res += abs(res[i][k][j]);
                }
            }
        }
        mean_res /= (N_S - 1) * (N_v - 1) * (N_t - 1);

        // Check for convergence or divergence
        if (mean_res < error_tol || iter == max_iter - 1) {
            #pragma omp critical
            cout << "Converged at iteration " << iter << ", Mean Residual: " << mean_res << ", Time Step: " << dtau << endl;
            break;
        }
        if (mean_res > prev_mean_res * 5 || isnan(mean_res) || isinf(mean_res)) {
            #pragma omp critical
            cout << "Error: Solver diverging at iteration " << iter << ", Mean Residual: " << mean_res << ", Time Step: " << dtau << endl;
            is_solved = false;
            break;
        }

        // Adaptive time step adjustment
        if (iter > 1) {
            double res_change = abs(mean_res - prev_mean_res) / (prev_mean_res + 1e-10);
            // Use a dynamic threshold based on initial residual
            if (mean_res < initial_mean_res * 0.01 && res_change < res_change_tol) {
                dtau = max(dtau_min, dtau * 0.5); // Decrease time step
                //#pragma omp critical
                //cout << "Decreasing time step at iteration " << iter << ", res_change: " << res_change << ", dtau: " << dtau << endl;
            } else if (res_change > res_change_tol * 10) {
                dtau = min(dtau_max, dtau * 1.5); // Increase time step
                //#pragma omp critical
                //cout << "Increasing time step at iteration " << iter << ", res_change: " << res_change << ", dtau: " << dtau << endl;
            }
            dtau_sixth = dtau / 6.0; // Update dtau_sixth
        }
        prev_prev_mean_res = prev_mean_res;
        prev_mean_res = mean_res;

        // Perform Runge-Kutta stages
        V_temp = V; // Copy V for intermediate stages
        compute_rk_stage(V, V_temp, dtau / 2.0, k1);
        compute_rk_stage(V_temp, V_temp, dtau / 2.0, k2);
        compute_rk_stage(V_temp, V_temp, dtau, k3);
        compute_rk_stage(V_temp, V_temp, dtau, k4);

        // Update solution
        #pragma omp parallel for collapse(3) num_threads(omp_get_num_procs()) schedule(static)
        for (int i = 0; i < N_S - 1; ++i) {
            for (int k = 0; k < N_v - 1; ++k) {
                for (int j = 0; j < N_t - 1; ++j) {
                    double update = dtau_sixth * (k1[i][k][j] + 2.0 * k2[i][k][j] + 2.0 * k3[i][k][j] + k4[i][k][j]);
                    double accel = res[i][k][j] - res_prev[i][k][j];
                    double apply_update = (use_arrested_newton && iter > 0 && update * accel < 0) ? update * 0.5 : update;
                    V[i][k][j] += apply_update;
                    if (V[i][k][j] < 0) V[i][k][j] = 0.0;
                    res_prev[i][k][j] = res[i][k][j];
                }
            }
        }
        apply_boundary_conditions(V);

        // Log iteration progress
        /*if (iter % 1000 == 0) {
            #pragma omp critical
            cout << "Iteration: " << iter << ", Mean Residual: " << mean_res << ", Time Step: " << dtau << endl;
        }*/
    }
}

// Solves Heston PDE for American options
void Heston_FD::solve_american() {
    const double error_tol = 1e-6;
    const double dtau_sixth = dtau / 6.0;
    bool use_arrested_newton = true;
    double prev_mean_res = 1e10;
    is_solved = true;
    solved_is_call = is_call;
    this->is_call = is_call;

    for (int iter = 0; iter < max_iter; ++iter) {
        double mean_res = 0.0;
        // Compute residuals
        #pragma omp parallel for collapse(3) num_threads(omp_get_num_procs()) reduction(+:mean_res) schedule(static)
        for (int i = 0; i < N_S - 1; ++i) {
            for (int k = 0; k < N_v - 1; ++k) {
                for (int j = 0; j < N_t - 1; ++j) {
                    calc_eom(i, k, j, k1[i][k][j], V);
                    res[i][k][j] = k1[i][k][j];
                    mean_res += abs(res[i][k][j]);
                }
            }
        }
        mean_res /= (N_S - 1) * (N_v - 1) * (N_t - 1);

        // Check for convergence or divergence
        if (mean_res < error_tol) {
            #pragma omp critical
            cout << "Converged at iteration " << iter << ", Mean Residual: " << mean_res << endl;
            break;
        }
        if (mean_res > prev_mean_res * 5 || isnan(mean_res) || isinf(mean_res)) {
            #pragma omp critical
            cout << "Error: Solver diverging at iteration " << iter << ", Mean Residual: " << mean_res << endl;
            is_solved = false;
            break;
        }
        prev_mean_res = mean_res;

        // Perform Runge-Kutta stages
        V_temp = V;
        compute_rk_stage(V, V_temp, dtau / 2.0, k1);
        compute_rk_stage(V_temp, V_temp, dtau / 2.0, k2);
        compute_rk_stage(V_temp, V_temp, dtau, k3);
        compute_rk_stage(V_temp, V_temp, dtau, k4);

        // Update solution with early exercise
        #pragma omp parallel for collapse(3) num_threads(omp_get_num_procs()) schedule(static)
        for (int i = 0; i < N_S - 1; ++i) {
            for (int k = 0; k < N_v - 1; ++k) {
                for (int j = 0; j < N_t - 1; ++j) {
                    double update = dtau_sixth * (k1[i][k][j] + 2.0 * k2[i][k][j] + 2.0 * k3[i][k][j] + k4[i][k][j]);
                    double accel = res[i][k][j] - res_prev[i][k][j];
                    double apply_update = (use_arrested_newton && iter > 0 && update * accel < 0) ? update * 0.5 : update;
                    V[i][k][j] -= apply_update;
                    double S = i * dS;
                    double intrinsic = is_call ? max(S - K, 0.0) : max(K - S, 0.0);
                    V[i][k][j] = max(V[i][k][j], intrinsic);
                    if (V[i][k][j] < 0) V[i][k][j] = 0.0;
                    res_prev[i][k][j] = res[i][k][j];
                }
            }
        }
        apply_boundary_conditions(V);

        // Log iteration progress
        if (iter % 5000 == 0) {
            #pragma omp critical
            cout << "Iteration: " << iter << ", Mean Residual: " << mean_res << endl;
        }
    }
}

// Interpolates option price from grid
double Heston_FD::get_option_price(double S, double v, bool is_call) {
    if (!is_solved || solved_is_call != is_call) {
        if (is_eu) {
            solve(is_call);
        } else {
            solve_american();
        }
    }
    if (!is_solved) return 0.0; // Return 0 if solver failed
    int i = static_cast<int>(floor(S / dS));
    int k = static_cast<int>(floor(v / dv));
    if (i < 0 || i >= N_S - 1 || k < 0 || k >= N_v - 1) return 0.0;
    double S_i = i * dS;
    double v_k = k * dv;
    double S_ip1 = (i + 1) * dS;
    double v_kp1 = (k + 1) * dv;

    // Perform bilinear interpolation
    double V00 = V[i][k][0];
    double V10 = V[i + 1][k][0];
    double V01 = V[i][k + 1][0];
    double V11 = V[i + 1][k + 1][0];
    if (isnan(V00) || isnan(V10) || isnan(V01) || isnan(V11)) return 0.0;
    double fS = (S - S_i) / (S_ip1 - S_i);
    double fv = (v - v_k) / (v_kp1 - v_k);
    double price = (1 - fS) * (1 - fv) * V00 + fS * (1 - fv) * V10 + (1 - fS) * fv * V01 + fS * fv * V11;
    return max(price, 0.0);
}

// Validates computed price
double Heston_FD::check_price(double S, double v, bool is_call) {
    double price = get_option_price(S, v, is_call);
    return isnan(price) || price < 0 ? 0.0 : price;
}