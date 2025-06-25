#include "Black_Scholes_FD_simd.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <vector>
#include <immintrin.h>

using namespace std;

// Constructor initializes finite difference grid with SIMD support
Black_Scholes_FD_simd::Black_Scholes_FD_simd(double S0, double K, double T, double r, double sigma, int N_S, int N_t, int max_iter, bool is_call, bool use_arrested_newton, vector<double>* V_ext, vector<double>* V_temp_ext, vector<double>* k_temp_ext, vector<double>* res_ext, vector<double>* res_prev_ext)
    : S0(S0), S_max(2 * S0), T(T), r(r), sigma(sigma), K(K), N_S(N_S), N_t(N_t), max_iter(max_iter), is_call(is_call), use_arrested_newton(use_arrested_newton) {
    // Set grid steps
    dS = S_max / (N_S - 1);
    dt = T / (N_t - 1);
    dtau = 1e-4;
    sigma2 = sigma * sigma;
    dS2 = dS * dS;
    dtau_sixth = dtau / 6.0;

    // Use external grids if provided
    if (V_ext && V_temp_ext && k_temp_ext && res_ext && res_prev_ext) {
        V = *V_ext;
        V_temp = *V_temp_ext;
        k_temp = *k_temp_ext;
        res = *res_ext;
        res_prev = *res_prev_ext;
    } else {
        // Allocate grids
        V.resize(N_S * N_t, 0.0);
        V_temp.resize(N_S * N_t, 0.0);
        k_temp.resize(N_S * N_t, 0.0);
        res.resize(N_S * N_t, 0.0);
        res_prev.resize(N_S * N_t, 0.0);
    }

    // Initialize grid for call or put
    if (is_call) {
        // Set terminal condition
        for (int i = 0; i < N_S; ++i) {
            double S = i * dS;
            V[i * N_t + (N_t - 1)] = max(S - K, 0.0);
        }
        // Set boundary conditions
        for (int j = 0; j < N_t; ++j) {
            V[0 * N_t + j] = 0.0;
            double t = j * dt;
            V[(N_S - 1) * N_t + j] = S_max - K * exp(-r * (T - t));
        }
        // Initialize interior points
        for (int j = 0; j < N_t - 1; ++j) {
            for (int i = 1; i < N_S - 1; ++i) {
                double S = i * dS;
                V[i * N_t + j] = max(S - K, 0.0);
            }
        }
    } else {
        // Set terminal condition
        for (int i = 0; i < N_S; ++i) {
            double S = i * dS;
            V[i * N_t + (N_t - 1)] = max(K - S, 0.0);
        }
        // Set boundary conditions
        for (int j = 0; j < N_t; ++j) {
            double t = j * dt;
            V[0 * N_t + j] = K * exp(-r * (T - t));
            V[(N_S - 1) * N_t + j] = 0.0;
        }
        // Initialize interior points
        for (int j = 0; j < N_t - 1; ++j) {
            for (int i = 1; i < N_S - 1; ++i) {
                double S = i * dS;
                V[i * N_t + j] = max(K - S, 0.0);
            }
        }
    }
}

// Computes first derivative with respect to stock price
double Black_Scholes_FD_simd::deriv1_S(const vector<double>& f, int i, int j) {
    if (i == 0) return (f[(i + 1) * N_t + j] - f[i * N_t + j]) / dS;
    if (i == N_S - 1) return (f[i * N_t + j] - f[(i - 1) * N_t + j]) / dS;
    return (f[(i + 1) * N_t + j] - f[(i - 1) * N_t + j]) / (2 * dS);
}

// Computes second derivative with respect to stock price
double Black_Scholes_FD_simd::deriv2_S(const vector<double>& f, int i, int j) {
    if (i == 0) return (f[(i + 2) * N_t + j] - 2.0 * f[(i + 1) * N_t + j] + f[i * N_t + j]) / dS2;
    if (i == N_S - 1) return (f[i * N_t + j] - 2.0 * f[(i - 1) * N_t + j] + f[(i - 2) * N_t + j]) / dS2;
    return (f[(i + 1) * N_t + j] - 2.0 * f[i * N_t + j] + f[(i - 1) * N_t + j]) / dS2;
}

// Computes first derivative with respect to time
double Black_Scholes_FD_simd::deriv1_t(const vector<double>& f, int i, int j) {
    if (j == 0) return (f[i * N_t + (j + 1)] - f[i * N_t + j]) / dt;
    if (j == N_t - 1) return (f[i * N_t + j] - f[i * N_t + (j - 1)]) / dt;
    return (f[i * N_t + (j + 1)] - f[i * N_t + (j - 1)]) / (2 * dt);
}

// Computes Black-Scholes PDE residual for single point
void Black_Scholes_FD_simd::calc_eom(int i, int j, double& res, const vector<double>& V_in) {
    double S = i * dS;
    double d1S = deriv1_S(V_in, i, j);
    double d2S = deriv2_S(V_in, i, j);
    double d1t = deriv1_t(V_in, i, j);
    res = d1t + 0.5 * sigma2 * S * S * d2S + r * S * d1S - r * V_in[i * N_t + j];
}

// Computes Black-Scholes PDE residual using SIMD
void Black_Scholes_FD_simd::calc_eom_simd(int i, int j_start, int j_end, vector<double>& res, const vector<double>& V_in) {
    if (N_t > 32) {
        // Initialize SIMD vectors
        __m256d sigma2_vec = _mm256_set1_pd(sigma2);
        __m256d r_vec = _mm256_set1_pd(r);
        __m256d S_vec = _mm256_set1_pd(i * dS);
        __m256d S2_vec = _mm256_set1_pd((i * dS) * (i * dS));
        __m256d dS_inv_vec = _mm256_set1_pd(1.0 / dS);
        __m256d dS2_inv_vec = _mm256_set1_pd(1.0 / dS2);
        __m256d dt_inv_vec = _mm256_set1_pd(1.0 / dt);
        __m256d two_vec = _mm256_set1_pd(2.0);
        __m256d half_vec = _mm256_set1_pd(0.5);

        // Process four time steps at once
        for (int j = j_start; j < j_end - 3; j += 4) {
            __m256d V_vec = _mm256_set_pd(V_in[i * N_t + (j + 3)], V_in[i * N_t + (j + 2)], V_in[i * N_t + (j + 1)], V_in[i * N_t + j]);
            __m256d d1t_vec;
            if (j == 0 || j >= N_t - 4) {
                // Fallback to scalar computation near boundaries
                for (int k = 0; k < 4; ++k) {
                    calc_eom(i, j + k, res[i * N_t + (j + k)], V_in);
                }
                continue;
            } else {
                // Compute time derivative using SIMD
                __m256d V_next = _mm256_set_pd(V_in[i * N_t + (j + 4)], V_in[i * N_t + (j + 3)], V_in[i * N_t + (j + 2)], V_in[i * N_t + (j + 1)]);
                __m256d V_prev = _mm256_set_pd(V_in[i * N_t + (j + 2)], V_in[i * N_t + (j + 1)], V_in[i * N_t + j], V_in[i * N_t + (j - 1)]);
                d1t_vec = _mm256_mul_pd(_mm256_sub_pd(V_next, V_prev), _mm256_mul_pd(dt_inv_vec, half_vec));
            }
            // Compute spatial derivatives
            double d1S = deriv1_S(V_in, i, j);
            double d2S = deriv2_S(V_in, i, j);
            __m256d d1S_vec = _mm256_set1_pd(d1S);
            __m256d d2S_vec = _mm256_set1_pd(d2S);
            // Compute PDE terms
            __m256d term1 = _mm256_mul_pd(half_vec, _mm256_mul_pd(sigma2_vec, _mm256_mul_pd(S2_vec, d2S_vec)));
            __m256d term2 = _mm256_mul_pd(r_vec, _mm256_mul_pd(S_vec, d1S_vec));
            __m256d term3 = _mm256_mul_pd(r_vec, V_vec);
            __m256d res_vec = _mm256_add_pd(d1t_vec, _mm256_sub_pd(_mm256_add_pd(term1, term2), term3));
            // Store results
            alignas(32) double res_array[4];
            _mm256_store_pd(res_array, res_vec);
            res[i * N_t + j] = res_array[0];
            res[i * N_t + (j + 1)] = res_array[1];
            res[i * N_t + (j + 2)] = res_array[2];
            res[i * N_t + (j + 3)] = res_array[3];
        }
        // Handle remaining time steps
        for (int j = j_end - (j_end % 4); j < j_end; ++j) {
            calc_eom(i, j, res[i * N_t + j], V_in);
        }
    } else {
        // Fallback to scalar computation for small grids
        for (int j = j_start; j < j_end; ++j) {
            calc_eom(i, j, res[i * N_t + j], V_in);
        }
    }
}

// Applies boundary conditions
void Black_Scholes_FD_simd::apply_boundary_conditions(vector<double>& V_out) {
    #pragma omp parallel for num_threads(8)
    for (int j = 0; j < N_t; ++j) {
        V_out[0 * N_t + j] = V[0 * N_t + j];
        V_out[(N_S - 1) * N_t + j] = V[(N_S - 1) * N_t + j];
    }
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < N_S; ++i) {
        V_out[i * N_t + (N_t - 1)] = V[i * N_t + (N_t - 1)];
    }
}

// Computes one Runge-Kutta stage
void Black_Scholes_FD_simd::compute_rk_stage(const vector<double>& V_in, vector<double>& V_out, double scale) {
    #pragma omp parallel for num_threads(8) schedule(static)
    for (int i = 0; i < N_S - 1; ++i) {
        calc_eom_simd(i, 0, N_t - 1, k_temp, V_in);
        for (int j = 0; j < N_t - 1; ++j) {
            V_out[i * N_t + j] = V[i * N_t + j] + scale * k_temp[i * N_t + j];
        }
    }
    apply_boundary_conditions(V_out);
}

// Solves Black-Scholes PDE for European options
void Black_Scholes_FD_simd::solve() {
    const double error_tol = 1e-6;
    for (int iter = 0; iter < max_iter; ++iter) {
        double mean_res = 0.0;
        // Compute residuals
        #pragma omp parallel for num_threads(8) reduction(+:mean_res) schedule(static)
        for (int i = 0; i < N_S - 1; ++i) {
            calc_eom_simd(i, 0, N_t - 1, res, V);
            for (int j = 0; j < N_t - 1; ++j) {
                k_temp[i * N_t + j] = res[i * N_t + j];
                mean_res += abs(res[i * N_t + j]);
            }
        }
        mean_res /= (N_S - 1) * (N_t - 1);
        if (mean_res < error_tol) break;

        // Perform Runge-Kutta stages
        compute_rk_stage(V, V_temp, dtau / 2.0);
        compute_rk_stage(V_temp, V_temp, dtau / 2.0);
        compute_rk_stage(V_temp, V_temp, dtau);
        compute_rk_stage(V_temp, V_temp, dtau);

        // Update solution
        #pragma omp parallel for collapse(2) num_threads(8) schedule(static)
        for (int i = 0; i < N_S - 1; ++i) {
            for (int j = 0; j < N_t - 1; ++j) {
                double update = dtau_sixth * (k_temp[i * N_t + j] + 2.0 * k_temp[i * N_t + j] + 2.0 * k_temp[i * N_t + j] + k_temp[i * N_t + j]);
                double accel = res[i * N_t + j] - res_prev[i * N_t + j];
                double apply_update = (use_arrested_newton && iter > 0 && update * accel < 0) ? 0.0 : update;
                V[i * N_t + j] += apply_update;
                res_prev[i * N_t + j] = res[i * N_t + j];
            }
        }
    }
}

// Solves Black-Scholes PDE for American options
void Black_Scholes_FD_simd::solve_american() {
    const double error_tol = 1e-6;
    for (int iter = 0; iter < max_iter; ++iter) {
        double mean_res = 0.0;
        // Compute residuals
        #pragma omp parallel for num_threads(8) reduction(+:mean_res) schedule(static)
        for (int i = 0; i < N_S - 1; ++i) {
            calc_eom_simd(i, 0, N_t - 1, res, V);
            for (int j = 0; j < N_t - 1; ++j) {
                k_temp[i * N_t + j] = res[i * N_t + j];
                mean_res += abs(res[i * N_t + j]);
            }
        }
        mean_res /= (N_S - 1) * (N_t - 1);
        if (mean_res < error_tol) break;

        // Perform Runge-Kutta stages
        compute_rk_stage(V, V_temp, dtau / 2.0);
        compute_rk_stage(V_temp, V_temp, dtau / 2.0);
        compute_rk_stage(V_temp, V_temp, dtau);
        compute_rk_stage(V_temp, V_temp, dtau);

        // Update solution and check early exercise
        #pragma omp parallel for collapse(2) num_threads(8) schedule(static)
        for (int i = 0; i < N_S - 1; ++i) {
            for (int j = 0; j < N_t - 1; ++j) {
                double update = dtau_sixth * (k_temp[i * N_t + j] + 2.0 * k_temp[i * N_t + j] + 2.0 * k_temp[i * N_t + j] + k_temp[i * N_t + j]);
                double accel = res[i * N_t + j] - res_prev[i * N_t + j];
                double apply_update = (use_arrested_newton && iter > 0 && update * accel < 0) ? 0.0 : update;
                V[i * N_t + j] += apply_update;
                double S = i * dS;
                double intrinsic = is_call ? max(S - K, 0.0) : max(K - S, 0.0);
                V[i * N_t + j] = max(V[i * N_t + j], intrinsic);
                res_prev[i * N_t + j] = res[i * N_t + j];
            }
        }
    }
}

// Interpolates option price from grid
double Black_Scholes_FD_simd::get_option_price(double S0) {
    int i = static_cast<int>(floor(S0 / dS));
    if (i < 0 || i >= N_S - 1) return 0.0;
    double S_i = i * dS;
    double S_ip1 = (i + 1) * dS;
    return V[i * N_t + 0] + ((V[(i + 1) * N_t + 0] - V[i * N_t + 0]) / (S_ip1 - S_i)) * (S0 - S_i);
}