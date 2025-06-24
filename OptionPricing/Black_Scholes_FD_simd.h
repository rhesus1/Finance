#include <cmath>
#include <algorithm>
#include <omp.h>
#include <vector>
#include <immintrin.h>

#ifndef BLACK_SCHOLES_FD_SIMD_H
#define BLACK_SCHOLES_FD_SIMD_H

class Black_Scholes_FD_simd {
private:
    double S0, S_max, T, r, sigma, K, dS, dt, dtau;
    double sigma2, dS2, dtau_sixth;
    int N_S, N_t, max_iter;
    bool is_call, use_arrested_newton;
    std::vector<double> V, V_temp, k_temp, res, res_prev; // 1D arrays

    double deriv1_S(const std::vector<double>& f, int i, int j);
    double deriv2_S(const std::vector<double>& f, int i, int j);
    double deriv1_t(const std::vector<double>& f, int i, int j);
    void calc_eom(int i, int j, double& res, const std::vector<double>& V_in);
    void calc_eom_simd(int i, int j_start, int j_end, std::vector<double>& res, const std::vector<double>& V_in);
    void apply_boundary_conditions(std::vector<double>& V_out);
    void compute_rk_stage(const std::vector<double>& V_in, std::vector<double>& V_out, double scale);

public:
    Black_Scholes_FD_simd(double S0, double K, double T, double r, double sigma, int N_S, int N_t, int max_iter, bool is_call, bool use_arrested_newton = false, std::vector<double>* V = nullptr, std::vector<double>* V_temp = nullptr, std::vector<double>* k_temp = nullptr, std::vector<double>* res = nullptr, std::vector<double>* res_prev = nullptr);

    void solve();
    void solve_american();
    double get_option_price(double S0);
};

#endif