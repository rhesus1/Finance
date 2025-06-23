#include <cmath>
#include <algorithm>
#include <vector>

#ifndef BLACK_SCHOLES_FD_H
#define BLACK_SCHOLES_FD_H

class Black_Scholes_FD {
private:
    double S0, S_max, T, r, sigma, K, dS, dt, dtau;
    double sigma2, dS2, dtau_sixth;
    int N_S, N_t, max_iter;
    bool is_call, use_arrested_newton;
    std::vector<std::vector<double>> V, V_temp, k_temp, res, res_prev;

    double deriv1_S(const std::vector<std::vector<double>>& f, int i, int j);
    double deriv2_S(const std::vector<std::vector<double>>& f, int i, int j);
    double deriv1_t(const std::vector<std::vector<double>>& f, int i, int j);
    void calc_eom(int i, int j, double& res, const std::vector<std::vector<double>>& V_in);
    void apply_boundary_conditions(std::vector<std::vector<double>>& V_out);
    void compute_rk_stage(const std::vector<std::vector<double>>& V_in, std::vector<std::vector<double>>& V_out, double scale);

public:
    Black_Scholes_FD(double S0, double K, double T, double r, double sigma, int N_S, int N_t, int max_iter, bool is_call, bool use_arrested_newton = false);

    void solve();
    void solve_american();
    double get_option_price(double S0);
    int get_convergence_iterations() const { return last_iter; }
private:
    int last_iter = 0;
};

#endif