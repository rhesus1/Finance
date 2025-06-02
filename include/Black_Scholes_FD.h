#ifndef BLACK_SCHOLES_FD_H
#define BLACK_SCHOLES_FD_H

#include <vector>

class Black_Scholes_FD {
private:
    double S0, S_max, T, r, sigma, K, dS, dt, dtau;
    int N_S, N_t, max_iter;
    bool is_call;
    std::vector<std::vector<double>> V; // Option price grid
    std::vector<std::vector<double>> res_prev; // Previous residuals for arrested Newton flow
    std::vector<std::vector<double>> V_temp; // Temporary grid for RK4 intermediate states
    std::vector<std::vector<double>> k1, k2, k3, k4; // RK4 slopes
    double deriv1_S(const std::vector<std::vector<double>>& f, int i, int j, bool forward);
    double deriv2_S(const std::vector<std::vector<double>>& f, int i, int j, bool forward);
    double deriv1_t(const std::vector<std::vector<double>>& f, int i, int j, bool forward);
    void calc_eom(int i, int j, double& res, bool forward_t, bool forward_S, const std::vector<std::vector<double>>& V_in);

public:
    Black_Scholes_FD(double S0, double K, double T, double r, double sigma, int N_S, int N_t, int max_iter, bool is_call, bool use_arrested_newton = false);
        void solve();
    double get_option_price(double S0);
};

#endif