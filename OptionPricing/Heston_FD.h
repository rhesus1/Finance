#ifndef HESTON_FD_H
#define HESTON_FD_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include "Black_Scholes.h"

using namespace std;


class Heston_FD
{
    public:
        Heston_FD(double S0, double K, double T, double r, double v0, double kappa, double theta, double xi, double rho,
                       int N_S, int N_v, int N_t, int max_iter, bool is_call);
        double get_option_price(double S, double v, bool is_call);
    protected:

    private:
        double check_price(double S, double v, bool is_call);
        double deriv1_S(const vector<vector<vector<double>>>& V_in, int i, int k, int j);
        double deriv2_S(const vector<vector<vector<double>>>& V_in, int i, int k, int j);
        double deriv1_v(const vector<vector<vector<double>>>& V_in, int i, int k, int j);
        double deriv2_v(const vector<vector<vector<double>>>& V_in, int i, int k, int j);
        double deriv2_Sv(const vector<vector<vector<double>>>& V_in, int i, int k, int j);
        double deriv1_t(const vector<vector<vector<double>>>& V_in, int i, int k, int j);
        void calc_eom(int i, int k, int j, double& res, const vector<vector<vector<double>>>& V_in);
        void solve(bool is_call);
        double S0, K, T, r, v0, kappa, theta, xi, rho, S_max, v_max, dS, dv, dt, dtau;
        int N_S, N_v, N_t, max_iter;
        bool is_call;
        vector<vector<vector<double>>> V, res, res_prev, V_temp, k1, k2, k3, k4;
};

#endif // HESTON_FD_H




