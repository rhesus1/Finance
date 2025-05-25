#ifndef BLACK_SCHOLES_FD_H
#define BLACK_SCHOLES_FD_H

#include <vector>

using namespace std;

class Black_Scholes_FD
{
    public:
        Black_Scholes_FD(double S0, double K, double T, double r, double sigma, int N_S, int N_t, int max_iter, bool is_call);
        void solve();
        double get_option_price(double S0);
    protected:

    private:
        int N_S, N_t;
        int max_iter;
        double S0, S_max, T, r, sigma, K;
        double dS, dt, dtau;
        bool is_call;
        vector<vector<double>> V;
        double deriv1_S(const vector<vector<double>>& f, int i, int j);
        double deriv2_S(const vector<vector<double>>& f, int i, int j);
        double deriv1_t(const vector<vector<double>>& f, int i, int j, bool forward);
        void calc_eom(int i, int j, double& res, bool foward_t);
};

#endif // BLACK_SCHOLES_FD_H
