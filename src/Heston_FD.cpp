#include "Heston_FD.h"
using namespace std;

Heston_FD::Heston_FD(double S0, double K, double T, double r, double v0, double kappa, double theta, double xi, double rho,
                       int N_S, int N_v, int N_t, int max_iter, bool is_call) :
                           S0(S0), K(K), T(T), r(r), v0(v0), kappa(kappa), theta(theta), xi(xi), rho(rho), N_S(N_S), N_v(N_v), N_t(N_t), max_iter(max_iter), is_call(is_call){
    S_max = 2 * S0;
    v_max = 4 * theta;
    dS = S_max / (N_S - 1);
    dv = v_max / (N_v - 1);
    dt = T / (N_t - 1);
    dtau = 1e-4;
    V.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    res.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    res_prev.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    V_temp.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    k1.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    k2.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    k3.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    k4.resize(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));

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
                    V[i][k][j] = is_call ? BS.call(S, K, time_to_maturity, r, vol)
                                         : BS.put(S, K, time_to_maturity, r, vol);
                    if (isnan(V[i][k][j])) {
                        V[i][k][j] = is_call ? max(S - K * exp(-r * time_to_maturity), 0.0)
                                             : max(K * exp(-r * time_to_maturity) - S, 0.0);
                    }
                }
            }
        }
    }

    for (int j = 0; j < N_t; ++j) {
        for (int i = 1; i < N_S - 1; ++i) {
            for (int k = 1; k < N_v - 1; ++k) {
                V[i][k][j] = 0.25 * (V[i][k][j] + V[i+1][k][j] + V[i-1][k][j] + V[i][k+1][j]);
            }
        }
    }

    for (int j = 0; j < N_t; ++j) {
        double t = j * dt;
        for (int k = 0; k < N_v; ++k) {
            V[0][k][j] = is_call ? 0.0 : K * exp(-r * (T - t));
            V[N_S - 1][k][j] = is_call ? S_max - K * exp(-r * (T - t)) : 0.0;
        }
        for (int i = 0; i < N_S; ++i) {
            V[i][0][j] = V[i][1][j];
            V[i][N_v - 1][j] = is_call ? i * dS : max(K * exp(-r * (T - j * dt)) - i * dS, 0.0);
        }
    }
}



double Heston_FD::deriv1_S(const vector<vector<vector<double>>>& f, int i, int k, int j) {
    if (i == 0) return (f[i + 1][k][j] - f[i][k][j]) / dS;
    return (f[i + 1][k][j] - f[i - 1][k][j]) / (2 * dS);
}

double Heston_FD::deriv2_S(const vector<vector<vector<double>>>& f, int i, int k, int j) {
    if (i == 0) return (f[i + 2][k][j] - 2 * f[i + 1][k][j] + f[i][k][j]) / (dS * dS);
    return (f[i + 1][k][j] - 2 * f[i][k][j] + f[i - 1][k][j]) / (dS * dS);
}

double Heston_FD::deriv1_v(const vector<vector<vector<double>>>& f, int i, int k, int j) {
    if (k == 0) return (f[i][k + 1][j] - f[i][k][j]) / dv;
    return (f[i][k + 1][j] - f[i][k - 1][j]) / (2 * dv);
}

double Heston_FD::deriv2_v(const vector<vector<vector<double>>>& f, int i, int k, int j) {
    if (k == 0) return (f[i][k + 2][j] - 2 * f[i][k + 1][j] + f[i][k][j]) / (dv * dv);
    return (f[i][k + 1][j] - 2 * f[i][k][j] + f[i][k - 1][j]) / (dv * dv);
}

double Heston_FD::deriv2_Sv(const vector<vector<vector<double>>>& f, int i, int k, int j) {
    if (i == 0 && k == 0) return (f[i + 1][k + 1][j] - f[i + 1][k][j] - f[i][k + 1][j] + f[i][k][j]) / (dS * dv);
    if (i == 0) return (f[i + 1][k + 1][j] - f[i + 1][k - 1][j] - f[i][k + 1][j] + f[i][k - 1][j]) / (2 * dS * dv);
    if (k == 0) return (f[i + 1][k + 1][j] - f[i + 1][k][j] - f[i - 1][k + 1][j] + f[i - 1][k][j]) / (2 * dS * dv);
    return (f[i + 1][k + 1][j] - f[i + 1][k - 1][j] - f[i - 1][k + 1][j] + f[i - 1][k - 1][j]) / (4 * dS * dv);
}

double Heston_FD::deriv1_t(const vector<vector<vector<double>>>& f, int i, int k, int j) {
    if (j == N_t - 1) return 0.0;
    if (j == 0) return (f[i][k][j + 1] - f[i][k][j]) / dt;
    return (f[i][k][j + 1] - f[i][k][j - 1]) / (2 * dt);
}

void Heston_FD::calc_eom(int i, int k, int j, double& res, const vector<vector<vector<double>>>& f) {
    double S = i * dS;
    double v = k * dv;
    res = deriv1_t(f, i, k, j) +
                   0.5 * v * S * S * deriv2_S(f, i, k, j) +
                   rho * xi * v * S * deriv2_Sv(f, i, k, j) +
                   0.5 * xi * xi * v * deriv2_v(f, i, k, j) +
                   r * S * deriv1_S(f, i, k, j) +
                   kappa * (theta - v) * deriv1_v(f, i, k, j) -
                   r * f[i][k][j];
}

void Heston_FD::solve(bool is_call) {
    vector<vector<vector<double>>> res(N_S, vector<vector<double>>(N_v, vector<double>(N_t, 0.0)));
    double mean_res = 0.0;
    int iter = 0;
    #pragma omp parallel num_threads(10) shared(V, res, res_prev, V_temp, k1, k2, k3, k4)  private(iter)
    {
        for (iter = 0; iter < max_iter; ++iter) {
            #pragma omp single
            mean_res = 0.0;
            #pragma omp for collapse(3) reduction(+:mean_res)
            for (int i = 0; i < N_S - 1; i++) {
                for (int k = 0; k < N_v - 1; k++) {
                    for (int j = 0; j < N_t - 1; j++) {
                        calc_eom(i, k, j, k1[i][k][j], V);
                        res[i][k][j] = k1[i][k][j];
                        mean_res += abs(res[i][k][j]);
                    }
                }
            }
            #pragma omp single
            mean_res /= ((N_S - 1) * (N_v - 1) * (N_t - 1));

            #pragma omp for collapse(3)
            for (int i = 0; i < N_S - 1; i++) {
                for (int k = 0; k < N_v - 1; k++) {
                    for (int j = 0; j < N_t - 1; j++) {
                        V_temp[i][k][j] = V[i][k][j] + (dtau / 2.0) * k1[i][k][j];
                    }
                }
            }
            #pragma omp for collapse(2)
            for (int k = 0; k < N_v; k++) {
                for (int j = 0; j < N_t; j++) {
                    V_temp[0][k][j] = is_call ? 0.0 : K * exp(-r * (T - j * dt));
                    V_temp[N_S - 1][k][j] = is_call ? S_max - K * exp(-r * (T - j * dt)) : 0.0;
                }
            }
            #pragma omp for collapse(2)
            for (int i = 0; i < N_S; i++) {
                for (int j = 0; j < N_t; j++) {
                    V_temp[i][0][j] = V_temp[i][1][j];
                    V_temp[i][N_v - 1][j] = is_call ? i * dS : max(K - i * dS, 0.0);
                }
            }
            #pragma omp for
            for (int i = 0; i < N_S; i++) {
                for (int k = 0; k < N_v; k++) {
                    V_temp[i][k][N_t - 1] = is_call ? max(i * dS - K, 0.0) : max(K - i * dS, 0.0);
                }
            }
            #pragma omp for collapse(3)
            for (int i = 0; i < N_S - 1; i++) {
                for (int k = 0; k < N_v - 1; k++) {
                    for (int j = 0; j < N_t - 1; j++) {
                        calc_eom(i, k, j, k2[i][k][j], V_temp);
                    }
                }
            }
            #pragma omp for collapse(3)
            for (int i = 0; i < N_S - 1; i++) {
                for (int k = 0; k < N_v - 1; k++) {
                    for (int j = 0; j < N_t - 1; j++) {
                        V_temp[i][k][j] = V[i][k][j] + (dtau / 2.0) * k2[i][k][j];
                    }
                }
            }
            #pragma omp for collapse(2)
            for (int k = 0; k < N_v; k++) {
                for (int j = 0; j < N_t; j++) {
                    V_temp[0][k][j] = is_call ? 0.0 : K * exp(-r * (T - j * dt));
                    V_temp[N_S - 1][k][j] = is_call ? S_max - K * exp(-r * (T - j * dt)) : 0.0;
                }
            }
            #pragma omp for collapse(2)
            for (int i = 0; i < N_S; i++) {
                for (int j = 0; j < N_t; j++) {
                    V_temp[i][0][j] = V_temp[i][1][j];
                    V_temp[i][N_v - 1][j] = is_call ? i * dS : max(K - i * dS, 0.0);
                }
            }
            #pragma omp for
            for (int i = 0; i < N_S; i++) {
                for (int k = 0; k < N_v; k++) {
                    V_temp[i][k][N_t - 1] = is_call ? max(i * dS - K, 0.0) : max(K - i * dS, 0.0);
                }
            }
            #pragma omp for collapse(3)
            for (int i = 0; i < N_S - 1; i++) {
                for (int k = 0; k < N_v - 1; k++) {
                    for (int j = 0; j < N_t - 1; j++) {
                        calc_eom(i, k, j, k3[i][k][j], V_temp);
                    }
                }
            }
            #pragma omp for collapse(3)
            for (int i = 0; i < N_S - 1; i++) {
                for (int k = 0; k < N_v - 1; k++) {
                    for (int j = 0; j < N_t - 1; j++) {
                        V_temp[i][k][j] = V[i][k][j] + dtau * k3[i][k][j];
                    }
                }
            }
            #pragma omp for collapse(2)
            for (int k = 0; k < N_v; k++) {
                for (int j = 0; j < N_t; j++) {
                    V_temp[0][k][j] = is_call ? 0.0 : K * exp(-r * (T - j * dt));
                    V_temp[N_S - 1][k][j] = is_call ? S_max - K * exp(-r * (T - j * dt)) : 0.0;
                }
            }
            #pragma omp for collapse(2)
            for (int i = 0; i < N_S; i++) {
                for (int j = 0; j < N_t; j++) {
                    V_temp[i][0][j] = V_temp[i][1][j];
                    V_temp[i][N_v - 1][j] = is_call ? i * dS : max(K - i * dS, 0.0);
                }
            }
            #pragma omp for
            for (int i = 0; i < N_S; i++) {
                for (int k = 0; k < N_v; k++) {
                    V_temp[i][k][N_t - 1] = is_call ? max(i * dS - K, 0.0) : max(K - i * dS, 0.0);
                }
            }
            #pragma omp for collapse(3)
            for (int i = 0; i < N_S - 1; i++) {
                for (int k = 0; k < N_v - 1; k++) {
                    for (int j = 0; j < N_t - 1; j++) {
                        calc_eom(i, k, j, k4[i][k][j], V_temp);
                    }
                }
            }
            #pragma omp for collapse(3)
            for (int i = 0; i < N_S - 1; i++) {
                for (int k = 0; k < N_v - 1; k++) {
                    for (int j = 0; j < N_t - 1; j++) {
                        double update = (dtau / 6.0) * (k1[i][k][j] + 2.0 * k2[i][k][j] + 2.0 * k3[i][k][j] + k4[i][k][j]);
                        double accel = res[i][k][j] - res_prev[i][k][j];
                        if (iter > 0 && update * accel < 0) {
                            res_prev[i][k][j] = res[i][k][j];
                            continue;
                        }
                        V[i][k][j] += update;
                        res_prev[i][k][j] = res[i][k][j];
                        if (V[i][k][j] < 0) V[i][k][j] = 0.0;
                    }
                }
            }
            #pragma omp single
            {
                if (iter % 10000 == 0) {
                    cout << "Iteration: " << iter << ", Mean Residual: " << mean_res << endl;
                }
            }
        }
    }
}

double Heston_FD::get_option_price(double S, double v, bool is_call) {
    solve(is_call);
    int i = static_cast<int>(S / dS);
    int k = static_cast<int>(v / dv);
    if (i < 0 || i >= N_S - 1 || k < 0 || k >= N_v - 1) return 0.0;
    double S_i = i * dS;
    double v_k = k * dv;
    double S_ip1 = (i + 1) * dS;
    double v_kp1 = (k + 1) * dv;

    double V00 = V[i][k][0];
    double V10 = V[i + 1][k][0];
    double V01 = V[i][k + 1][0];
    double V11 = V[i + 1][k + 1][0];
    double fS = (S - S_i) / dS;
    double fv = (v - v_k) / dv;

    return (1 - fS) * (1 - fv) * V00 + fS * (1 - fv) * V10 + (1 - fS) * fv * V01 + fS * fv * V11;
}
