#include "Monte_Carlo.h"

using namespace std;

Monte_Carlo::Monte_Carlo()
{
    //ctor
}

Monte_Carlo::~Monte_Carlo()
{
    //dtor
}

double Monte_Carlo::option_price(double S, double K, double T, double r, double sigma, bool is_call, int num_sims)
{
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0);
    double sum_payoffs = 0.0;

    #pragma omp parallel num_threads(6) reduction(+:sum_payoffs)
    {
        mt19937 local_gen(rd() + omp_get_thread_num());
        normal_distribution<double> local_dist(0.0, 1.0);
        #pragma omp for
        for (int i = 0; i < num_sims; ++i) {
            double Z = local_dist(local_gen);
            double ST = S * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
            double payoff = is_call ? max(ST - K, 0.0) : max(K - ST, 0.0);
            sum_payoffs += payoff;
        }
    }

    return exp(-r * T) * sum_payoffs / num_sims;
}

double Monte_Carlo::Heston_option_price(double S0, double K, double T, double r, double v0, double kappa, double theta, double xi, double rho, bool is_call, int num_sims, int num_steps)
{
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0);
    double dt = T / num_steps;
    double sqrt_dt = sqrt(dt);
    double sum_payoffs = 0.0;

    #pragma omp parallel num_threads(6) reduction(+:sum_payoffs)
    {
        mt19937 local_gen(rd() + omp_get_thread_num());
        normal_distribution<double> local_dist(0.0, 1.0);
        #pragma omp for
        for (int i = 0; i < num_sims; i++) {
            double S = S0;
            double v = v0;
            for (int t = 0; t < num_steps; t++){
                double z1 = dist(gen);
                double z2 = rho * z1 + sqrt(1 - rho * rho) * dist(gen);
                v = max(v + kappa * (theta - v) * dt + xi * sqrt(max(v, 0.0)) * sqrt_dt * z2, 0.0);
                S *= exp((r - 0.5 * v) * dt + sqrt(max(v, 0.0)) * sqrt_dt * z1);
            }
            sum_payoffs += is_call ? max(S - K, 0.0) : max(K - S, 0.0);
        }

    }

    return exp(-r * T) * sum_payoffs / num_sims;
}
