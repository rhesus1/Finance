#include "Monte_Carlo.h"

using namespace std;

Monte_Carlo::Monte_Carlo()
{
    //ctor
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
