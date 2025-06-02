#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <random>
#include <cmath>
#include <vector>
#include <omp.h>

using namespace std;

class Monte_Carlo
{
    public:
        Monte_Carlo();
        virtual ~Monte_Carlo();
        double option_price(double S, double K, double T, double r, double sigma, bool is_call, int num_simulations);
        double Heston_option_price(double S0, double K, double T, double r, double v0, double kappa, double theta, double xi, double rho, bool is_call, int num_sims, int num_steps);
    protected:

    private:
};

#endif // MONTE_CARLO_H
