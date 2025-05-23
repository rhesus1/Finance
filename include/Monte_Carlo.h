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
        double option_price(double S, double K, double T, double r, double sigma, bool is_call, int num_simulations);
    protected:

    private:
};

#endif // MONTE_CARLO_H
