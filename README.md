C++ Program for option pricing.
- Black-Scholes
- Black-Scholes PDE finite difference
- Black-Scholes Monte-Carlo
- Heston Fourier Inversion
- Heston Monte-Carlo
- Heston PDE finite difference


For the finite difference methods, we use a second-order finite difference for the derivatives, and a 4th order Runge-Kutta for numerical relaxation, with an arresting condition to stabilise the flow.

Compile : g++ src/*.cpp main.cpp -fopenmp -std=c++17 -I include -I /c/Boost/include/boost-1_88 -I /mingw64/include -L /mingw64/lib -lcurl -O3 -o main

Note: Ensure installation of Boost, libcurl and nlohmann-json
