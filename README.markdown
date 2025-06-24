# Finance: Quantitative Finance Models in C++

This repository contains high-performance C++ implementations of quantitative finance models, including option pricing (Black-Scholes, Heston) and LSTM neural networks for stock price prediction. Optimized with OpenMP and SIMD, these projects demonstrate expertise in computational finance, numerical methods, and machine learning applied to financial markets.

## Project Overview

The repository implements advanced quantitative finance models for option pricing and stock price prediction, leveraging modern C++ (C++17), parallelization (OpenMP), and vectorization (SIMD). Key features include second-order finite difference methods, 4th-order Runge-Kutta solvers, Monte Carlo simulations, Fourier inversion for Heston models, and LSTM-based stock price forecasting.

## Key Projects

- **LSTM Stock Prediction**: A custom LSTM neural network for predicting stock prices (e.g., Amazon), with data fetched via libcurl and parsed using nlohmann/json. [View visualization](https://morganjrees.co.uk/lstm).
- **Option Pricing**:
  - **Black-Scholes**: Analytical solution, finite difference (FD) with SIMD optimization, and Monte Carlo (MC) methods.
  - **Heston**: Fourier inversion, finite difference, and Monte Carlo implementations for stochastic volatility modeling.
  - [View visualizations](https://morganjrees.co.uk/option-pricing).
- **Portfolio Optimization**: Tools for mean-variance optimization and financial analysis, including custom linear regression implementations.

## Tech Stack

- **Languages**: C++17
- **Libraries**: Boost (math), nlohmann/json, libcurl
- **Optimizations**: OpenMP (parallelization), SIMD (AVX2 for finite difference solvers)
- **Tools**: Git, g++, MinGW (for Windows compatibility)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rhesus1/Finance
   ```
2. Install dependencies (Boost, libcurl, nlohmann/json):
   - On Ubuntu:
     ```bash
     sudo apt-get install libboost-all-dev libcurl4-openssl-dev
     ```
   - On Windows, ensure MinGW is installed and Boost/libcurl paths are configured.
3. Compile the code:
   ```bash
   g++ src/*.cpp main.cpp -fopenmp -std=c++17 -I include -I /c/Boost/include/boost-1_88 -I /mingw64/include -L /mingw64/lib -lcurl -O3 -march=native -mavx2 -ffast-math -o main
   ```

**Note**: Ensure Boost, libcurl, and nlohmann/json are installed. For Windows, adjust include/lib paths as needed.

## Usage

### Black-Scholes Example
Calculate a call option price using the analytical Black-Scholes model:

```cpp
#include "OptionPricing/Black_Scholes.h"
double S = 100.0, K = 100.0, T = 1.0, r = 0.05, sigma = 0.2;
double price = BlackScholesCall(S, K, T, r, sigma);
std::cout << "Call Price: " << price << std::endl;
```

### LSTM Stock Prediction Example
Run the LSTM model to predict stock prices (requires data in `data/` folder):

```cpp
#include "LSTM/LSTM.h"
LSTM model;
model.load_data("data/AMZN_stock.csv");
model.train();
double prediction = model.predict();
std::cout << "Predicted Stock Price: " << prediction << std::endl;
```

## Results

- **Heston Fourier Inversion**: Achieves ~0.1% pricing error compared to market data for European options.
- **Black-Scholes FD (SIMD)**: 2x speedup over scalar implementation due to AVX2 vectorization.
- **LSTM Stock Prediction**: ~5% RMSE on Amazon stock price predictions over a 1-year test period.

## Repository Structure

```
Finance/
├── LSTM/                 # LSTM stock price prediction
├── OptionPricing/        # Black-Scholes and Heston models
├── Portfolio/            # Portfolio optimization tools
├── Utils/                # Shared utilities (e.g., Monte Carlo, linear regression)
├── data/                 # Sample datasets (e.g., AMZN_stock.csv)
├── docs/                 # Technical reports (e.g., LSTM_Report.pdf)
├── tests/                # Unit tests for model validation
├── include/              # Header files
├── src/                  # Source files
├── main.cpp              # Main executable
├── README.md             # This file
```

## Optimizations

- **SIMD**: `Black_Scholes_FD_simd.h` and `Heston_FD.h` use AVX2 intrinsics for accelerated finite difference solvers, achieving significant performance gains.
- **OpenMP**: Parallelized Monte Carlo simulations and finite difference grid computations.
- **Numerical Methods**: Second-order finite difference for derivatives, 4th-order Runge-Kutta for numerical relaxation, with an arresting condition to stabilize convergence.

## Links

- [Portfolio Website](https://morganjrees.co.uk)
- [LinkedIn](https://www.linkedin.com/in/morganjrees)
- [CV](https://morganjrees.co.uk/cv)