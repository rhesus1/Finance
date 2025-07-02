# MSTR vs BTC Price Analysis

## Overview
This project, implemented in a Python script, analyses the price relationship between MicroStrategy (MSTR) and Bitcoin (BTC) using historical data from an Excel file (`MSTR VS BITCOIN.xlsx`). The script cleans the data, handles outliers, and performs various analyses including price comparisons, correlations, volatility, and regression modelling. Visualisations are generated using Plotly and Matplotlib to support the analysis.

## Features
- **Data Loading**:
  - Reads MSTR and BTC price data from Excel sheets 'MSTR' and 'BITCOIN'.
  - Renames columns for consistency (`date`/`Date` to `timestamp`, `lst px`/`Close` to `price`).
- **Data Cleaning**:
  - Converts price strings (e.g., '10k', '5M') to numerical values using a custom `convert_price` function.
  - Standardises timestamps to UTC and filters MSTR data to NASDAQ trading hours (9:30 AM - 4:00 PM ET, weekdays), excluding US federal holidays.
  - Removes outliers using a z-score method (>3) over a 60-minute rolling window, with linear interpolation for missing or invalid values.
  - Upsamples BTC data to 1-minute intervals using linear, forward fill, and time-based interpolation methods.
- **Data Storage**:
  - Stores cleaned data in an SQLite database (`fourier_crypto_stock_data.db`) with tables 'mstr', 'btc', and 'cleaned_data'.
- **Analysis**:
  - Computes price and return correlations using Pearson correlation.
  - Performs linear regression to predict MSTR prices based on BTC prices, both raw and normalised.
  - Calculates rolling correlations (60-minute and 1-day windows) and volatilities (60-minute window, annualised).
  - Determines MSTR's beta relative to BTC and analyses the spread between actual and predicted MSTR prices.
- **Visualisation**:
  - Generates dual-axis plots for raw and normalised price comparisons.
  - Plots interpolation methods for BTC data, volatility, rolling correlations, spread, and MSTR volume.

## Prerequisites
- Python 3.12.7 (or compatible version)
- Required packages:
  ```bash
  pip install pandas sqlalchemy numpy scikit-learn scipy plotly matplotlib
  ```
- Input file: `MSTR VS BITCOIN.xlsx` in the project directory.

## Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Place `MSTR VS BITCOIN.xlsx` in the project directory.
3. Run the script:
   ```bash
   python script.py
   ```
4. View generated plots and output metrics in the console and saved figures.

## Analysis Summary
- **Price Correlation**: 0.883 (raw and normalised), indicating strong co-movement due to MSTR's BTC holdings.
- **Return Correlation**: 0.406, showing moderate short-term co-movement with intraday noise.
- **Regression**: MSTR_price = 0.004 * BTC_price + -66.370 (raw); MSTR_price = 0.883 * BTC_price + -0.000 (normalised).
- **Volatility**: MSTR average 1.379, BTC average 229.172; Volatility correlation 0.643, suggesting a moderately strong link between risk profiles.
- **Beta**: MSTR Beta = 0.847, indicating a 1% BTC price change leads to a 0.847% MSTR price change.
- **Spread**: Near zero on average, with significant deviations suggesting market overreactions.

## Observations
- MSTR prices peak higher relative to BTC, but BTC shows larger fluctuations due to its scale.
- Normalised prices follow similar trends, confirming MSTR's heavy reliance on BTC.
- Rolling correlations vary, indicating other factors influence MSTR price movements.
- MSTR volatility is lower than BTC during peaks but shows higher overall risk.
- Volume data offers limited insight into spread fluctuations.

## Licence
This project is licensed under the MIT Licence. See the [LICENSE](LICENSE) file for details.