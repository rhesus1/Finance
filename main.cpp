#include <cmath>
#include <boost/math/distributions/normal.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <curl/curl.h>
#include <ctime>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <random>
#include "Black_Scholes.h"
#include "Financial_Analysis.h"
#include "Monte_Carlo.h"
#include "Linear_Regression.h"
#include "Black_Scholes_FD.h"
#include "Heston_FD.h"
#include "Heston_Fourier.h"


using namespace std;
using json = nlohmann::json;

string API_KEY = "1RJ8JIJMMFBG3IPH";
string api_key_poly = "nbOtOYuPxQXHpkwnFrsTQfk6OFaw1SBO";

struct OptionData {
    double market_price,S,K,T;
    bool is_call;
};

string unixMsToDate(long long timestamp_ms, bool bst_adjust = false) {
    time_t seconds = static_cast<time_t>(timestamp_ms / 1000); // Convert ms to seconds
    if (bst_adjust) {
        seconds += 3600; // Add 1 hour for BST (UTC+1)
    }
    struct tm timeinfo;
    char buffer[80];
#ifdef _WIN32
    gmtime_s(&timeinfo, &seconds);
#else
    gmtime_r(&seconds, &timeinfo);
#endif

    // Format as YYYY-MM-DD
    strftime(buffer, sizeof(buffer), "%Y-%m-%d", &timeinfo);
    return std::string(buffer);
}

double rng(int rmax){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(10, rmax);
    return distrib(gen);
}

void evaluate_trades(const vector<OptionData>& options, double r, double sigma) {
    Black_Scholes BS;
    for (const auto& option : options) {
        double theoretical_price = option.is_call
            ? BS.call(option.S, option.K, option.T, r, sigma)
            : BS.put(option.S, option.K, option.T, r, sigma);

        if (option.market_price < theoretical_price * 0.95) {
            cout << "Buy " << (option.is_call ? "Call" : "Put")
                << ": Black-Scholes: " << theoretical_price << endl;
        } else {
            cout << "Don't Buy " << (option.is_call ? "Call" : "Put")
                << ": Black-Scholes: " << theoretical_price << endl;
        }
    }
}

void evaluate_trades_FD(const vector<OptionData>& options, double r, double sigma, int N_S, int N_t, int max_iter) {
    for (const auto& option : options) {
        Black_Scholes_FD FD(option.S, option.K, option.T, r, sigma, N_S, N_t, max_iter,option.is_call);
        FD.solve();
        double fdm_price = FD.get_option_price(option.S);
        if (option.market_price < fdm_price * 0.95) {
            cout << "Buy " << (option.is_call ? "Call" : "Put")
                << ": Black-Scholes Finite-Difference: " << fdm_price << endl;
        } else {
            cout << "Don't Buy " << (option.is_call ? "Call" : "Put")
                << ": Black-Scholes Finite-Difference: " << fdm_price << endl;
        }
    }
}

void evaluate_trades_HestonFD(const vector<OptionData>& options, double r, double v0, double sigma, double kappa, double xi, double rho) {
    for (const auto& option : options) {
        Heston_FD HFD(option.S, option.K, option.T, r, v0, kappa, sigma, xi, rho, 101, 51, 51, 200000, option.is_call);

        double hfd_price = HFD.get_option_price(option.S, v0, option.is_call);

        if (option.market_price < hfd_price * 0.95) {
            cout << "Buy " << (option.is_call ? "Call" : "Put")
                << ": Heston Finite-Difference: " << hfd_price << endl;
        } else {
            cout << "Don't Buy " << (option.is_call ? "Call" : "Put")
                << ": Heston Finite-Difference: " << hfd_price << endl;
        }
    }
}

void evaluate_trades_mc(const vector<OptionData>& options, double r, double sigma, int num_sims) {
    Monte_Carlo MC;
    for (const auto& option : options) {
        double theoretical_price = option.is_call
            ? MC.option_price(option.S, option.K, option.T, r, sigma, true, num_sims)
            : MC.option_price(option.S, option.K, option.T, r, sigma, false, num_sims);

        if (option.market_price < theoretical_price * 0.95) {
            cout << "Buy " << (option.is_call ? "Call" : "Put")
                << ": Black-Scholes Monte Carlo: " << theoretical_price << endl;
        } else {
            cout << "Don't Buy " << (option.is_call ? "Call" : "Put")
                << ": Black-Scholes Monte Carlo: " << theoretical_price << endl;
        }
    }
}

void evaluate_trades_mch(const vector<OptionData>& options, double r, double v0, double kappa, double theta, double xi, double rho, int num_sims, int num_steps) {
    Monte_Carlo MC;
    for (const auto& option : options) {
        double theoretical_price = option.is_call
            ? MC.Heston_option_price(option.S, option.K, option.T, r, v0, kappa, theta, xi, rho, true, num_sims, num_steps)
            : MC.Heston_option_price(option.S, option.K, option.T, r, v0, kappa, theta, xi, rho, false, num_sims, num_steps);

        if (option.market_price < theoretical_price * 0.95) {
            cout << "Buy " << (option.is_call ? "Call" : "Put")
                << ": Heston Monte Carlo: " << theoretical_price << endl;

        } else {
            cout << "Don't Buy " << (option.is_call ? "Call" : "Put")
                << ": Heston Monte Carlo: " << theoretical_price << endl;
        }
    }
}

void evaluate_trades_fourier(const vector<OptionData>& options, double r, double v0, double kappa, double theta, double xi, double rho) {
    for (const auto& option : options) {
        Heston_Fourier fourier(option.S, option.K, option.T, r, v0, kappa, theta, xi, rho);
        double theoretical_price = option.is_call
            ? fourier.price_call()
            : fourier.price_call();

        if (option.market_price < theoretical_price * 0.95) {
                            cout << "Buy " << (option.is_call ? "Call" : "Put")
                      << ": Heston Fourier: " << theoretical_price << endl;
        } else {
            cout << "Don't Buy " << (option.is_call ? "Call" : "Put")
                      << ": Heston Fourier: " << theoretical_price << endl;
        }
    }
}


size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    s->append((char*)contents, size * nmemb);
    return size * nmemb;
}

vector<OptionData> fetch_market_data(const string& symbol, vector<double>& close, vector<double>& dayInd, bool read_file) {
    CURL* curl = curl_easy_init();
    string response;
    vector<OptionData> options;
    string filename = symbol + "_market_data.json";

    if (read_file){
        ifstream in_file(filename);
        if (in_file.is_open()) {
            cout << "Reading " << filename << endl;
            json j;
            in_file >> j;
            close = j["close"].get<vector<double>>();
            dayInd = j["dayInd"].get<vector<double>>();
            in_file.close();

            return options;
        }
    }
    else{
        if (curl) {
            string ticker = symbol;
            string url = "https://api.polygon.io/v2/aggs/ticker/" + ticker +
                            "/range/1/day/2025-01-01/2025-06-02?adjusted=true&sort=asc&limit=50000&apiKey=" + api_key_poly;

            cout << "Fetching Market data from " << url << endl;
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            CURLcode res = curl_easy_perform(curl);
            curl_easy_cleanup(curl);


            if (res == CURLE_OK) {
                json j = json::parse(response);
                if (j.contains("results")) {
                    int ind = 0;
                    for (const auto& result : j["results"]) {
                        long long timestamp_ms = result["t"].get<long long>();
                        //std::string date_utc = unixMsToDate(timestamp_ms, false);
                        //std::string date_bst = unixMsToDate(timestamp_ms, true);
                        dayInd.push_back(ind);
                        close.push_back(result["c"].get<double>());
                        /*std::cout << "Date (UTC): " << date_utc << "\n"
                                  << "Date (BST): " << date_bst << "\n"
                                  << "Open Price: $" << result["o"].get<double>() << "\n"
                                  << "High Price: $" << result["h"].get<double>() << "\n"
                                  << "Low Price: $" << result["l"].get<double>() << "\n"
                                  << "Close Price: $" << result["c"].get<double>() << " (Primary Option Price)\n"
                                  << "Volume-Weighted Avg Price: $" << result["vw"].get<double>() << "\n"
                                  << "Volume: " << result["v"].get<long long>() << "\n"
                                  << "Transactions: " << result["n"].get<long long>() << "\n\n";*/

                        ind++;
                    }
                    json j_data;
                    j_data["close"] = close;
                    j_data["dayInd"] = dayInd;
                    std::ofstream out_file(filename);
                    if (out_file.is_open()) {
                        out_file << j_data.dump(4);
                        out_file.close();
                    }
                }
            }
        }
    }
    return options;
}

vector<OptionData> fetch_option_data(const string& symbol, vector<double>& close, vector<double>& dayInd, int strike, bool read_file) {
    CURL* curl = curl_easy_init();
    string response;
    vector<OptionData> options;
    string filename = symbol + "_option_data.json";

    if (read_file){
        ifstream in_file(filename);
        if (in_file.is_open()) {
            cout << "Reading " << filename << endl;
            json j;
            in_file >> j;
            close = j["close"].get<vector<double>>();
            dayInd = j["dayInd"].get<vector<double>>();
            in_file.close();
            return options;
        }
    }
    else{
        if (curl) {

            string ticker = "O:"+symbol+"251219C00"+to_string(strike)+"000";
            string url = "https://api.polygon.io/v2/aggs/ticker/" + ticker +
                            "/range/1/day/2025-01-01/2025-06-02?adjusted=true&sort=asc&limit=50000&apiKey=" + api_key_poly;

            cout << "Fetching Option data from " << url << endl;
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            CURLcode res = curl_easy_perform(curl);
            curl_easy_cleanup(curl);


            if (res == CURLE_OK) {
                json j = json::parse(response);
                if (j.contains("results")) {
                    int ind = 0;
                    for (const auto& result : j["results"]) {
                        long long timestamp_ms = result["t"].get<long long>();
                        //std::string date_utc = unixMsToDate(timestamp_ms, false);
                        //std::string date_bst = unixMsToDate(timestamp_ms, true);
                        dayInd.push_back(ind);
                        close.push_back(result["c"].get<double>());
                        /*std::cout << "Date (UTC): " << date_utc << "\n"
                                  << "Date (BST): " << date_bst << "\n"
                                  << "Open Price: $" << result["o"].get<double>() << "\n"
                                  << "High Price: $" << result["h"].get<double>() << "\n"
                                  << "Low Price: $" << result["l"].get<double>() << "\n"
                                  << "Close Price: $" << result["c"].get<double>() << " (Primary Option Price)\n"
                                  << "Volume-Weighted Avg Price: $" << result["vw"].get<double>() << "\n"
                                  << "Volume: " << result["v"].get<long long>() << "\n"
                                  << "Transactions: " << result["n"].get<long long>() << "\n\n";*/

                        ind++;
                    }
                    json j_data;
                    j_data["close"] = close;
                    j_data["dayInd"] = dayInd;
                    std::ofstream out_file(filename);
                    if (out_file.is_open()) {
                        out_file << j_data.dump(4);
                        out_file.close();
                    }
                    //double S = close[0];
                    //options.push_back({100.42, S, 105, 0.5, true});
                }
            }
        }
    }
    return options;
}

double Estimate_Stock(vector<double>& x, vector<double>& y){
    Linear_Regression model(0.01,10000);
    model.train(x,y);
    double nd = -1;
    double S_t = model.predict(nd);
    return S_t;
}



double computeMean(const vector<double>& data) {
    if (data.empty()) return 0.0;
    return accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double computeStdDev(const vector<double>& data, double mean) {
    if (data.size() < 2) return 0.0;
    double sum_squared_diff = 0.0;
    for (double x : data) {
        sum_squared_diff += (x - mean) * (x - mean);
    }
    return sqrt(sum_squared_diff / (data.size() - 1));
}

double computeZScore(double value, double mean, double stddev) {
    if (stddev == 0.0) return 0.0;
    return (value - mean) / stddev;
}

vector<double> smoothData(const vector<double>& data, int window = 5) {
    vector<double> smoothed;
    if (data.size() < static_cast<size_t>(window)) return data;
    for (size_t i = 0; i <= data.size() - window; ++i) {
        double sum = 0.0;
        for (int j = 0; j < window; ++j) {
            sum += data[i + j];
        }
        smoothed.push_back(sum / window);
    }
    return smoothed;
}


void generate_heston_surface_data(double S0, double r, double v0, double kappa, double theta, double sigma, double rho, const string& filename) {
    vector<double> strikes;
    vector<double> maturities;
    vector<vector<double>> prices;

    for (int i = 0; i < 20; ++i) {
        strikes.push_back(S0 * (0.8 + i * 0.02));
    }
    for (int i = 0; i < 20; ++i) {
        maturities.push_back(0.1 + i * 0.045);
    }

    for (double T : maturities) {
        vector<double> price_row;
        for (double K : strikes) {
            Heston_Fourier fourier(S0, K, T, r, v0, kappa, theta, sigma, rho);
            double price = fourier.price_call();
            price_row.push_back(max(0.0, price));
        }
        prices.push_back(price_row);
    }

    json j_data;
    j_data["strikes"] = strikes;
    j_data["maturities"] = maturities;
    j_data["prices"] = prices;
    ofstream out_file(filename);
    if (out_file.is_open()) {
        out_file << j_data.dump(4);
        out_file.close();
        cout << "Heston surface data saved to " << filename << endl;
    } else {
        cerr << "Failed to write to " << filename << endl;
    }
}

void generate_heston_smile_data(double S0, double r, double v0, double kappa, double theta, double sigma, double rho, double T, const string& filename, Financial_Analysis& FA) {
    vector<double> strikes;
    vector<double> times;
    vector<vector<double>> implied_vols;

    // Strike range: 80% to 120% of S0
    for (int i = 0; i < 20; ++i) {
        strikes.push_back(S0 * (0.8 + i * 0.02));
    }
    // Time points: from T to T-0.5 years, stepping by 0.05 years
    for (int i = 0; i <= 10; ++i) {
        times.push_back(T - i * 0.05);
    }

    for (double t : times) {
        if (t <= 0) continue; // Skip non-positive maturities
        vector<double> vol_row;
        for (double K : strikes) {
            Heston_Fourier fourier(S0, K, t, r, v0, kappa, theta, sigma, rho);
            double price = fourier.price_call();
            double implied_vol = FA.impliedVolatility(S0, K, r, t, max(0.01, price));
            vol_row.push_back(implied_vol > 0 && implied_vol < 100 ? implied_vol : 0.2); // Default to 0.2 if invalid
        }
        implied_vols.push_back(vol_row);
    }

    json j_data;
    j_data["strikes"] = strikes;
    j_data["times"] = times;
    j_data["implied_vols"] = implied_vols;
    ofstream out_file(filename);
    if (out_file.is_open()) {
        out_file << j_data.dump(4);
        out_file.close();
        cout << "Heston smile data saved to " << filename << endl;
    } else {
        cerr << "Failed to write to " << filename << endl;
    }
}

void generate_comparison_data(double S0, double r, double sigma, double v0, double kappa, double theta, double xi, double rho, double T, int N_S, int N_t, int max_iter, int num_sims, int num_steps, const string& filename) {
    vector<double> strikes;
    vector<double> bs_prices, bs_fd_prices, bs_mc_prices, heston_fourier_prices, heston_mc_prices, heston_fd_prices;

    // Strike range: $50 to $150
    for (int i = 0; i < 10; ++i) {
        strikes.push_back(S0 * (0.7 + i * 0.05));
    }

    Black_Scholes BS;
    Monte_Carlo MC;
    for (double K : strikes) {
        cout << "Strike = " << K << endl;
        // Black-Scholes Analytical
        double bs_price = BS.call(S0, K, T, r, sigma);
        bs_prices.push_back(max(0.0, bs_price));

        Heston_FD hfd(S0,K,T,r,v0,kappa,theta,xi,rho,101,51,51,100000,true);
        double heston_fd_price = hfd.get_option_price(S0, v0, true);
        heston_fd_prices.push_back(max(0.0, heston_fd_price));

        // Black-Scholes Finite Difference
        Black_Scholes_FD FD(S0, K, T, r, sigma, 101, 51, 100000, true);
        FD.solve();
        double fd_price = FD.get_option_price(S0);
        bs_fd_prices.push_back(max(0.0, fd_price));

        // Black-Scholes Monte Carlo
        double mc_price = MC.option_price(S0, K, T, r, sigma, true, num_sims);
        bs_mc_prices.push_back(max(0.0, mc_price));

        // Heston Fourier
        Heston_Fourier fourier(S0, K, T, r, v0, kappa, theta, xi, rho);
        double fourier_price = fourier.price_call();
        heston_fourier_prices.push_back(max(0.0, fourier_price));

        // Heston Monte Carlo
        double heston_mc_price = MC.Heston_option_price(S0, K, T, r, v0, kappa, theta, xi, rho, true, num_sims, num_steps);
        heston_mc_prices.push_back(max(0.0, heston_mc_price));


    }

    json j_data;
    j_data["strikes"] = strikes;
    j_data["bs_analytical"] = bs_prices;
    j_data["bs_fd"] = bs_fd_prices;
    j_data["bs_mc"] = bs_mc_prices;
    j_data["heston_fourier"] = heston_fourier_prices;
    j_data["heston_mc"] = heston_mc_prices;
    j_data["heston_fd"] = heston_fd_prices;

    ofstream out_file(filename);
    if (out_file.is_open()) {
        out_file << j_data.dump(4);
        out_file.close();
        cout << "Comparison data saved to " << filename << endl;
    } else {
        cerr << "Failed to write to " << filename << endl;
    }
}


// Existing evaluate_trades functions remain unchanged

int main(int argc, char* argv[]) {
    auto start = chrono::high_resolution_clock::now();
    Financial_Analysis FA;
    string ticker = argv[1];
    int strike = atoi(argv[2]);
    double T = atof(argv[3]);
    double r = atof(argv[4]);
    bool read_file = atoi(argv[5]);
    double dt = 1.0 / 252;

    vector<double> close, dayInd, callPrice, callInd;
    vector<OptionData> options;

    auto options1 = fetch_market_data(ticker, close, dayInd, read_file);
    auto options2 = fetch_option_data(ticker, callPrice, callInd, strike, read_file);

    vector<double> aligned_close, aligned_callPrice, aligned_dayInd;
    for (size_t i = 0; i < min(close.size(), callPrice.size()); ++i) {
        if (dayInd[i] == callInd[i]) {
            aligned_close.push_back(close[i]);
            aligned_callPrice.push_back(callPrice[i]);
            aligned_dayInd.push_back(dayInd[i]);
        }
    }

    vector<double> cleaned_close, cleaned_callPrice, cleaned_dayInd;
    for (size_t i = 0; i < aligned_close.size(); ++i) {
        if (aligned_close[i] > 0 && aligned_callPrice[i] > 0 &&
            isfinite(aligned_close[i]) && isfinite(aligned_callPrice[i])) {
            cleaned_close.push_back(aligned_close[i]);
            cleaned_callPrice.push_back(aligned_callPrice[i]);
            cleaned_dayInd.push_back(aligned_dayInd[i]);
        } else {
            cout << "Removed invalid price at day " << aligned_dayInd[i]
                 << ": stock=" << aligned_close[i] << ", option=" << aligned_callPrice[i] << endl;
        }
    }

    double close_mean = computeMean(cleaned_close);
    double close_stddev = computeStdDev(cleaned_close, close_mean);
    double call_mean = computeMean(cleaned_callPrice);
    double call_stddev = computeStdDev(cleaned_callPrice, call_mean);
    vector<double> outlier_free_close, outlier_free_callPrice, outlier_free_dayInd;
    for (size_t i = 0; i < cleaned_close.size(); ++i) {
        double close_z = computeZScore(cleaned_close[i], close_mean, close_stddev);
        double call_z = computeZScore(cleaned_callPrice[i], call_mean, call_stddev);
        if (abs(close_z) < 3.0 && abs(call_z) < 3.0) {
            outlier_free_close.push_back(cleaned_close[i]);
            outlier_free_callPrice.push_back(cleaned_callPrice[i]);
            outlier_free_dayInd.push_back(cleaned_dayInd[i]);
        } else {
            cout << "Removed outlier at day " << cleaned_dayInd[i]
                 << ": stock z=" << close_z << ", option z=" << call_z << endl;
        }
    }

    vector<double> smoothed_callPrice = smoothData(outlier_free_callPrice, 3);
    cout << "Smoothed option prices: " << smoothed_callPrice.size() << " days\n";

    for (size_t i = 0; i < min(outlier_free_close.size(), smoothed_callPrice.size()); ++i) {
        options.push_back({smoothed_callPrice[i], outlier_free_close[i], (double)strike, T - i * dt, true});
    }

    vector<double> impliedVols;
    for (size_t i = 0; i < callPrice.size(); ++i) {
        double vol = FA.impliedVolatility(outlier_free_close[i], strike, r, T - i * dt, smoothed_callPrice[i]);
        if (vol > 0) impliedVols.push_back(vol * vol);
    }

    double vol_mean = computeMean(impliedVols);
    double vol_stddev = computeStdDev(impliedVols, vol_mean);
    vector<double> cleaned_impliedVols;
    for (size_t i = 0; i < impliedVols.size(); ++i) {
        double vol_z = computeZScore(impliedVols[i], vol_mean, vol_stddev);
        if (abs(vol_z) < 2.0 && impliedVols[i] > 0.05 && impliedVols[i] < 100.0) {
            cleaned_impliedVols.push_back(impliedVols[i]);
        } else {
            cout << "Removed outlier volatility at index " << i << ": vol=" << impliedVols[i]
                 << ", z=" << vol_z << endl;
            if (!cleaned_impliedVols.empty()) {
                cleaned_impliedVols.push_back(cleaned_impliedVols.back());
                cout << "Forward-filled volatility at index " << i << endl;
            } else if (i < impliedVols.size() - 1) {
                cleaned_impliedVols.push_back(impliedVols[i + 1]);
                cout << "Used next volatility at index " << i << endl;
            }
        }
    }

    vector<double> smoothed_impliedVols = smoothData(cleaned_impliedVols, 3);
    cout << "Smoothed implied volatilities: " << smoothed_impliedVols.size() << " days\n";

    vector<double> impliedVariances;
    for (double vol : smoothed_impliedVols) {
        impliedVariances.push_back(vol * vol);
    }

    vector<double> returns;
    for (size_t i = 1; i < outlier_free_close.size(); ++i) {
        double ret = log(outlier_free_close[i] / outlier_free_close[i - 1]);
        if (isfinite(ret) && abs(ret) < 0.5) {
            returns.push_back(ret);
        } else {
            cout << "Invalid return at day " << outlier_free_dayInd[i] << ": " << ret << endl;
        }
    }

    double S0 = outlier_free_close.empty() ? 0 : outlier_free_close[0];
    double marketPrice = smoothed_callPrice.empty() ? 0 : smoothed_callPrice[0];
    double impliedVol = FA.impliedVolatility(S0, strike, r, T, marketPrice);
    double v0 = impliedVol * impliedVol;
    cout << "Call - Stock: " << S0 << ", Strike: " << strike << ", Market: " << marketPrice << endl;
    cout << "v0 (Initial Variance): " << v0 << endl;

    double rho = 0.0;
    if (returns.size() >= 2 && impliedVariances.size() >= 2) {
        size_t min_size = min(returns.size(), impliedVariances.size());
        returns.resize(min_size);
        impliedVariances.resize(min_size);
        rho = FA.computeCorrelation(returns, impliedVariances);
        cout << "rho (Correlation): " << rho << endl;
    } else {
        cout << "rho (Correlation): Not enough data\n";
        return 1;
    }

    double kappa, theta, sigma;
    try {
        FA.estimateHestonParameters(impliedVols, S0, (double)strike, r, T, dt, kappa, theta, sigma, rho, v0);
        cout << "kappa (Speed of Mean Reversion): " << kappa << endl;
        cout << "theta (Long-Term Variance): " << theta << endl;
        cout << "sigma (Volatility of Variance): " << sigma << endl;
        cout << "rho (Correlation): " << rho << endl;
    } catch (const std::exception& e) {
        cerr << e.what() << ": using defaults\n";
        kappa = 2.0;
        theta = v0;
        sigma = 0.3;
        rho = -0.5;
    }

    int N_S = 501;
    int N_t = 201;
    int max_iter = 300000;
    int num_sims = 10000;
    int num_steps = 100;
    // Generate data for both plots
    generate_comparison_data(S0, r, 0.2, v0, kappa, theta, sigma, rho, T, N_S, N_t, max_iter, num_sims, num_steps, "option_pricing_comparison.json");
    generate_heston_surface_data(S0, r, v0, kappa, theta, sigma, rho, "heston_surface_data.json");
    generate_heston_smile_data(S0, r, v0, kappa, theta, sigma, rho, T, "heston_smile_data.json", FA);



    //evaluate_trades_fourier(options, r, v0, kappa, theta, sigma, rho);

    auto end = chrono::high_resolution_clock::now();
    auto duration_seconds = chrono::duration<float>(end - start);
    cout << "Algorithm took " << duration_seconds.count() << " seconds" << endl;

    return 0;
}


