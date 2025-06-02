#include <cmath>
#include <boost/math/distributions/normal.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <curl/curl.h>
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

struct OptionData {
    double market_price,S,K,T;
    bool is_call;
};

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

            if (!close.empty()) {
                double S = close[0];
                options.push_back({100.42, S, 105, 0.5, true}); // Mock call option (Market)
            }
            return options;
        }
    }
    else{
        if (curl) {
            string url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + symbol + "&apikey=" + API_KEY;
            cout << "Fetching data from " << url << endl;
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            CURLcode res = curl_easy_perform(curl);
            curl_easy_cleanup(curl);


            if (res == CURLE_OK) {
                json j = json::parse(response);

                if (j.contains("Time Series (Daily)")) {
                    auto time_series = j["Time Series (Daily)"];
                    int ind = 0;
                    for (auto it = time_series.begin(); it != time_series.end(); ++it) {
                        dayInd.push_back(ind);
                        close.push_back(stod(it.value()["4. close"].get<string>()));
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
                    double S = close[0];
                    options.push_back({100.42, S, 105, 0.5, true});
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


int main(int argc, char* argv[]) {
    auto start = chrono::high_resolution_clock::now();
    Financial_Analysis FA;
    string ticker = argv[1];
    int num_sims = atoi(argv[2]);
    bool read_file = atoi(argv[3]);
    vector<double> close;
    vector<double> dayInd;
    auto options = fetch_market_data(ticker,close,dayInd,read_file);
    double r = 0.02; // Risk Free Rate
    double impliedVol;
    double v0;
    for (const auto& option : options) {
        cout << (option.is_call ? "Call" : "Put")
            << " - Stock: " << option.S
                      << ", Strike: " << option.K
                      << ", Market: " << option.market_price << endl;
        impliedVol = (FA.impliedVolatility(option.S,option.K,r,0.5,option.market_price));
        theta =
    }
    v0 = impliedVol * impliedVol;

    double sigma_long = FA.calc_volatility(close,90); //Long term volatility (mean)
    double sigma_short = FA.calc_volatility(close,5); //Initial Volatility
    double v0 = sigma_short * sigma_short; //Initial Variance
    double theta = sigma_long * sigma_long; //Long term variance
    double rho = -0.7; //Correlation between variance and price
    double kappa = 1.0; // Speed of mean reversion
    double xi = 0.1; // Volatility of variance
    int N_S = 501;
    int N_t = 201;
    int max_iter = 300000;



    evaluate_trades(options, r, sigma_long);
    //evaluate_trades_mc(options, r, sigma_long, num_sims);
    //evaluate_trades_FD(options, r, sigma_long, N_S, N_t, max_iter);
    //evaluate_trades_mch(options, r, v0, kappa, theta, xi, rho, num_sims,100);
    evaluate_trades_fourier(options, r, v0, kappa, theta, xi, rho);
    evaluate_trades_HestonFD(options, r, v0, theta, kappa, xi, rho);
        //double pred_S = Estimate_Stock(dayInd,close);
    //cout << "Predicted Stock Tomorrow: " << pred_S << endl
    auto end = chrono::high_resolution_clock::now();
    auto duration_seconds = chrono::duration<float>(end - start);
    cout << "Algorithm took " << duration_seconds.count() << " seconds" << endl;

    return 0;
}



