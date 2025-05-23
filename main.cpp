#include <cmath>
#include <boost/math/distributions/normal.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <random>
#include "Black_Scholes.h"
#include "Financial_Analysis.h"
#include "Monte_Carlo.h"

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
                      << " (Stock: " << option.S
                      << ", Strike: " << option.K
                      << ", Theoretical: " << theoretical_price
                      << ", Market: " << option.market_price << ")" << endl;
        } else {
            cout << "Don't Buy " << (option.is_call ? "Call" : "Put")
                      << " (Stock: " << option.S
                      << ", Strike: " << option.K
                      << ", Theoretical: " << theoretical_price
                      << ", Market: " << option.market_price << ")" << endl;
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
                      << " (Stock: " << option.S
                      << ", Strike: " << option.K
                      << ", Theoretical: " << theoretical_price
                      << ", Market: " << option.market_price << ")" << endl;
        } else {
            cout << "Don't Buy " << (option.is_call ? "Call" : "Put")
                      << " (Stock: " << option.S
                      << ", Strike: " << option.K
                      << ", Theoretical: " << theoretical_price
                      << ", Market: " << option.market_price << ")" << endl;
        }
    }
}

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    s->append((char*)contents, size * nmemb);
    return size * nmemb;
}

vector<OptionData> fetch_market_data(const string& symbol, vector<double>& close) {
    CURL* curl = curl_easy_init();
    string response;
    vector<OptionData> options;


    if (curl) {
        string url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + symbol + "&apikey=" + API_KEY;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);


        if (res == CURLE_OK) {
            json j = json::parse(response);

            if (j.contains("Time Series (Daily)")) {
                auto time_series = j["Time Series (Daily)"];
                for (auto it = time_series.begin(); it != time_series.end(); ++it) {
                    close.push_back(stod(it.value()["4. close"].get<string>()));
                }

                double S = close[0];

                options.push_back({rng((int)(floor(S/10))), S, S* 1.03, 0.5, true}); // Mock call option (Market)
                options.push_back({rng((int)(floor(S/10))), S, S* 1.03, 0.5, false}); // Mock call option (Market)
            }
        }
    }
    return options;
}

int main(int argc, char* argv[]) {
    Financial_Analysis FA;
    string ticker = argv[1];
    int num_sims = atoi(argv[2]);
    vector<double> close;
    auto options = fetch_market_data(ticker,close);
    double r = 0.03;
    double sigma = FA.calc_volatility(close);
    cout << "Estimated Volatility: " << sigma << std::endl;
    cout << "Black-Scholes" << endl << flush;
    evaluate_trades(options, r, sigma);
    cout << "Monte-Carlo" << endl << flush;
    evaluate_trades_mc(options, r, sigma, num_sims);
    return 0;
}


