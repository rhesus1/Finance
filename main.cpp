#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <chrono>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <boost/math/distributions/normal.hpp>
#include "Black_Scholes.h"
#include "Financial_Analysis.h"
#include "Monte_Carlo.h"
#include "Linear_Regression.h"
#include "Black_Scholes_FD.h"
#include "Heston_FD.h"
#include "Heston_Fourier.h"

using namespace std;
using json = nlohmann::json;

random_device rd;
mt19937 gen(rd());

bool debug = false;

double rng() {
    uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen);
}

size_t WriteCallback(void* contents, size_t size, size_t nmemb, string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

double days_to_years(const string& expiry_str) {
    const int ref_year = 2025, ref_month = 6, ref_day = 2;

    int year = stoi(expiry_str.substr(0, 2)) + 2000;
    int month = stoi(expiry_str.substr(2, 2));
    int day = stoi(expiry_str.substr(4, 2));

    auto to_days = [](int y, int m, int d) {
        return y * 365 + m * 30 + d;
    };
    double ref_days = to_days(ref_year, ref_month, ref_day);
    double expiry_days = to_days(year, month, day);
    double days_diff = expiry_days - ref_days;

    if (days_diff <= 0) return 0.01;
    return days_diff / 365.0;
}

struct OptionData {
    double market_price, S, K, T, volume, bid_ask_spread;
    bool is_call;
};

struct HestonParams {
    double v0, kappa, theta, sigma, rho;
};

struct Vertex {
    vector<double> x;
    double f;
};

vector<double> nelderMeadOptimize(
    function<double(const vector<double>&, vector<double>&, void*)> obj_func,
    void* data,
    const vector<double>& x0,
    const vector<double>& lb,
    const vector<double>& ub,
    int max_iter,
    double tol
) {
    const int n = x0.size();
    vector<Vertex> simplex(n + 1);
    vector<double> grad(n, 0.0);
    double alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5;
    int max_restarts = 3;
    double best_f = 1e10;
    vector<double> best_x = x0;

    for (int restart = 0; restart <= max_restarts; ++restart) {
        simplex[0].x = (restart == 0) ? x0 : best_x;
        for (int i = 0; i < n; ++i) {
            simplex[0].x[i] = max(lb[i], min(ub[i], simplex[0].x[i]));
        }
        simplex[0].f = obj_func(simplex[0].x, grad, data);
        for (int i = 1; i <= n; ++i) {
            simplex[i].x = simplex[0].x;
            simplex[i].x[i - 1] += 0.1 * (ub[i - 1] - lb[i - 1]);
            for (int j = 0; j < n; ++j) {
                simplex[i].x[j] = max(lb[j], min(ub[j], simplex[i].x[j]));
            }
            simplex[i].f = obj_func(simplex[i].x, grad, data);
        }

        for (int iter = 0; iter < max_iter; ++iter) {
            sort(simplex.begin(), simplex.end(), [](const Vertex& a, const Vertex& b) {
                return a.f < b.f;
            });

            if (abs(simplex[n].f - simplex[0].f) < tol && simplex[0].f < best_f) {
                best_f = simplex[0].f;
                best_x = simplex[0].x;
            }
            if (abs(simplex[n].f - simplex[0].f) < tol) break;

            vector<double> centroid(n, 0.0);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    centroid[j] += simplex[i].x[j];
                }
            }
            for (int j = 0; j < n; ++j) {
                centroid[j] /= n;
            }

            vector<double> xr(n);
            for (int j = 0; j < n; ++j) {
                xr[j] = centroid[j] + alpha * (centroid[j] - simplex[n].x[j]);
                xr[j] = max(lb[j], min(ub[j], xr[j]));
            }
            double fr = obj_func(xr, grad, data);

            if (fr < simplex[0].f) {
                vector<double> xe(n);
                for (int j = 0; j < n; ++j) {
                    xe[j] = centroid[j] + gamma * (xr[j] - centroid[j]);
                    xe[j] = max(lb[j], min(ub[j], xe[j]));
                }
                double fe = obj_func(xe, grad, data);
                if (fe < sigma) {
                    simplex[n].x = xe;
                    simplex[n].f = fe;
                } else {
                    simplex[n].x = xr;
                    simplex[n].f = fr;
                }
            } else if (fr < simplex[n - 1].f) {
                simplex[n].x = xr;
                simplex[n].f = fr;
            } else {
                vector<double> xc(n);
                for (int j = 0; j < n; ++j) {
                    xc[j] = centroid[j] + rho * (simplex[n].x[j] - centroid[j]);
                    xc[j] = max(lb[j], min(ub[j], xc[j]));
                }
                double fc = obj_func(xc, grad, data);
                if (fc < simplex[n].f) {
                    simplex[n].x = xc;
                    simplex[n].f = fc;
                } else {
                    for (int i = 1; i <= n; ++i) {
                        for (int j = 0; j < n; ++j) {
                            simplex[i].x[j] = simplex[0].x[j] + sigma * (simplex[i].x[j] - simplex[0].x[j]);
                            simplex[i].x[j] = max(lb[j], min(ub[j], simplex[i].x[j]));
                        }
                        simplex[i].f = obj_func(simplex[i].x, grad, data);
                    }
                }
            }
        }
        if (restart < max_restarts) {
            for (int i = 0; i < n; ++i) {
                best_x[i] += 0.05 * (ub[i] - lb[i]) * (rng() - 0.5);
                best_x[i] = max(lb[i], min(ub[i], best_x[i]));
            }
        }
    }

    return best_x;
}

/*vector<double> differentialEvolution(
    function<double(const vector<double>&, vector<double>&, void*, const vector<double>&, const vector<double>&)> obj_func,
    void* data, const vector<double>& x0, const vector<double>& lb, const vector<double>& ub,
    int max_gen, double tol) {
    int n = x0.size(), pop_size = 100;
    vector<vector<double>> pop(pop_size, vector<double>(n));
    vector<double> fitness(pop_size);
    vector<double> grad(n, 0.0);
    // Initialize population
    for (auto& p : pop) {
        for (int i = 0; i < n; ++i) {
            p[i] = lb[i] + rng() * (ub[i] - lb[i]);
        }
    }
    // Evaluate initial population
    for (int i = 0; i < pop_size; ++i) {
        fitness[i] = obj_func(pop[i], grad, data, lb, ub);
    }
    for (int gen = 0; gen < max_gen; ++gen) {
        for (int i = 0; i < pop_size; ++i) {
            // Select three random individuals
            int a, b, c;
            do { a = rng() * pop_size; } while (a == i);
            do { b = rng() * pop_size; } while (b == i || b == a);
            do { c = rng() * pop_size; } while (c == i || c == a || c == b);
            // Mutation and crossover
            vector<double> trial(n);
            double F = 0.6, CR = 0.8;
            for (int j = 0; j < n; ++j) {
                if (rng() < CR || j == (int)(rng() * n)) {
                    trial[j] = pop[a][j] + F * (pop[b][j] - pop[c][j]);
                    trial[j] = max(lb[j], min(ub[j], trial[j]));
                } else {
                    trial[j] = pop[i][j];
                }
            }
            // Selection
            double trial_f = obj_func(trial, grad, data, lb, ub);
            if (trial_f < fitness[i]) {
                pop[i] = trial;
                fitness[i] = trial_f;
            }
        }
        // Check convergence
        auto [min_f, max_f] = minmax_element(fitness.begin(), fitness.end());
        if (*max_f - *min_f < tol) break;
    }
    auto best_idx = min_element(fitness.begin(), fitness.end()) - fitness.begin();
    return pop[best_idx];
}
*/
void fetch_market_data(const string& symbol, vector<double>& close, vector<double>& dayInd, vector<long long>& timestamps, bool read_file) {
    string filename = symbol + "_market_data.json";
    if (read_file) {
        ifstream in_file(filename);
        if (in_file.is_open()) {
            cout << "Reading " << filename << endl;
            json j;
            try {
                in_file >> j;
                close = j["close"].get<vector<double>>();
                dayInd = j["dayInd"].get<vector<double>>();
                timestamps = j["timestamps"].get<vector<long long>>();
            } catch (const json::exception& e) {
                cerr << "JSON parse error in " << filename << ": " << e.what() << endl;
            }
            in_file.close();
            return;
        } else {
            cerr << "Failed to open " << filename << endl;
        }
    }
    CURL* curl = curl_easy_init();
    string response;
    if (curl) {
        string api_key = "nbOtOYuPxQXHpkwnFrsTQfk6OFaw1SBO";
        string url = "https://api.polygon.io/v2/aggs/ticker/" + symbol + "/range/1/day/2024-01-01/2025-06-08?adjusted=true&sort=asc&limit=50000&apiKey=" + api_key;
        cout << "Fetching Stock Data from: " << url << endl;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            const char* ca_cert_path = "cacert.pem";
    curl_easy_setopt(curl, CURLOPT_CAINFO, ca_cert_path);
        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        if (res != CURLE_OK) {
            cerr << "CURL error: " << curl_easy_strerror(res) << endl;
            return;
        }
        try {
            json j = json::parse(response);
            if (j.contains("results")) {
                int ind = 0;
                for (const auto& result : j["results"]) {
                    dayInd.push_back(ind++);
                    close.push_back(result["c"].get<double>());
                    timestamps.push_back(result["t"].get<long long>());
                }
                json j_data;
                j_data["close"] = close;
                j_data["dayInd"] = dayInd;
                j_data["timestamps"] = timestamps;
                ofstream out_file(filename);
                if (out_file.is_open()) {
                    out_file << j_data.dump(4);
                    out_file.close();
                } else {
                    cerr << "Failed to write to " << filename << endl;
                }
            } else {
                cerr << "No results in API response" << endl;
            }
        } catch (const json::parse_error& e) {
            cerr << "JSON parse error: " << e.what() << endl;
        }
    } else {
        cerr << "Failed to initialize CURL" << endl;
    }
}

vector<OptionData> fetch_option_data(const string& symbol, const vector<double>& stock_prices, const vector<long long>& stock_timestamps, bool read_file = false) {
    string filename = symbol + "_option_data.json";
    vector<OptionData> options_data;
    if (read_file) {
        ifstream ifs(filename);
        if (ifs.is_open()) {
            json j;
            try {
                ifs >> j;
                for (const auto& item : j["results"]) {
                    OptionData opt;
                    opt.market_price = item["last_price"].get<double>();
                    opt.S = item["underlying_price"].get<double>();
                    opt.K = item["strike_price"].get<double>();
                    opt.T = item["expiration_date"].get<double>();
                    opt.volume = item["volume"].get<double>();
                    if (item.contains("ask") && item.contains("bid") && item["ask"].is_number() && item["bid"].is_number()) {
                        opt.bid_ask_spread = item["ask"].get<double>() - item["bid"].get<double>();
                    } else {
                        opt.bid_ask_spread = 0.0;
                    }
                    opt.is_call = item["option_type"] == "call";
                    if (opt.market_price > 0.01 && opt.volume > 10 && opt.bid_ask_spread <= 0.05 * opt.market_price) {
                        options_data.push_back(opt);
                    }
                }
                cout << "Loaded " << options_data.size() << " valid options from file." << endl;
            } catch (const json::exception& e) {
                cerr << "JSON parse error in " << filename << ": " << e.what() << endl;
            }
            ifs.close();
            return options_data;
        } else {
            cerr << "Failed to open " << filename << endl;
        }
    }
    CURL* curl = curl_easy_init();
    string response;
    const char* ca_cert_path = "cacert.pem";
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_CAINFO, ca_cert_path);
        string api_key = "nbOtOYuPxQXHpkwnFrsTQfk6OFaw1SBO";
        vector<string> expiries = {"250620", "250718", "250815", "250919", "251017", "251121", "251219"};
        vector<double> strikes = {100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300};
        for (const auto& expiry : expiries) {
            for (double K : strikes) {
                for (const auto& type : {"call", "put"}) {
                    string contract_type = (type == "call" ? "call" : "put");
                    string expiry_date = "20" + expiry.substr(0, 2) + "-" + expiry.substr(2, 2) + "-" + expiry.substr(4, 2);
                    string url = "https://api.polygon.io/v3/snapshot/options/" + symbol + "?strike_price=" + to_string((int)K) +
                                 "&expiration_date=" + expiry_date + "&contract_type=" + contract_type +
                                 "&order=asc&limit=1&sort=ticker&apiKey=" + api_key;
                    cout << "Fetching " << type << " Option data for strike " << K << ", expiry " << expiry << " from: " << url <<endl;
                    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
                    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
                    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
                    response.clear();
                    CURLcode res = curl_easy_perform(curl);
                    if (res != CURLE_OK) {
                        cerr << "CURL error for " << url << ": " << curl_easy_strerror(res) << endl;
                        continue;
                    }
                    try {
                        json j = json::parse(response);
                        if (!j.contains("results") || !j["results"].is_array() || j["results"].empty()) {
                            continue;
                        }
                        const auto& result = j["results"][0];
                        if (!result.contains("day") || !result.contains("details") ||
                            !result["day"].contains("close") || !result["day"].contains("volume") || !result["day"].contains("last_updated") ||
                            !result["details"].contains("strike_price") || !result["details"].contains("expiration_date") ||
                            !result["details"].contains("contract_type")) {
                            continue;
                        }
                        long long option_ts = result["day"]["last_updated"].get<long long>() / 1000000;
                        double S = 0.0;
                        if (!stock_timestamps.empty()) {
                            auto it = min_element(stock_timestamps.begin(), stock_timestamps.end(),
                                [option_ts](long long a, long long b) {
                                    return abs(a - option_ts) < abs(b - option_ts);
                                });
                            size_t idx = distance(stock_timestamps.begin(), it);
                            S = stock_prices[idx];
                        } else {
                            S = stock_prices.back();
                        }
                        OptionData opt;
                        opt.market_price = result["day"]["close"].get<double>();
                        opt.K = result["details"]["strike_price"].get<double>();
                        opt.T = days_to_years(expiry);
                        opt.S = S;
                        opt.volume = result["day"]["volume"].get<double>();
                        opt.bid_ask_spread = 0.0;//result["ask"].get<double>() - result["bid"].get<double>();
                        opt.is_call = result["details"]["contract_type"] == "call";
                        if (opt.market_price > 0.01 && opt.volume > 100 && opt.bid_ask_spread <= 0.05 * opt.market_price) {
                            Financial_Analysis fa;
                            double iv = fa.impliedVolatility(opt.S, opt.K, 0.05, opt.T, opt.market_price);
                            cout << "Implied_vol = " << iv << endl;
                            if (iv > 0.05 && iv < 2.0) {
                                options_data.push_back(opt);
                            } else {
                                cout << "Not Good " << endl;
                            }
                        }
                    } catch (const json::exception& e) {
                        cerr << "JSON error for option " << url << ": " << e.what() << endl;
                    }
                }
            }
        }
        curl_easy_cleanup(curl);
    } else {
        cerr << "Failed to initialize CURL" << endl;
    }
    cout << "Fetched " << options_data.size() << " valid options from API." << endl;
    json j;
    j["results"] = json::array();
    for (const auto& opt : options_data) {
        json item;
        item["last_price"] = opt.market_price;
        item["underlying_price"] = opt.S;
        item["strike_price"] = opt.K;
        item["expiration_date"] = opt.T;//to_string(int(opt.T * 365));
        item["volume"] = opt.volume;
        item["ask"] = opt.market_price + opt.bid_ask_spread / 2;
        item["bid"] = opt.market_price - opt.bid_ask_spread / 2;
        item["option_type"] = opt.is_call ? "call" : "put";
        j["results"].push_back(item);
    }
    ofstream ofs(filename);
    if (ofs.is_open()) {
        ofs << j.dump(4);
        ofs.close();
    }
    return options_data;
}



double computeMean(const vector<double>& data) {
    if (data.empty()) return 0.0;
    double sum = 0.0;
    for (double x : data) sum += x;
    return sum / data.size();
}

double computeStdDev(const vector<double>& data, double mean) {
    if (data.size() < 2) return 0.0;
    double sum = 0.0;
    for (double x : data) sum += (x - mean) * (x - mean);
    return sqrt(sum / (data.size() - 1));
}

vector<double> computeZScore(const vector<double>& data, double mean, double stddev) {
    vector<double> z_scores;
    for (double x : data) z_scores.push_back(stddev > 0 ? (x - mean) / stddev : 0.0);
    return z_scores;
}

vector<double> smoothData(const vector<double>& data, int window_size = 5) {
    vector<double> smoothed;
    for (size_t i = 0; i < data.size(); ++i) {
        double sum = 0.0;
        int count = 0;
        for (int j = -window_size / 2; j <= window_size / 2; ++j) {
            if (i + j >= 0 && i + j < data.size()) {
                sum += data[i + j];
                count++;
            }
        }
        smoothed.push_back(sum / count);
    }
    return smoothed;
}

double Estimate_Stock(const vector<double>& prices) {
    Linear_Regression lr(0.01, 1000);
    vector<double> x(prices.size());
    for (size_t i = 0; i < x.size(); ++i) x[i] = i;
    lr.train(x, prices);
    return lr.predict(x.back() + 1);
}

double EstimateVolatility(const vector<OptionData>& option_data, double S0) {
    Financial_Analysis fa;
    vector<double> implied_vols;
    for (const auto& option : option_data) {
        if (option.is_call && abs(option.K / S0 - 1.0) <= 0.05) {
            double sigma = fa.impliedVolatility(option.S, option.K, 0.05, option.T, option.market_price);
            if (sigma > 0.0 && isfinite(sigma)) implied_vols.push_back(sigma);
        }
    }
    if (implied_vols.empty()) return 0.2;
    return computeMean(implied_vols);
}

HestonParams HestonInitial(const vector<OptionData>& option_data, double S0) {
    Financial_Analysis fa;
    HestonParams params;

    vector<double> ivs_short, ivs_long;
        double sum = 0.0, count = 0.0;
    for (const auto& option : option_data) {
        if (option.is_call && abs(option.K / option.S - 1.0) <= 0.15) {
            double iv = fa.impliedVolatility(option.S, option.K, 0.05, option.T, option.market_price);
            if (isfinite(iv)) {
                if (option.T < 0.25) ivs_short.push_back(iv * iv);
                else ivs_long.push_back(iv * iv);
            }
        }
        if (option.T < 0.25 && abs(option.K / option.S - 1.0) > 0.1) {
            double iv = fa.impliedVolatility(option.S, option.K, 0.05, option.T, option.market_price);
            if (isfinite(iv)) {
                sum += (option.K > option.S ? -1.0 : 1.0) * iv;
                count += 1.0;
            }
        }
    }
    params.v0 = ivs_short.empty() ? 0.04 : computeMean(ivs_short);
    params.theta = ivs_long.empty() ? 0.04 : computeMean(ivs_long);
params.kappa = 2.0;//ivs_short.empty() || ivs_long.empty() ? 2.0 : 2.0 * abs(computeMean(ivs_short) - computeMean(ivs_long)) / max(0.01, computeMean(ivs_long));
    params.sigma = ivs_short.empty() ? 0.2 : computeStdDev(ivs_short, params.v0) * sqrt(2.0);
params.rho = -0.5;//count > 0 ? max(-0.9, min(-0.1, sum / count)) : -0.5;

    return params;
}
/*
double hestonObjective(const vector<double>& x, vector<double>& grad, void* data, const vector<double>& lb, const vector<double>& ub) {
    double lambda_reg = 1e-3;
    double feller_penalty = 100;
    auto* params = static_cast<pair<vector<OptionData>, double>*>(data);
    const auto& option_data = params->first;
    double S0 = params->second;
    double v0 = x[0], kappa = x[1], theta = x[2], sigma = x[3], rho = x[4];
    double loss = 0.0;
    double total_weight = 0.0;

    vector<double> losses(option_data.size(), 0.0);
    try {
        //#pragma omp parallel for reduction(+:loss,total_weight)
        for (size_t i = 0; i < option_data.size(); ++i) {
            const auto& option = option_data[i];
            Heston_Fourier hf(option.S, option.K, option.T, 0.05, v0, kappa, theta, sigma, rho);
            double model_price = option.is_call ? hf.price_call() : hf.price_put();
            double vega = hf.vega(option.is_call);
            double weight = vega > 1e-6 ? vega : 1e-6;
            Financial_Analysis fa;
            double model_iv = fa.impliedVolatility(option.S, option.K, 0.05, option.T, model_price);
            double market_iv = fa.impliedVolatility(option.S, option.K, 0.05, option.T, option.market_price);
            double diff;
            if (isfinite(model_iv) && isfinite(market_iv)) {
                diff = model_iv - market_iv;
                losses[i] = weight * diff * diff;
            } else {
                losses[i] = 10.0;
            }
            loss += losses[i];
            total_weight += weight;
        }
        if (total_weight > 1e-6) {
            loss /= total_weight; // Normalize by sum of weights
        } else {
            loss /= option_data.size();
        }
        //double feller = 2.0 * kappa * theta - sigma * sigma;
        double penalty = 0.0;// = feller <= 0.0 ? 10.0 * (-feller * - feller) : 0.0;
        double reg_term = lambda_reg * (v0 * v0 + kappa * kappa + theta * theta + sigma * sigma + rho * rho);
        /*if (rho < -0.9) {
            loss += 100.0 * (rho + 0.9) * (rho + 0.9);
        }*/
/*
        double penalty_weight = 100.0;
        for (size_t i = 0; i < x.size(); ++i) {
            if (x[i] <= lb[i]) {
                penalty += penalty_weight * std::pow(lb[i] - x[i], 2);
            } else if (x[i] >= ub[i]) {
                penalty += penalty_weight * std::pow(x[i] - ub[i], 2);
            }
        }


        loss += penalty + reg_term;
    } catch (const exception& e) {
        ofstream ofs("optimization_log.txt", ios::app);
        ofs << "Exception: " << e.what() << ", v0=" << v0 << ", kappa=" << kappa << ", theta=" << theta << ", sigma=" << sigma << ", rho=" << rho << endl;
        return 1e6;
    }
    return loss;
}

double hestonObjectiveWithLog(const vector<double>& x, vector<double>& grad, void* data, vector<double>& lb, vector<double>& ub) {
    auto* params = static_cast<pair<vector<OptionData>, double>*>(data);
    double loss = hestonObjective(x, grad, data, lb, ub);
    ofstream ofs("optimization_log.txt", ios::app);
    ofs << "Parameters: v0=" << x[0] << ", kappa=" << x[1] << ", theta=" << x[2] << ", sigma=" << x[3] << ", rho=" << x[4]
        << ", loss=" << loss << ", Feller=" << (2 * x[1] * x[2] - x[3] * x[3]) << endl;
    return loss;
}

HestonParams calibrateHestonParameters(const vector<OptionData>& option_data, double S0, bool recalibrate = false) {
    string filename = "AMZN_heston_params.json";
    if (!recalibrate) {
        ifstream ifs(filename);
        if (ifs.is_open()) {
            json j;
            try {
                ifs >> j;
                HestonParams params;
                params.v0 = j["v0"].get<double>();
                params.kappa = j["kappa"].get<double>();
                params.theta = j["theta"].get<double>();
                params.sigma = j["sigma"].get<double>();
                params.rho = j["rho"].get<double>();
                ifs.close();
                return params;
            } catch (const json::exception& e) {
                cerr << "JSON parse error in " << filename << ": " << e.what() << endl;
            }
            ifs.close();
        }
    }
    HestonParams init_params = HestonInitial(option_data, S0);
    vector<double> x = {init_params.v0, init_params.kappa, init_params.theta, init_params.sigma, init_params.rho};
    cout << "Initial Parameters" << endl;
    cout << "v0 = " << init_params.v0 << endl;
    cout << "kappa = " << init_params.kappa << endl;
    cout << "theta = " << init_params.theta << endl;
    cout << "sigma = " << init_params.sigma << endl;
    cout << "rho = " << init_params.rho << endl;

    vector<double> lb = {0.001, 0.001, 0.001, 0.01, -1.0};
    vector<double> ub = {0.1, 10.0, 1.0, 2.0, 0.0};
    int max_iter = 10000;
    double tol = 1e-10;
    pair<vector<OptionData>, double> data = {option_data, S0};
    int max_gen = 1000;
    //vector<double> result = nelderMeadOptimize(hestonObjectiveWithLog, &data, x, lb, ub, max_iter, tol);
    //differentialEvolution(hestonObjectiveWithLog, &data, x, lb, ub, max_gen, tol);
    //

    vector<double> global_result = differentialEvolution(hestonObjectiveWithLog, &data, x, lb, ub, max_gen, tol);
        cout << "Global result" << endl;
    cout << "v0 = " << global_result[0] << endl;
    cout << "kappa = " << global_result[1] << endl;
    cout << "theta = " << global_result[2] << endl;
    cout << "sigma = " << global_result[3] << endl;
    cout << "rho = " << global_result[4] << endl;
//vector<double> result = nelderMeadOptimize(hestonObjectiveWithLog, &data, x, lb, ub, max_iter, tol);
 /*cout << "Lcaol result" << endl;
    cout << "v0 = " << result[0] << endl;
    cout << "kappa = " << result[1] << endl;
    cout << "theta = " << result[2] << endl;
    cout << "sigma = " << result[3] << endl;
    cout << "rho = " << result[4] << endl;*/

  /*  HestonParams params;// = init_params;
    params.v0 = global_result[0];
    params.kappa = global_result[1];
    params.theta = global_result[2];
    params.sigma = global_result[3];
    params.rho = global_result[4];

    json j;
    j["v0"] = params.v0;
    j["kappa"] = params.kappa;
    j["theta"] = params.theta;
    j["sigma"] = params.sigma;
    j["rho"] = params.rho;
    ofstream ofs(filename);
    if (ofs.is_open()) {
        ofs << j.dump(4);
        ofs.close();
    }
    return params;
}*/

void generate_heston_surface_data(const string& symbol, double S0, double r, const HestonParams& params) {
    Financial_Analysis fa;
    vector<double> strikes = {90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280};
    vector<double> maturities = {0.0833, 0.1667, 0.25, 0.3333, 0.4167, 0.5, 0.5833, 0.6667, 0.75, 0.8333, 0.9167, 1.0};
    json j;
    j["symbol"] = symbol;
    j["S0"] = S0;
    j["r"] = r;
    j["params"] = {{"v0", params.v0}, {"kappa", params.kappa}, {"theta", params.theta}, {"sigma", params.sigma}, {"rho", params.rho}};
    j["data"] = json::array();
    int Trap = 1;
    double Lphi = 0.00001;
    double Uphi = 50;
    double dphi = 0.001;
    double q = 0.0;
    for (double T : maturities) {
        for (double K : strikes) {
            HestonFourier heston(S0, K, params.v0, T, r, q, params.kappa, params.theta, params.sigma, params.rho, Lphi, Uphi, dphi, debug);
            double HestonC, HestonP;
            heston.HestonPrice("Both", Trap, HestonC, HestonP);
            double iv = fa.impliedVolatility(S0, K, r, T, HestonC);
            json item;
            item["strike"] = K;
            item["maturity"] = T;
            item["call_price"] = HestonC;
            item["put_price"] = HestonP;
            item["implied_vol"] = iv;
            j["data"].push_back(item);
        }
    }
    ofstream ofs(symbol + "_heston_surface_data.json");
    if (ofs.is_open()) {
        ofs << j.dump(4);
        ofs.close();
    }
}


void generate_comparison_data(const string& symbol, double S0, double r, double sigma, const HestonParams& heston_params, double T = 0.25) {
    vector<double> strikes;
    strikes = {160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270};
    json call_j, put_j;
    call_j["symbol"] = symbol;
    call_j["S0"] = S0;
    call_j["r"] = r;
    call_j["T"] = T;
    call_j["sigma"] = sigma;
    call_j["heston_params"] = {{"v0", heston_params.v0}, {"kappa", heston_params.kappa}, {"theta", heston_params.theta}, {"sigma", heston_params.sigma}, {"rho", heston_params.rho}};
    call_j["data"] = json::array();
    put_j["symbol"] = symbol;
    put_j["S0"] = S0;
    put_j["r"] = r;
    put_j["T"] = T;
    put_j["sigma"] = sigma;
    put_j["heston_params"] = {{"v0", heston_params.v0}, {"kappa", heston_params.kappa}, {"theta", heston_params.theta}, {"sigma", heston_params.sigma}, {"rho", heston_params.rho}};
    put_j["data"] = json::array();
    Black_Scholes bs;
    Monte_Carlo mc;
    for (double K : strikes) {
        json call_item, put_item;
        call_item["strike"] = K;
        put_item["strike"] = K;
        call_item["bs_analytical"] = bs.call(S0, K, T, r, sigma);
        put_item["bs_analytical"] = bs.put(S0, K, T, r, sigma);
        call_item["bs_mc"] = mc.option_price(S0, K, T, r, sigma, true, 200000);
        put_item["bs_mc"] = mc.option_price(S0, K, T, r, sigma, false, 200000);


        int Trap = 1;
        double Lphi = 0.00001;
        double Uphi = 50;
        double dphi = 0.001;
        double q = 0.0;

        HestonFourier heston(S0, K, heston_params.v0, T, r, q, heston_params.kappa, heston_params.theta, heston_params.sigma, heston_params.rho, Lphi, Uphi, dphi, debug);
        double HestonC, HestonP;
        heston.HestonPrice("Both", Trap, HestonC, HestonP);

        call_item["heston_fourier"] = HestonC;
        cout << "Heston Fourier Call Price at Strike " << K << " = " << HestonC << endl;
        put_item["heston_fourier"] = HestonP;
        cout << "Heston Fourier Put Price at Strike " << K << " = " << HestonP << endl;


        call_item["heston_mc"] = mc.Heston_option_price(S0, K, T, r, heston_params.v0, heston_params.kappa, heston_params.theta, heston_params.sigma, heston_params.rho, true, 100000, 500);
        put_item["heston_mc"] = mc.Heston_option_price(S0, K, T, r, heston_params.v0, heston_params.kappa, heston_params.theta, heston_params.sigma, heston_params.rho, false, 100000, 500);
        Black_Scholes_FD bsfd(S0, K, T, r, sigma, 101, 51, 10000, true);
        bsfd.solve();
        call_item["bs_fd"] = bsfd.get_option_price(S0);
        Black_Scholes_FD bsfd_p(S0, K, T, r, sigma, 101, 51, 10000, false);
        bsfd_p.solve();
        put_item["bs_fd"] = bsfd_p.get_option_price(S0);
        // Heston_FD hfd(S0, K, T, r, heston_params.v0, heston_params.kappa, heston_params.theta, heston_params.sigma, heston_params.rho, 100, 100, 100, true);
        // call_item.put("heston_fd"] = hfd.get_option_price(S0, heston_params.v0, true);
        call_j["data"].push_back(call_item);
        put_j["data"].push_back(put_item);
    }
    ofstream call_ofs(symbol + "_call_option_pricing_comparison.json");
    if (call_ofs.is_open()) {
            call_ofs << call_j.dump(4);
            call_ofs.close();
    }
    ofstream put_ofs(symbol + "_put_option_pricing_comparison.json");
    if (put_ofs.is_open()) {
        put_ofs << put_j.dump(4);
        put_ofs.close();
    }
}

int main(int argc, char** argv) {
    string symbol = "AMZN";
    double r = 0.035;
    bool read_file = false;
    bool recalibrate = false;
    for (int i = 1; i < argc; i++) {
        string arg = string(argv[i]);
        if ((arg == "--ticker" || arg == "-t") && i + 1 < argc) {
            symbol = argv[++i];
        } else if (arg == "--read-file" || arg == "-r") {
            read_file = true;
        } else if (arg == "--recalibrate" || arg == "-c") {
            recalibrate = true;
        } else if (arg == "--debug" || arg == "-d"){
            debug = true;
        }
    }
    vector<double> close, dayInd;
vector<long long> timestamps;
fetch_market_data(symbol, close, dayInd, timestamps, read_file);
if (close.empty()) {
    cerr << "Error: No market data available for " << symbol << endl;
    return 1;
}
double S0 = close.back();
vector<OptionData> option_data = fetch_option_data(symbol, close, timestamps, read_file);
    if (option_data.empty()) {
        cerr << "Error: No option data available for " << symbol << endl;
        return 1;
    }
    double estimated_price = Estimate_Stock(close);
    double sigma = EstimateVolatility(option_data, S0);
    HestonParams heston_params;// = calibrateHestonParameters(option_data, S0, recalibrate);
    heston_params.v0 = 0.084873;
    heston_params.kappa = 6.237368;
    heston_params.theta = 0.067924;
    heston_params.sigma = 0.920504;
    heston_params.rho = -0.755989;
    cout << "Stock Price: " << S0 << endl;
    cout << "Estimated Stock Price: " << estimated_price << endl;
    cout << "Estimated Volatility: " << sigma << endl;
    cout << "Heston Parameters: v0=" << heston_params.v0 << ", kappa=" << heston_params.kappa << ", theta=" << heston_params.theta << ", sigma=" << heston_params.sigma << ", rho=" << heston_params.rho << endl;
    //generate_heston_surface_data(symbol, S0, r, heston_params);
    generate_comparison_data(symbol, S0, r, sigma, heston_params);
    return 0;
}