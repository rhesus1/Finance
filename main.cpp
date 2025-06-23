#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <random>
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
#include "Black_Scholes_FD_simd.h"
#include "LSTM.h"
#include <map>
#include <algorithm>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using json = nlohmann::json;

random_device rd;
mt19937 gen(rd());

bool debug = false;

double rng() {
    uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen);
}

double random_num(double min, double max) {
    uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

size_t WriteCallback(void* contents, size_t size, size_t nmemb, string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

double days_to_years(const string& expiry_str) {
    const int ref_year = 2025, ref_month = 6, ref_day = 15;
    int year = stoi(expiry_str.substr(0, 2)) + 2000;
    int month = stoi(expiry_str.substr(2, 2));
    int day = stoi(expiry_str.substr(4, 2));
    auto to_days = [](int y, int m, int d) { return y * 365 + m * 30 + d; };
    double ref_days = to_days(ref_year, ref_month, ref_day);
    double expiry_days = to_days(year, month, day);
    double days_diff = expiry_days - ref_days;
    return (days_diff <= 0) ? 0.01 : days_diff / 365.0;
}

struct OptionData {
    double market_price, S, K, T, volume, bid_ask_spread;
    bool is_call;
};

struct HestonParams {
    double v0, kappa, theta, sigma, rho;
};

struct NMSettings {
    int N;
    int MaxIters;
    double Tolerance;
    double alpha, gamma, rho, sigma;
};

struct SimplexVertex {
    vector<double> params;
    double value;
};

struct HestonOptData {
    vector<OptionData> option_data;
    double S0;
    vector<double> lb;
    vector<double> ub;
    bool use_eu;
};

void fetch_market_data(const string& symbol, vector<double>& close, vector<double>& dayInd, vector<long long>& timestamps, vector<double>& volumes, vector<double>& highs, vector<double>& lows, bool read_file) {
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
                volumes = j["volumes"].get<vector<double>>();
                highs = j["highs"].get<vector<double>>();
                lows = j["lows"].get<vector<double>>();
            } catch (const json::exception& e) {
                cerr << "JSON parse error in " << filename << ": " << e.what() << endl;
            }
            in_file.close();
            cout << "Loaded " << close.size() << " valid stocks from file." << endl;
            return;
        } else {
            cerr << "Failed to open " << filename << endl;
        }
    }
    CURL* curl = curl_easy_init();
    string response;
    if (curl) {
        string api_key = "nbOtOYuPxQXHpkwnFrsTQfk6OFaw1SBO";
        string url = "https://api.polygon.io/v2/aggs/ticker/" + symbol + "/range/1/day/2022-01-01/2025-06-15?adjusted=true&sort=asc&limit=50000&apiKey=" + api_key;
        cout << "Fetching Stock Data from: " << url << endl;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_CAINFO, "cacert.pem");
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
                    volumes.push_back(result["v"].get<double>());
                    highs.push_back(result["h"].get<double>());
                    lows.push_back(result["l"].get<double>());
                }
                json j_data;
                j_data["close"] = close;
                j_data["dayInd"] = dayInd;
                j_data["timestamps"] = timestamps;
                j_data["volumes"] = volumes;
                j_data["highs"] = highs;
                j_data["lows"] = lows;
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

double interp1(const vector<double>& X, const vector<double>& Y, double xi) {
    if (X.size() != Y.size() || X.size() < 2) {
        throw std::invalid_argument("X and Y must have same size and at least 2 elements");
    }
    if (xi <= X[0]) return Y[0];
    if (xi >= X.back()) return Y.back();

    for (size_t i = 1; i < X.size(); ++i) {
        if (xi >= X[i - 1] && xi <= X[i]) {
            double p = (xi - X[i - 1]) / (X[i] - X[i - 1]);
            return (1.0 - p) * Y[i - 1] + p * Y[i];
        }
    }
    return 0.0;
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
                    if (opt.market_price > 0.01 && opt.volume > 100 && opt.bid_ask_spread <= 0.05 * opt.market_price) {
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
        int K0 = (int)round(stock_prices.back());
        int K_min = (int)round(K0*0.6/10)*10;
        int K_max = (int)round(K0*1.4/10)*10;
        vector<double> strikes;
        for (double strike = K_min; strike < K_max; strike+=10) {
            strikes.push_back(strike);
        }
        for (const auto& expiry : expiries) {
            for (double K : strikes) {
                for (const auto& type : {"call", "put"}) {
                    string contract_type = (type == "call" ? "call" : "put");
                    string expiry_date = "20" + expiry.substr(0, 2) + "-" + expiry.substr(2, 2) + "-" + expiry.substr(4, 2);
                    string url = "https://api.polygon.io/v3/snapshot/options/" + symbol + "?strike_price=" + to_string((int)K) +
                                 "&expiration_date=" + expiry_date + "&contract_type=" + contract_type +
                                 "&order=asc&limit=1&sort=ticker&apiKey=" + api_key;
                    cout << "Fetching " << type << " Option data for strike " << K << ", expiry " << expiry << " from: " << url << endl;
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
                        opt.bid_ask_spread = 0.0;
                        opt.is_call = result["details"]["contract_type"] == "call";
                        if (opt.market_price > 0.01 && opt.volume > 100 && opt.bid_ask_spread <= 0.05 * opt.market_price) {
                            options_data.push_back(opt);
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
        item["expiration_date"] = opt.T;
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

double BisecBSIV(double MktPrice, double S0, double K, double T, double r, double q, bool is_call, bool use_eu) {
    double y;
    double a = 0.0;
    double b = 2.0;
    int MaxIter = 1000;
    double Tol = 1e-6;
    Black_Scholes bs;
    double price_a, price_b;
    if (use_eu) {
        price_a = is_call ? bs.call(S0, K, T, r, q, a) : bs.put(S0, K, T, r, q, a);
        price_b = is_call ? bs.call(S0, K, T, r, q, b) : bs.put(S0, K, T, r, q, b);
    } else {
        price_a = is_call ? bs.AmericanCall(S0, K, T, r, q, a) : bs.AmericanPut(S0, K, T, r, q, a);
        price_b = is_call ? bs.AmericanCall(S0, K, T, r, q, b) : bs.AmericanPut(S0, K, T, r, q, b);
    }
    double lowCdif = MktPrice - price_a;
    double highCdif = MktPrice - price_b;
    if (lowCdif * highCdif > 0) {
        y = 0.2; // Fallback to reasonable volatility
    } else {
        double midP;
        for (int x = 1; x < MaxIter; ++x) {
            midP = (a + b) / 2;
            double price_mid;
            if (use_eu) {
                price_mid = is_call ? bs.call(S0, K, T, r, q, midP) : bs.put(S0, K, T, r, q, midP);
            } else {
                price_mid = is_call ? bs.AmericanCall(S0, K, T, r, q, midP) : bs.AmericanPut(S0, K, T, r, q, midP);
            }
            double midCdif = MktPrice - price_mid;
            if (abs(midCdif) < Tol) {
                break;
            } else {
                if (midCdif > 0) {
                    a = midP;
                } else {
                    b = midP;
                }
            }
        }
        y = midP;
    }
    return max(0.05, min(2.0, y)); // Bound IV
}

double EstimateVolatility(const vector<OptionData>& option_data, bool use_eu) {
    vector<double> implied_vols;
    for (const auto& option : option_data) {
        if (option.is_call && abs(option.K / option.S - 1.0) <= 0.05) {
            double sigma = BisecBSIV(option.market_price, option.S, option.K, option.T, 0.035, 0.0, true, use_eu);
            if (sigma > 0.0 && isfinite(sigma)) implied_vols.push_back(sigma);
        }
    }
    if (implied_vols.empty()) return 0.2;
    return computeMean(implied_vols);
}

double computeVega(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0 || S <= 0.0 || K <= 0.0) {
        return 0.0;
    }
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    boost::math::normal_distribution<> standard_normal(0.0, 1.0);
    double N_prime_d1 = boost::math::pdf(standard_normal, d1);
    double vega = S * sqrt(T) * N_prime_d1;
    return vega;
}

double BSC(double S, double K, double rf, double q, double sigma, double T, bool use_eu) {
    Black_Scholes bs;
    return use_eu ? bs.call(S, K, T, rf, q, sigma) : bs.AmericanCall(S, K, T, rf, q, sigma);
}

double BSP(double S, double K, double rf, double q, double sigma, double T, bool use_eu) {
    Black_Scholes bs;
    return use_eu ? bs.put(S, K, T, rf, q, sigma) : bs.AmericanPut(S, K, T, rf, q, sigma);
}

double VarianceSwap(const vector<double>& KCI, const vector<double>& CallVI,
                    const vector<double>& KPI, const vector<double>& PutVI,
                    double S, double T, double rf, double q, bool use_eu) {
    if (KCI.size() != CallVI.size() || KPI.size() != PutVI.size()) {
        throw std::invalid_argument("Strike and volatility arrays must have same size");
    }
    if (KCI.empty() || KPI.empty() || T <= 0) {
        throw std::invalid_argument("Invalid input: empty arrays or non-positive maturity");
    }

    double Sb = S; // ATM boundary point
    auto f = [Sb, T](double S) { return 2.0 / T * ((S - Sb) / Sb - log(S / Sb)); };

    vector<double> Temp(KCI.size()), CallWeight(KCI.size()), CallValue(KCI.size()), CallContrib(KCI.size());
    for (size_t i = 0; i < KCI.size() - 1; ++i) {
        Temp[i] = (f(KCI[i + 1]) - f(KCI[i])) / (KCI[i + 1] - KCI[i]);
        if (i == 0) CallWeight[0] = Temp[0];
        else CallWeight[i] = Temp[i] - Temp[i - 1];
        CallValue[i] = BSC(S, KCI[i], rf, q, CallVI[i], T, use_eu);
        CallContrib[i] = CallValue[i] * CallWeight[i];
    }
    double Pi1 = 0.0;
    for (double contrib : CallContrib) Pi1 += contrib;

    vector<double> KPI_rev = KPI;
    vector<double> PutVI_rev = PutVI;
    reverse(KPI_rev.begin(), KPI_rev.end());
    reverse(PutVI_rev.begin(), PutVI_rev.end());

    vector<double> Temp2(KPI_rev.size()), PutWeight(KPI_rev.size()), PutValue(KPI_rev.size()), PutContrib(KPI_rev.size());
    for (size_t i = 0; i < KPI_rev.size() - 1; ++i) {
        Temp2[i] = (f(KPI_rev[i + 1]) - f(KPI_rev[i])) / (KPI_rev[i + 1] - KPI_rev[i]);
        if (i == 0) PutWeight[0] = Temp2[0];
        else PutWeight[i] = Temp2[i] - Temp2[i - 1];
        PutValue[i] = BSP(S, KPI_rev[i], rf, q, PutVI_rev[i], T, use_eu);
        PutContrib[i] = PutValue[i] * PutWeight[i];
    }
    double Pi2 = 0.0;
    for (double contrib : PutContrib) Pi2 += contrib;

    double Pi_CP = Pi1 + Pi2;
    double forward = S * exp((rf - q) * T);
    double Kvar = 2.0 / T * (rf * T - (forward / Sb - 1.0) - log(Sb / S) + exp(rf * T) * Pi_CP);

    return Kvar;
}

double hestonObjective(const vector<double>& params, void* data) {
    const double feller_penalty = 100.0;
    const double boundary_penalty = 100.0;
    auto* opt_data = static_cast<HestonOptData*>(data);
    const auto& option_data = opt_data->option_data;
    double S0 = opt_data->S0;
    const auto& lb = opt_data->lb;
    const auto& ub = opt_data->ub;
    bool use_eu = opt_data->use_eu;

    double v0 = params[0], kappa = params[1], theta = params[2], sigma = params[3], rho = params[4];
    double error = 0.0;
    double total_weight = 0.0;

    bool bounds_violated = false;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i] < lb[i] || params[i] > ub[i]) {
            bounds_violated = true;
            break;
        }
    }

    double feller = 2.0 * kappa * theta - sigma * sigma;
    bool feller_violated = feller <= 0.001;

    if (bounds_violated || feller_violated) {
        double penalty = 0.0;
        if (feller_violated) {
            penalty += feller_penalty * (0.001 - feller) * (0.001 - feller);
        }
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i] < lb[i]) {
                penalty += boundary_penalty * (lb[i] - params[i]) * (lb[i] - params[i]);
            } else if (params[i] > ub[i]) {
                penalty += boundary_penalty * (params[i] - ub[i]) * (params[i] - ub[i]);
            }
        }
        return 1e50 + penalty;
    }

    try {
        const int Trap = 1;
        const double Lphi = 0.00001;
        const double Uphi = 50.0;
        const double dphi = 0.0001;
        const double r = 0.035;
        const double q = 0.0;

        for (const auto& option : option_data) {
            double model_price;
            HestonFourier heston(S0, option.K, v0, option.T, r, q, kappa, theta, sigma, rho, Lphi, Uphi, dphi, debug);
            double HestonC, HestonP;
            if (use_eu) {
                heston.HestonPrice("Both", Trap, HestonC, HestonP);
                model_price = option.is_call ? HestonC : HestonP;
            } else {
                heston.HestonAmericanPrice("Both", Trap, HestonC, HestonP);
                model_price = option.is_call ? HestonC : HestonP;
            }
            double vega = computeVega(S0, option.K, option.T, r, sqrt(v0));
            double weight = vega > 1e-6 ? vega : 1e-6;

            double diff = model_price - option.market_price;
            error += weight * diff * diff;
            total_weight += weight;
        }

        error = (total_weight > 1e-6) ? error / total_weight : error / option_data.size();
    } catch (const exception& e) {
        return 1e50;
    }

    return error;
}

vector<double> nelderMeadOptimize(function<double(const vector<double>&, void*)> obj_func, void* data, const vector<double>& x0, const NMSettings& settings, const vector<double>& lb, const vector<double>& ub) {
    const int N = settings.N;
    vector<SimplexVertex> simplex(N + 1);
    int num_iters = 0;

    simplex[0].params = x0;
    simplex[0].value = obj_func(simplex[0].params, data);
    for (int j = 1; j <= N; ++j) {
        simplex[j].params = x0;
        simplex[j].params[j - 1] += random_num(-0.01, 0.01);
        for (int i = 0; i < N; ++i) {
            simplex[j].params[i] = max(lb[i], min(ub[i], simplex[j].params[i]));
        }
        simplex[j].value = obj_func(simplex[j].params, data);
    }

    while (num_iters < settings.MaxIters) {
        sort(simplex.begin(), simplex.end(), [](const SimplexVertex& a, const SimplexVertex& b) {
            return a.value < b.value;
        });

        double f1 = simplex[0].value;
        double fn = simplex[N - 1].value;
        double fn1 = simplex[N].value;

        if (abs(f1 - fn1) < settings.Tolerance) {
            break;
        }

        vector<double> centroid(N, 0.0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                centroid[j] += simplex[i].params[j];
            }
        }
        for (int j = 0; j < N; ++j) {
            centroid[j] /= N;
        }

        vector<double> xr(N);
        for (int j = 0; j < N; ++j) {
            xr[j] = centroid[j] + settings.alpha * (centroid[j] - simplex[N].params[j]);
            xr[j] = max(lb[j], min(ub[j], xr[j]));
        }
        double fr = obj_func(xr, data);

        if (f1 <= fr && fr < fn) {
            simplex[N].params = xr;
            simplex[N].value = fr;
            num_iters++;
            continue;
        }

        if (fr < f1) {
            vector<double> xe(N);
            for (int j = 0; j < N; ++j) {
                xe[j] = centroid[j] + settings.gamma * (xr[j] - centroid[j]);
                xe[j] = max(lb[j], min(ub[j], xe[j]));
            }
            double fe = obj_func(xe, data);
            simplex[N].params = (fe < fr) ? xe : xr;
            simplex[N].value = (fe < fr) ? fe : fr;
            num_iters++;
            continue;
        }

        if (fn <= fr && fr < fn1) {
            vector<double> xoc(N);
            for (int j = 0; j < N; ++j) {
                xoc[j] = centroid[j] + settings.rho * (xr[j] - centroid[j]);
                xoc[j] = max(lb[j], min(ub[j], xoc[j]));
            }
            double foc = obj_func(xoc, data);
            if (foc <= fr) {
                simplex[N].params = xoc;
                simplex[N].value = foc;
                num_iters++;
                continue;
            }
        }

        if (fr >= fn1) {
            vector<double> xic(N);
            for (int j = 0; j < N; ++j) {
                xic[j] = centroid[j] - settings.rho * (centroid[j] - simplex[N].params[j]);
                xic[j] = max(lb[j], min(ub[j], xic[j]));
            }
            double fic = obj_func(xic, data);
            if (fic < fn1) {
                simplex[N].params = xic;
                simplex[N].value = fic;
                num_iters++;
                continue;
            }
        }

        for (int j = 1; j <= N; ++j) {
            for (int i = 0; i < N; ++i) {
                simplex[j].params[i] = simplex[0].params[i] + settings.sigma * (simplex[j].params[i] - simplex[0].params[i]);
                simplex[j].params[i] = max(lb[i], min(ub[i], simplex[j].params[i]));
            }
            simplex[j].value = obj_func(simplex[j].params, data);
        }
        num_iters++;
    }

    return simplex[0].params;
}

double HestonLVApprox(double dt, double dK, double S0, double K, double r, double T, const HestonParams& params, bool use_eu) {
    int Trap = 1;
    double Lphi = 0.00001;
    double Uphi = 50;
    double dphi = 0.0001;
    double q = 0.0;
    HestonFourier heston(S0, K, params.v0, T, r, q, params.kappa, params.theta, params.sigma, params.rho, Lphi, Uphi, dphi, debug);
    double HestonC, HestonP;
    if (use_eu) {
        heston.HestonPrice("Both", Trap, HestonC, HestonP);
    } else {
        heston.HestonAmericanPrice("Both", Trap, HestonC, HestonP);
    }
    HestonFourier hestonT(S0, K, params.v0, T+dt, r, q, params.kappa, params.theta, params.sigma, params.rho, Lphi, Uphi, dphi, debug);
    HestonFourier hestonT1(S0, K, params.v0, T-dt, r, q, params.kappa, params.theta, params.sigma, params.rho, Lphi, Uphi, dphi, debug);
    double CT, CT_, dum;
    if (use_eu) {
        hestonT.HestonPrice("Both", Trap, CT, dum);
        hestonT1.HestonPrice("Both", Trap, CT_, dum);
    } else {
        hestonT.HestonAmericanPrice("Both", Trap, CT, dum);
        hestonT1.HestonAmericanPrice("Both", Trap, CT_, dum);
    }
    double dCdT = (CT - CT_) / (2 * dt);
    HestonFourier hestonK(S0, K+dK, params.v0, T, r, q, params.kappa, params.theta, params.sigma, params.rho, Lphi, Uphi, dphi, debug);
    HestonFourier hestonK1(S0, K-dK, params.v0, T, r, q, params.kappa, params.theta, params.sigma, params.rho, Lphi, Uphi, dphi, debug);
    double CK, CK0, CK_;
    if (use_eu) {
        hestonK.HestonPrice("Both", Trap, CK, dum);
        hestonK1.HestonPrice("Both", Trap, CK_, dum);
    } else {
        hestonK.HestonAmericanPrice("Both", Trap, CK, dum);
        hestonK1.HestonAmericanPrice("Both", Trap, CK_, dum);
    }
    CK0 = HestonC;
    double dC2dK2 = (CK - 2.0*CK0 + CK_) / (dK*dK);
    double LocalVar = (abs(dC2dK2) > 1e-10) ? 2*dCdT / (K*K*dC2dK2) : 0.0;
    return LocalVar;
}

void compare_heston(const string& symbol, double S0, double r, const HestonParams& params, bool use_eu) {
    int K0 = (int)round(S0);
    int K_min = (int)round(K0*0.75);
    int K_max = (int)round(K0*1.25);
    vector<double> strikes;
    for (double strike = K_min; strike < K_max; strike+=(int)round(K0*0.025)) {
        strikes.push_back(strike);
    }
    double T = 0.25;
    int Trap = 1;
    double Lphi = 0.00001;
    double Uphi = 50;
    double dphi = 0.0001;
    double q = 0.0;
    for (double K : strikes) {
        HestonFourier heston(S0, K, params.v0, T, r, q, params.kappa, params.theta, params.sigma, params.rho, Lphi, Uphi, dphi, debug);
        double HestonC, HestonP;
        if (use_eu) {
            heston.HestonPrice("Both", Trap, HestonC, HestonP);
        } else {
            heston.HestonAmericanPrice("Both", Trap, HestonC, HestonP);
        }
        cout << "Heston Fourier Call = " << HestonC << endl;
        cout << "Heston Fourier Put = " << HestonP << endl;
        double HestonC_con, HestonP_con;
        heston.PriceConsol("Both", Trap, HestonC_con, HestonP_con);
        cout << "Heston Fourier Consol Call = " << HestonC_con << endl;
        cout << "Heston Fourier Consol Put = " << HestonP_con << endl;
    }
}

void generate_heston_surface_data(const string& symbol, double S0, double r, const HestonParams& params, bool use_eu) {
    int K0 = (int)round(S0);
    int K_min = (int)round(K0*0.75);
    int K_max = (int)round(K0*1.25);
    vector<double> strikes;
    for (double strike = K_min; strike < K_max; strike+=(int)round(K0*0.025)) {
        strikes.push_back(strike);
    }
    vector<double> maturities = {0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0};
    json j;
    j["symbol"] = symbol;
    j["S0"] = S0;
    j["r"] = r;
    j["params"] = {{"v0", params.v0}, {"kappa", params.kappa}, {"theta", params.theta}, {"sigma", params.sigma}, {"rho", params.rho}};
    j["data"] = json::array();
    int Trap = 1;
    double Lphi = 0.00001;
    double Uphi = 50;
    double dphi = 0.0001;
    double q = 0.0;
    double dt, dK;
    for (double T : maturities) {
        for (double K : strikes) {
            dt = 0.01 * T;
            dK = 0.005 * K;
            HestonFourier heston(S0, K, params.v0, T, r, q, params.kappa, params.theta, params.sigma, params.rho, Lphi, Uphi, dphi, debug);
            double HestonC, HestonP;
            if (use_eu) {
                heston.HestonPrice("Both", Trap, HestonC, HestonP);
            } else {
                heston.HestonAmericanPrice("Both", Trap, HestonC, HestonP);
            }
            double iv = BisecBSIV(HestonC, S0, K, T, r, q, true, use_eu);
            double LocalVar = HestonLVApprox(dt, dK, S0, K, r, T, params, use_eu);
            iv = max(0.0, min(1.0, iv));
            LocalVar = max(0.0, min(1.0, LocalVar));
            json item;
            item["strike"] = K;
            item["maturity"] = T;
            item["call_price"] = HestonC;
            item["put_price"] = HestonP;
            item["implied_vol"] = iv;
            item["local_vol"] = sqrt(LocalVar);
            j["data"].push_back(item);
        }
    }
    ofstream ofs(symbol + "_heston_surface_data.json");
    if (ofs.is_open()) {
        ofs << j.dump(4);
        ofs.close();
    }
}

void generate_comparison_data(const string& symbol, double S0, double r, double sigma, const HestonParams& heston_params, double T, int Num_Steps, int Num_Sims, bool use_eu) {
    int K0 = (int)round(S0);
    int K_min = (int)round(K0*0.8/10)*10;
    int K_max = (int)round(K0*1.2/10)*10;
    vector<double> strikes;
    for (double strike = K_min; strike < K_max; strike+=10) {
        strikes.push_back(strike);
    }
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

    // Allocate grids once for SIMD
    const int N_S = 101, N_t = 51;
    vector<double> V(N_S * N_t, 0.0), V_temp(N_S * N_t, 0.0), k_temp(N_S * N_t, 0.0), res(N_S * N_t, 0.0), res_prev(N_S * N_t, 0.0);

    for (double K : strikes) {
        cout << "Strike = " << K << endl;
        json call_item, put_item;
        call_item["strike"] = K;
        put_item["strike"] = K;
        auto start = std::chrono::high_resolution_clock::now();

        double bs_call, bs_put;
        if (use_eu) {
            bs_call = bs.call(S0, K, T, r, 0.0, sigma);
            bs_put = bs.put(S0, K, T, r, 0.0, sigma);
        } else {
            bs_call = bs.AmericanCall(S0, K, T, r, 0.0, sigma);
            bs_put = bs.AmericanPut(S0, K, T, r, 0.0, sigma);
        }
        call_item["bs_analytical"] = bs_call;
        put_item["bs_analytical"] = bs_put;
        cout << "Black-Scholes Call = " << bs_call << endl;
        cout << "Black-Scholes Put = " << bs_put << endl;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Function took " << duration.count()/1000000.0 << " seconds" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        double bs_mc_call, bs_mc_put;
        if (use_eu) {
            bs_mc_call = mc.option_price(S0, K, T, r, sigma, true, 500000);
            bs_mc_put = mc.option_price(S0, K, T, r, sigma, false, 500000);
        } else {
            bs_mc_call = mc.American_option_price(S0, K, T, r, sigma, true, 50000, 1000);
            bs_mc_put = mc.American_option_price(S0, K, T, r, sigma, false, 50000, 1000);
        }
        call_item["bs_mc"] = bs_mc_call;
        put_item["bs_mc"] = bs_mc_put;
        cout << "Black-Scholes Monte-Carlo Call = " << bs_mc_call << endl;
        cout << "Black-Scholes Monte-Carlo Put = " << bs_mc_put << endl;
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Function took " << duration.count()/1000000.0 << " seconds" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        double BlackScholesCall_FD, BlackScholesPut_FD;
        Black_Scholes_FD bsfd(S0, K, T, r, sigma, N_S, N_t, 5000, true);
        if (use_eu) {
            bsfd.solve();
        } else {
            bsfd.solve_american();
        }
        BlackScholesCall_FD = bsfd.get_option_price(S0);
        BlackScholesPut_FD = BlackScholesCall_FD - S0 + K * exp(-r * T);
        call_item["bs_fd"] = BlackScholesCall_FD;
        put_item["bs_fd"] = BlackScholesPut_FD;
        cout << "Black-Scholes Finite Difference Call = " << BlackScholesCall_FD << endl;
        cout << "Black-Scholes Finite Difference Put = " << BlackScholesPut_FD << endl;
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Function took " << duration.count()/1000000.0 << " seconds" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        double BlackScholesCall_FD_simd, BlackScholesPut_FD_simd;
        Black_Scholes_FD_simd bsfd_simd(S0, K, T, r, sigma, N_S, N_t, 5000, true, false, &V, &V_temp, &k_temp, &res, &res_prev);
        if (use_eu) {
            bsfd_simd.solve();
        } else {
            bsfd_simd.solve_american();
        }
        BlackScholesCall_FD_simd = bsfd_simd.get_option_price(S0);
        BlackScholesPut_FD_simd = BlackScholesCall_FD_simd - S0 + K * exp(-r * T);
        call_item["bs_fd_simd"] = BlackScholesCall_FD_simd;
        put_item["bs_fd_simd"] = BlackScholesPut_FD_simd;
        cout << "Black-Scholes Finite Difference SIMD Call = " << BlackScholesCall_FD_simd << endl;
        cout << "Black-Scholes Finite Difference SIMD Put = " << BlackScholesPut_FD_simd << endl;
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Function took " << duration.count()/1000000.0 << " seconds" << std::endl;

        cout << "************************************" << endl;

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
/*void generate_comparison_data(const string& symbol, double S0, double r, double sigma, const HestonParams& heston_params, double T, int Num_Steps, int Num_Sims, bool use_eu) {
    int K0 = (int)round(S0);
    int K_min = (int)round(K0*0.8/10)*10;
    int K_max = (int)round(K0*1.2/10)*10;
    vector<double> strikes;
    for (double strike = K_min; strike < K_max; strike+=10) {
        strikes.push_back(strike);
    }
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
        cout << "Strike = " << K << endl;
        json call_item, put_item;
        call_item["strike"] = K;
        put_item["strike"] = K;
        auto start = std::chrono::high_resolution_clock::now();

        double bs_call, bs_put;
        if (use_eu) {
            bs_call = bs.call(S0, K, T, r, 0.0, sigma);
            bs_put = bs.put(S0, K, T, r, 0.0, sigma);
        } else {
            bs_call = bs.AmericanCall(S0, K, T, r, 0.0, sigma);
            bs_put = bs.AmericanPut(S0, K, T, r, 0.0, sigma);
        }
        call_item["bs_analytical"] = bs_call;
        put_item["bs_analytical"] = bs_put;
        cout << "Black-Scholes Call = " << bs_call << endl;
        cout << "Black-Scholes Put = " << bs_put << endl;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Function took " << duration.count() << " microseconds" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        double bs_mc_call, bs_mc_put;
        if (use_eu) {
            bs_mc_call = mc.option_price(S0, K, T, r, sigma, true, 1000000);
            bs_mc_put = mc.option_price(S0, K, T, r, sigma, false, 1000000);
        } else {
            bs_mc_call = mc.American_option_price(S0, K, T, r, sigma, true, 50000, 1000);
            bs_mc_put = mc.American_option_price(S0, K, T, r, sigma, false, 50000, 1000);
        }
        call_item["bs_mc"] = bs_mc_call;
        put_item["bs_mc"] = bs_mc_put;
        cout << "Black-Scholes Monte-Carlo Call = " << bs_mc_call << endl;
        cout << "Black-Scholes Monte-Carlo Put = " << bs_mc_put << endl;
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Function took " << duration.count() << " microseconds" << std::endl;

        /* int Trap = 1;
        double Lphi = 0.00001;
        double Uphi = 50;
        double dphi = 0.0001;
        double q = 0.0;

        HestonFourier heston(S0, K, heston_params.v0, T, r, q, heston_params.kappa, heston_params.theta, heston_params.sigma, heston_params.rho, Lphi, Uphi, dphi, debug);
        double HestonC, HestonP;
        if (use_eu) {
            heston.HestonPrice("Both", Trap, HestonC, HestonP);
        } else {
            heston.HestonAmericanPrice("Both", Trap, HestonC, HestonP);
        }
        call_item["heston_fourier"] = HestonC;
        put_item["heston_fourier"] = HestonP;
        cout << "Heston Fourier Call = " << HestonC << endl;
        cout << "Heston Fourier Put = " << HestonP << endl;

        double Heston_MC_C, Heston_MC_P;
        if (use_eu) {
            Heston_MC_C = mc.Heston_option_price(S0, K, T, r, heston_params.v0, heston_params.kappa, heston_params.theta, heston_params.sigma, heston_params.rho, true, Num_Sims, Num_Steps);
            Heston_MC_P = mc.Heston_option_price(S0, K, T, r, heston_params.v0, heston_params.kappa, heston_params.theta, heston_params.sigma, heston_params.rho, false, Num_Sims, Num_Steps);
        } else {
            Heston_MC_C = mc.Heston_American_option_price(S0, K, T, r, heston_params.v0, heston_params.kappa, heston_params.theta, heston_params.sigma, heston_params.rho, true, Num_Sims, Num_Steps);
            Heston_MC_P = mc.Heston_American_option_price(S0, K, T, r, heston_params.v0, heston_params.kappa, heston_params.theta, heston_params.sigma, heston_params.rho, false, Num_Sims, Num_Steps);
        }
        call_item["heston_mc"] = Heston_MC_C;
        put_item["heston_mc"] = Heston_MC_P;
        cout << "Heston Monte-Carlo Call = " << Heston_MC_C << endl;
        cout << "Heston Monte-Carlo Put = " << Heston_MC_P << endl;
        */
/*
        start = std::chrono::high_resolution_clock::now();
        double BlackScholesCall_FD, BlackScholesPut_FD;
        Black_Scholes_FD bsfd(S0, K, T, r, sigma, 101, 51, 10000, true);
        if (use_eu) {
            bsfd.solve();
        } else {
            bsfd.solve_american();
        }
        BlackScholesCall_FD = bsfd.get_option_price(S0);
        BlackScholesPut_FD = BlackScholesCall_FD - S0 + K * exp(-r * T);
        call_item["bs_fd"] = BlackScholesCall_FD;
        put_item["bs_fd"] = BlackScholesPut_FD;
        cout << "Black-Scholes Finite Difference Call = " << BlackScholesCall_FD << endl;
        cout << "Black-Scholes Finite Difference Put = " << BlackScholesPut_FD << endl;
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Function took " << duration.count() << " microseconds" << std::endl;

        // Added SIMD finite difference call
        start = std::chrono::high_resolution_clock::now();
        double BlackScholesCall_FD_simd, BlackScholesPut_FD_simd;
        Black_Scholes_FD_simd bsfd_simd(S0, K, T, r, sigma, 101, 51, 10000, true);
        if (use_eu) {
            bsfd_simd.solve();
        } else {
            bsfd_simd.solve_american();
        }
        BlackScholesCall_FD_simd = bsfd_simd.get_option_price(S0);
        BlackScholesPut_FD_simd = BlackScholesCall_FD_simd - S0 + K * exp(-r * T);
        call_item["bs_fd_simd"] = BlackScholesCall_FD_simd;
        put_item["bs_fd_simd"] = BlackScholesPut_FD_simd;
        cout << "Black-Scholes Finite Difference SIMD Call = " << BlackScholesCall_FD_simd << endl;
        cout << "Black-Scholes Finite Difference SIMD Put = " << BlackScholesPut_FD_simd << endl;
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Function took " << duration.count() << " microseconds" << std::endl;

        cout << "************************************" << endl;

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
}*/

HestonParams HestonInitial(const vector<OptionData>& option_data, double S0, bool use_eu) {
    HestonParams params;
    map<double, vector<double>> otm_call_strikes, otm_call_ivs, otm_put_strikes, otm_put_ivs;
    vector<double> maturities;
    map<double, double> kvars;
    vector<double> tempivs_skew;

    for (const auto& option : option_data) {
        double iv = BisecBSIV(option.market_price, S0, option.K, option.T, 0.035, 0.0, option.is_call, use_eu);
        if (isfinite(iv) && iv > 0.05 && iv < 2.0) {
            if (option.is_call && option.K > option.S) {
                otm_call_strikes[option.T].push_back(option.K);
                otm_call_ivs[option.T].push_back(iv);
            } else if (!option.is_call && option.K < option.S) {
                otm_put_strikes[option.T].push_back(option.K);
                otm_put_ivs[option.T].push_back(iv);
            }

            if (option.T < 0.25 && abs(option.K / option.S - 1.0) > 0.1 && abs(option.K / option.S - 1.0) < 0.3) {
                tempivs_skew.push_back(((option.K > option.S ? -1.0 : 1.0)) * iv);
            }
            if (find(maturities.begin(), maturities.end(), option.T) == maturities.end()) {
                maturities.push_back(option.T);
            }
        }
    }

    // Compute variance swaps
    sort(maturities.begin(), maturities.end());
    for (double T : maturities) {
        if (otm_call_strikes[T].size() >= 2 && otm_put_strikes[T].size() >= 2) {
            vector<size_t> call_indices(otm_call_strikes[T].size());
            iota(call_indices.begin(), call_indices.end(), 0);
            sort(call_indices.begin(), call_indices.end(),
                 [&otm_call_strikes, T](size_t i, size_t j) { return otm_call_strikes[T][i] < otm_call_strikes[T][j]; });
            vector<double> sorted_call_strikes, sorted_call_ivs;
            for (size_t i : call_indices) {
                sorted_call_strikes.push_back(otm_call_strikes[T][i]);
                sorted_call_ivs.push_back(otm_call_ivs[T][i]);
            }

            vector<size_t> put_indices(otm_put_strikes[T].size());
            iota(put_indices.begin(), put_indices.end(), 0);
            sort(put_indices.begin(), put_indices.end(),
                 [&otm_put_strikes, T](size_t i, size_t j) { return otm_put_strikes[T][i] < otm_put_strikes[T][j]; });
            vector<double> sorted_put_strikes, sorted_put_ivs;
            for (size_t i : put_indices) {
                sorted_put_strikes.push_back(otm_put_strikes[T][i]);
                sorted_put_ivs.push_back(otm_put_ivs[T][i]);
            }

            double Kvar = VarianceSwap(sorted_call_strikes, sorted_call_ivs, sorted_put_strikes, sorted_put_ivs, S0, T, 0.035, 0.0, use_eu);
            if (isfinite(Kvar) && Kvar > 0.0) {
                kvars[T] = Kvar;
            }
        }
    }

    // Estimate v0
    if (kvars.size() > 0) {
        params.v0 = kvars.begin()->second; // Shortest maturity
        params.v0 = max(0.001, min(0.1, params.v0)); // Bound v0
    } else {
        params.v0 = 0.04; // Fallback
    }

    // Estimate theta
    double theta_sum = 0.0;
    int theta_count = 0;
    for (const auto& [T, Kvar] : kvars) {
        if (T > 0.25) {
            theta_sum += Kvar;
            theta_count++;
        }
    }
    params.theta = (theta_count > 0) ? theta_sum / theta_count : 0.04;
    params.theta = max(0.001, min(0.1, params.theta));

    // Estimate kappa
    if (kvars.size() >= 2) {
        double T1 = maturities[0];
        double T2 = maturities[maturities.size() - 1];
        double Kvar1 = kvars[T1];
        double Kvar2 = kvars[T2];
        if (T2 > T1 && abs(Kvar1 - Kvar2) > 1e-6) {
            params.kappa = -log(Kvar2 / Kvar1) / (T2 - T1);
            params.kappa = max(0.5, min(10.0, params.kappa));
        } else {
            params.kappa = 2.0;
        }
    } else {
        params.kappa = 3.0;
    }

    // Estimate sigma
    vector<double> all_ivs;
    for (const auto& [T, ivs] : otm_call_ivs) {
        all_ivs.insert(all_ivs.end(), ivs.begin(), ivs.end());
    }
    for (const auto& [T, ivs] : otm_put_ivs) {
        all_ivs.insert(all_ivs.end(), ivs.begin(), ivs.end());
    }
    double iv_mean = computeMean(all_ivs);
    params.sigma = all_ivs.empty() ? 0.3 : computeStdDev(all_ivs, iv_mean) * sqrt(2.0);
    params.sigma = max(0.1, min(1.0, params.sigma));

    // Estimate rho
    params.rho = tempivs_skew.size() ? computeMean(tempivs_skew) : -0.5;
    params.rho = max(-0.9, min(-0.1, params.rho));

    // Ensure Feller condition
    if (2.0 * params.kappa * params.theta < params.sigma * params.sigma) {
        params.sigma = sqrt(1.9 * params.kappa * params.theta); // Adjust sigma
        params.sigma = max(0.1, min(1.0, params.sigma));
    }

    return params;
}

HestonParams calibrateHestonParameters(const vector<OptionData>& option_data, double S0, bool recalibrate, bool use_eu) {
    string filename = "AMZN_teston_params.json";
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

    HestonParams init_params = HestonInitial(option_data, S0, use_eu);
    vector<double> x0 = {init_params.v0, init_params.kappa, init_params.theta, init_params.sigma, init_params.rho};

    NMSettings settings;
    settings.N = 5;
    settings.MaxIters = 1000;
    settings.Tolerance = 1e-7;
    settings.alpha = 1.0;
    settings.gamma = 2.0;
    settings.rho = 0.5;
    settings.sigma = 0.5;

    vector<double> lb = {0.001, 0.5, 0.001, 0.1, -0.9};
    vector<double> ub = {0.1, 10.0, 0.1, 1.0, -0.1};

    HestonOptData data;
    data.option_data = option_data;
    data.S0 = S0;
    data.lb = lb;
    data.ub = ub;
    data.use_eu = use_eu;

    vector<double> optimized_params = nelderMeadOptimize(hestonObjective, &data, x0, settings, lb, ub);

    HestonParams params;
    params.v0 = optimized_params[0];
    params.kappa = optimized_params[1];
    params.theta = optimized_params[2];
    params.sigma = optimized_params[3];
    params.rho = optimized_params[4];

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

}

double Estimate_Stock(const vector<double>& prices, const vector<double>& volumes,
                     const vector<double>& highs, const vector<double>& lows,
                     bool use_lstm, bool read_model, const string& symbol) {
    if (use_lstm && prices.size() >= 20) {
        for (size_t i = 0; i < prices.size(); ++i) {
            if (isnan(prices[i]) || isinf(prices[i]) ||
                isnan(volumes[i]) || isinf(volumes[i]) ||
                isnan(highs[i]) || isinf(highs[i]) ||
                isnan(lows[i]) || isinf(lows[i])) {
                cerr << "NaN or Inf detected in input data at index " << i << endl;
                return numeric_limits<double>::quiet_NaN();
            }
        }

        // Compute moving average
        vector<double> ma(prices.size(), 0.0);
        for (size_t i = 4; i < prices.size(); ++i) {
            ma[i] = (prices[i] + prices[i-1] + prices[i-2] + prices[i-3] + prices[i-4]) / 5.0;
        }
        for (size_t i = 0; i < 4; ++i) {
            ma[i] = ma[4];
        }

        // Prepare lstm_data
        vector<vector<double>> lstm_data(prices.size(), vector<double>(5));
        vector<double> min_vals(5, numeric_limits<double>::max());
        vector<double> max_vals(5, numeric_limits<double>::lowest());
        for (size_t i = 0; i < prices.size(); ++i) {
            lstm_data[i][0] = prices[i];
            lstm_data[i][1] = log(volumes[i] + 1e-8);
            lstm_data[i][2] = ma[i];
            lstm_data[i][3] = highs[i] - lows[i];
            lstm_data[i][4] = (i >= 5) ? prices[i] - prices[i-5] : 0.0;
            for (int j = 0; j < 5; ++j) {
                min_vals[j] = min(min_vals[j], lstm_data[i][j]);
                max_vals[j] = max(max_vals[j], lstm_data[i][j]);
            }
        }

        // Normalize lstm_data
        vector<vector<double>> lstm_data_normalized = lstm_data;
        for (size_t i = 0; i < lstm_data.size(); ++i) {
            for (int j = 0; j < 5; ++j) {
                if (max_vals[j] != min_vals[j]) {
                    lstm_data_normalized[i][j] = (lstm_data[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j] + 1e-8);
                } else {
                    lstm_data_normalized[i][j] = 0.0;
                }
            }
        }

        // Store price_min and price_max for denormalization
        double price_min = min_vals[0];
        double price_max = max_vals[0];

        int seq_length = 20;
        LSTM lstm(5, 512, seq_length);
        string model_filename = symbol + "_lstm_model.json";

        if (read_model) {
            ifstream ifs(model_filename);
            if (ifs.is_open()) {
                json j;
                try {
                    ifs >> j;
                    lstm.load_model(j);
                    cout << "Loaded LSTM model from " << model_filename << endl;
                } catch (const json::exception& e) {
                    cerr << "JSON parse error in " << model_filename << ": " << e.what() << endl;
                    return numeric_limits<double>::quiet_NaN();
                } catch (const runtime_error& e) {
                    cerr << "Error loading model: " << e.what() << endl;
                    return numeric_limits<double>::quiet_NaN();
                }
                ifs.close();
            } else {
                cerr << "Failed to open " << model_filename << ", proceeding with training" << endl;
                read_model = false;
            }
        }

        if (!read_model) {
            lstm.train(lstm_data_normalized, 500);
            lstm.save_model(model_filename);
        }

        int train_size = static_cast<int>(0.8 * lstm_data.size());
        vector<vector<double>> test_data(lstm_data_normalized.begin() + train_size, lstm_data_normalized.end());
        vector<double> test_predictions(test_data.size(), 0.0);
        double mse = lstm.forward(test_data, test_predictions);
        if (isnan(mse)) {
            cerr << "LSTM forward pass returned NaN MSE" << endl;
            return numeric_limits<double>::quiet_NaN();
        }

        json j;
        j["stock_prices"] = prices;
        j["time_indices"] = vector<int>(prices.size());
        for (size_t i = 0; i < prices.size(); ++i) {
            j["time_indices"][i] = static_cast<int>(i);
        }
        j["predictions"] = json::array();
        for (size_t i = seq_length; i < test_data.size(); ++i) {
            json pred;
            pred["time"] = train_size + i;
            pred["actual"] = prices[train_size + i];
            pred["predicted"] = test_predictions[i] * (price_max - price_min + 1e-8) + price_min;
            j["predictions"].push_back(pred);
        }

        ofstream out("lstm_predictions.json");
        if (out.is_open()) {
            out << j.dump(4);
            out.close();
            cout << "Predictions saved to lstm_predictions.json" << endl;
        } else {
            cerr << "Error: Could not open lstm_predictions.json" << endl;
            return numeric_limits<double>::quiet_NaN();
        }

        vector<vector<double>> last_seq(lstm_data_normalized.end() - seq_length, lstm_data_normalized.end());
        double pred_norm = lstm.predict(last_seq);
        if (isnan(pred_norm)) {
            cerr << "LSTM prediction returned NaN" << endl;
            return numeric_limits<double>::quiet_NaN();
        }
        return pred_norm * (price_max - price_min + 1e-8) + price_min;
    } else {
        Linear_Regression lr(0.01, 1000);
        vector<double> x(prices.size());
        for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<double>(i);
        lr.train(x, prices);
        return lr.predict(static_cast<double>(x.size()));
    }
}

int main(int argc, char** argv) {
    string symbol = "AMZN";
    double r = 0.035;
    double T = 0.25;
    int Num_Steps = 500000;
    int Num_Sims = 1000;
    bool read_file = false;
    bool recalibrate = true;
    bool use_eu = false;
    bool use_lstm = false;
    bool read_model = false;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if ((arg == "--ticker" || arg == "-t") && i + 1 < argc) {
            symbol = argv[++i];
        } else if (arg == "--read-file" || arg == "-r") {
            read_file = true;
        } else if (arg == "--recalibrate" || arg == "-c") {
            recalibrate = true;
        } else if (arg == "--debug" || arg == "-d") {
            debug = true;
        } else if (arg == "-n") {
            Num_Steps = atoi(argv[++i]);
            Num_Sims = atoi(argv[++i]);
        } else if (arg == "--expiry" || arg == "-T") {
            T = atof(argv[++i]);
        } else if (arg == "--eu") {
            use_eu = true;
        } else if (arg == "--lstm") {
            use_lstm = true;
        } else if (arg == "--read-model") {
            read_model = true;
        }
    }
    cout << "N_Steps = " << Num_Steps << ", Num_Sims = " << Num_Sims << ", Use_European = " << (use_eu ? "True" : "False")
         << ", Use_LSTM = " << (use_lstm ? "True" : "False") << ", Read_Model = " << (read_model ? "True" : "False") << endl;

    vector<double> close, dayInd, volumes, highs, lows;
    vector<long long> timestamps;
    fetch_market_data(symbol, close, dayInd, timestamps, volumes, highs, lows, read_file);
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

    double sigma = EstimateVolatility(option_data, true);
    HestonParams heston_params;
    heston_params.v0 = 0.084873;
    heston_params.kappa = 6.237368;
    heston_params.theta = 0.067924;
    heston_params.sigma = 0.920504;
    heston_params.rho = -0.755989;
    cout << "Stock Price: " << S0 << endl;

    double estimated_price = Estimate_Stock(close, volumes, highs, lows, use_lstm, read_model, symbol);
    cout << "Estimated Stock Price (LSTM): " << estimated_price << endl;
    estimated_price = Estimate_Stock(close, volumes, highs, lows, false, false, symbol);
    cout << "Estimated Stock Price (Linear Regression): " << estimated_price << endl;

//cout << "Estimated Volatility: " << sigma << endl;
    //cout << "Heston Parameters: v0=" << heston_params.v0 << ", kappa=" << heston_params.kappa << ", theta=" << heston_params.theta << ", sigma=" << heston_params.sigma << ", rho=" << heston_params.rho << endl;
    generate_comparison_data(symbol, S0, r, sigma, heston_params, T, Num_Steps, Num_Sims, use_eu);
    return 0;
}