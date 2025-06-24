#ifndef LSTM_H
#define LSTM_H

#include <vector>
#include <random>
#include <nlohmann/json.hpp>

using namespace std;

using json = nlohmann::json;

class LSTM {
private:
    int input_size, hidden_size, seq_length;
    double learning_rate = 0.001;
    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    double weight_decay = 1e-4;
    double dropout_rate = 0.2;
    int t;

    // Weights and biases
    vector<vector<double>> Wf, Wi, Wc, Wo;
    vector<double> bf, bi, bc, bo;
    vector<double> Wd;
    double bd;

    // Optimizer state (Adam)
    vector<vector<double>> mWf, mWi, mWc, mWo;
    vector<vector<double>> vWf, vWi, vWc, vWo;
    vector<double> mbf, mbi, mbc, mbo;
    vector<double> vbf, vbi, vbc, vbo;
    vector<double> mWd, vWd;
    double mbd, vbd;

    // Data storage
    vector<vector<double>> X_all, X_train, X_test;
    double price_min, price_max;

    vector<double> min_vals;
    vector<double> max_vals;

    random_device rd;
    mt19937 gen;
    bernoulli_distribution dropout_dist;

    void init_weights();
    vector<double> sigmoid(const vector<double>& x);
    vector<double> sigmoid_deriv(const vector<double>& x);
    vector<double> tanh(const vector<double>& x);
    vector<double> tanh_deriv(const vector<double>& x);
    vector<double> elementwise_mult(const vector<double>& x, const vector<double>& y);
    vector<double> matrix_vector_mult(const vector<vector<double>>& W, const vector<double>& x);
    vector<double> vector_add(const vector<double>& x, const vector<double>& y);
    vector<double> apply_dropout(const vector<double>& x, double dropout_rate);
    void normalize_data(vector<vector<double>>& X);

public:
    LSTM(int input_size, int hidden_size, int seq_length);
    double forward(const vector<vector<double>>& X, vector<double>& predictions);
    void train(const vector<vector<double>>& X, int epochs);
    double predict(const vector<vector<double>>& X);
    void save_model(const string& filename);
    void load_model(const json& j);
    pair<double, double> get_price_normalization() const;
    vector<double> get_min_vals() const;
    vector<double> get_max_vals() const;
    void set_min_max_vals(const vector<double>& min_v, const vector<double>& max_v);
};

#endif