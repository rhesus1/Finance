#ifndef LSTM_H
#define LSTM_H

#include <vector>
#include <random>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

// Class for Long Short-Term Memory (LSTM) neural network
class LSTM {
private:
    // Model parameters
    int input_size;    // Size of input layer
    int hidden_size;   // Size of hidden layer
    int seq_length;    // Sequence length for input data
    double learning_rate = 0.001; // Learning rate for training
    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8; // Adam optimizer parameters
    double weight_decay = 1e-4; // Weight decay for regularization
    double dropout_rate = 0.2; // Dropout rate for regularization
    int t;             // Current time step

    // Weights and biases for LSTM gates
    vector<vector<double>> Wf, Wi, Wc, Wo; // Forget, input, cell, output gate weights
    vector<double> bf, bi, bc, bo;         // Forget, input, cell, output gate biases
    vector<double> Wd;                     // Dense layer weights
    double bd;                             // Dense layer bias

    // Adam optimizer state
    vector<vector<double>> mWf, mWi, mWc, mWo; // Momentum for gate weights
    vector<vector<double>> vWf, vWi, vWc, vWo; // Variance for gate weights
    vector<double> mbf, mbi, mbc, mbo;         // Momentum for gate biases
    vector<double> vbf, vbi, vbc, vbo;         // Variance for gate biases
    vector<double> mWd, vWd;                   // Momentum and variance for dense weights
    double mbd, vbd;                           // Momentum and variance for dense bias

    // Data storage
    vector<vector<double>> X_all, X_train, X_test; // All, training, and test data
    double price_min, price_max;                   // Min/max prices for normalization
    vector<double> min_vals, max_vals;             // Min/max values for each feature

    random_device rd;                              // Random device for dropout
    mt19937 gen;                                   // Random number generator
    bernoulli_distribution dropout_dist;           // Dropout distribution

    // Initialize weights and biases
    void init_weights();
    // Sigmoid activation function
    vector<double> sigmoid(const vector<double>& x);
    // Derivative of sigmoid
    vector<double> sigmoid_deriv(const vector<double>& x);
    // Tanh activation function
    vector<double> tanh(const vector<double>& x);
    // Derivative of tanh
    vector<double> tanh_deriv(const vector<double>& x);
    // Element-wise multiplication of vectors
    vector<double> elementwise_mult(const vector<double>& x, const vector<double>& y);
    // Matrix-vector multiplication
    vector<double> matrix_vector_mult(const vector<vector<double>>& W, const vector<double>& x);
    // Vector addition
    vector<double> vector_add(const vector<double>& x, const vector<double>& y);
    // Apply dropout to a vector
    vector<double> apply_dropout(const vector<double>& x, double dropout_rate);
    // Normalize input data
    void normalize_data(vector<vector<double>>& X);

public:
    // Constructor to initialize LSTM model
    LSTM(int input_size, int hidden_size, int seq_length);

    // Forward pass to compute predictions
    double forward(const vector<vector<double>>& X, vector<double>& predictions);

    // Train the model for given epochs
    void train(const vector<vector<double>>& X, int epochs);

    // Predict output for given input
    double predict(const vector<vector<double>>& X);

    // Save model to file
    void save_model(const string& filename);

    // Load model from JSON
    void load_model(const json& j);

    // Get price normalization parameters
    pair<double, double> get_price_normalization() const;

    // Get minimum feature values
    vector<double> get_min_vals() const;

    // Get maximum feature values
    vector<double> get_max_vals() const;

    // Set min/max feature values
    void set_min_max_vals(const vector<double>& min_v, const vector<double>& max_v);
};

#endif