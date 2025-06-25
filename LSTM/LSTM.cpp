#include "LSTM.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <random>
#include <nlohmann/json.hpp>
#include <fstream>
#include <limits>

using namespace std;
using json = nlohmann::json;

// Constructor initializes LSTM model with given dimensions and parameters
LSTM::LSTM(int input_size, int hidden_size, int seq_length)
    : input_size(input_size), hidden_size(hidden_size), seq_length(seq_length),
      gen(rd()), dropout_dist(1.0 - dropout_rate), t(0) {
    // Input: input_size (input dimension), hidden_size (LSTM units), seq_length (sequence length)
    // Output: None (initializes member variables)
    // Logic: Sets model dimensions, initializes random number generator for dropout,
    //        and calls init_weights to set up weight matrices and biases
    init_weights();
}

// Applies sigmoid activation function to a vector
vector<double> LSTM::sigmoid(const vector<double>& x) {
    // Input: Vector x of input values
    // Output: Vector with sigmoid function applied element-wise
    // Logic: Computes sigmoid(x) = 1/(1 + e^-x) for each element, mapping to (0,1).
    //        Handles potential overflow by capping exp(-x) at max double
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        double exp_neg_x = exp(-x[i]);
        if (isinf(exp_neg_x)) exp_neg_x = numeric_limits<double>::max();
        result[i] = 1.0 / (1.0 + exp_neg_x);
    }
    return result;
}

// Computes derivative of sigmoid function
vector<double> LSTM::sigmoid_deriv(const vector<double>& x) {
    // Input: Vector x of input values
    // Output: Vector of sigmoid derivative values
    // Logic: Computes derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    //        for each element, reusing sigmoid function for efficiency
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        double s = sigmoid({x[i]})[0];
        result[i] = s * (1.0 - s);
    }
    return result;
}

// Applies tanh activation function to a vector
vector<double> LSTM::tanh(const vector<double>& x) {
    // Input: Vector x of input values
    // Output: Vector with tanh function applied element-wise
    // Logic: Computes tanh(x) = (e^x - e^-x)/(e^x + e^-x) for each element,
    //        mapping to (-1,1). Handles overflow by returning sign of input
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        double ex = exp(x[i]);
        double emx = exp(-x[i]);
        if (isinf(ex) || isinf(emx)) {
            result[i] = (ex > emx) ? 1.0 : -1.0;
        } else {
            result[i] = (ex - emx) / (ex + emx);
        }
    }
    return result;
}

// Computes derivative of tanh function
vector<double> LSTM::tanh_deriv(const vector<double>& x) {
    // Input: Vector x of input values
    // Output: Vector of tanh derivative values
    // Logic: Computes derivative of tanh(x) = 1 - tanh(x)^2 for each element,
    //        reusing tanh function to compute base value
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        double t = tanh({x[i]})[0];
        result[i] = 1.0 - t * t;
    }
    return result;
}

// Performs element-wise multiplication of two vectors
vector<double> LSTM::elementwise_mult(const vector<double>& x, const vector<double>& y) {
    // Input: Two vectors x and y of same size
    // Output: Vector with element-wise products
    // Logic: Multiplies corresponding elements of x and y, assuming equal sizes
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] * y[i];
    }
    return result;
}

// Performs matrix-vector multiplication
vector<double> LSTM::matrix_vector_mult(const vector<vector<double>>& W, const vector<double>& x) {
    // Input: Matrix W (rows x cols) and vector x (cols)
    // Output: Vector of size rows, result of W * x
    // Logic: Computes dot product of each row of W with x to produce output vector
    vector<double> result(W.size(), 0.0);
    for (size_t i = 0; i < W.size(); ++i) {
        for (size_t j = 0; j < x.size(); ++j) {
            result[i] += W[i][j] * x[j];
        }
    }
    return result;
}

// Adds two vectors element-wise
vector<double> LSTM::vector_add(const vector<double>& x, const vector<double>& y) {
    // Input: Two vectors x and y of same size
    // Output: Vector with element-wise sums
    // Logic: Adds corresponding elements of x and y, assuming equal sizes
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] + y[i];
    }
    return result;
}

// Applies dropout to a vector
vector<double> LSTM::apply_dropout(const vector<double>& x, double dropout_rate) {
    // Input: Vector x and dropout rate (probability of dropping an element)
    // Output: Vector with dropout applied
    // Logic: Randomly sets elements to zero based on dropout_rate using Bernoulli distribution
    vector<double> result(x);
    if (dropout_rate == 0.0) return result;
    for (size_t i = 0; i < x.size(); ++i) {
        if (!dropout_dist(gen)) result[i] = 0.0;
    }
    return result;
}

// Initializes weights and biases with random values
void LSTM::init_weights() {
    // Input: None (uses class members input_size, hidden_size)
    // Output: None (initializes weight matrices and biases)
    // Logic: Initializes weights with normal distribution (std = 1/sqrt(input_size + hidden_size))
    //        and biases with specific values (forget gate bias set to -1.0 to encourage forgetting)
    normal_distribution<double> dist(0.0, sqrt(1.0 / (input_size + hidden_size)));
    Wf = vector<vector<double>>(hidden_size, vector<double>(input_size + hidden_size));
    Wi = Wf; Wc = Wf; Wo = Wf;
    bf = vector<double>(hidden_size);
    bi = bf; bc = bf; bo = bf;
    Wd = vector<double>(hidden_size);
    bd = 0.0;

    // Initialize optimizer moments for Adam
    mWf = vector<vector<double>>(hidden_size, vector<double>(input_size + hidden_size, 0.0));
    mWi = mWf; mWc = mWf; mWo = mWf;
    vWf = mWf; vWi = mWf; vWc = mWf; vWo = mWf;
    mbf = vector<double>(hidden_size, 0.0);
    mbi = mbf; mbc = mbf; mbo = mbf;
    vbf = vector<double>(hidden_size, 0.0);
    vbi = vbf; vbc = vbf; vbo = vbf;
    mWd = vector<double>(hidden_size, 0.0);
    vWd = mWd;
    mbd = 0.0; vbd = 0.0;

    // Assign random weights and biases
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_size + hidden_size; ++j) {
            Wf[i][j] = dist(gen); // Forget gate weights
            Wi[i][j] = dist(gen); // Input gate weights
            Wc[i][j] = dist(gen); // Cell state weights
            Wo[i][j] = dist(gen); // Output gate weights
        }
        bf[i] = -1.0; // Bias forget gate to encourage forgetting
        bi[i] = dist(gen); // Input gate bias
        bc[i] = dist(gen); // Cell state bias
        bo[i] = dist(gen); // Output gate bias
        Wd[i] = dist(gen); // Output layer weights
    }
}

// Performs forward pass and computes MSE loss
double LSTM::forward(const vector<vector<double>>& X, vector<double>& predictions) {
    // Input: Input data X (time series, size n x input_size), reference to predictions vector
    // Output: Mean squared error (MSE) of predictions
    // Logic: Processes sequences of seq_length timesteps through LSTM, computes predictions
    //        from final hidden state, calculates MSE, and includes diagnostic checks
    int n = X.size();
    predictions.resize(n, 0.0);
    double mse = 0.0;

    // Iterate over sequences starting after seq_length
    for (int i = seq_length; i < n; ++i) {
        vector<vector<double>> h_t(seq_length + 1, vector<double>(hidden_size, 0.0)); // Hidden states for sequence
        vector<vector<double>> C_t(seq_length + 1, vector<double>(hidden_size, 0.0)); // Cell states for sequence

        // Process each timestep in the sequence
        for (int t = 0; t < seq_length; ++t) {
            vector<double> x_t(input_size);
            for (int j = 0; j < input_size; ++j) {
                x_t[j] = X[i - seq_length + t][j]; // Extract input from sequence
            }

            // Concatenate previous hidden state and current input
            vector<double> concat_t(input_size + hidden_size);
            for (int j = 0; j < hidden_size; ++j) concat_t[j] = h_t[t][j];
            for (int j = 0; j < input_size; ++j) concat_t[hidden_size + j] = x_t[j];

            // Compute LSTM gates: forget, input, cell candidate, output
            vector<double> f_t = sigmoid(vector_add(matrix_vector_mult(Wf, concat_t), bf)); // Forget gate
            vector<double> i_t = sigmoid(vector_add(matrix_vector_mult(Wi, concat_t), bi)); // Input gate
            vector<double> c_tilde = tanh(vector_add(matrix_vector_mult(Wc, concat_t), bc)); // Cell candidate
            vector<double> o_t = sigmoid(vector_add(matrix_vector_mult(Wo, concat_t), bo)); // Output gate

            // Update cell state: C_t = f_t * C_{t-1} + i_t * c_tilde
            C_t[t + 1] = vector_add(elementwise_mult(f_t, C_t[t]), elementwise_mult(i_t, c_tilde));
            // Update hidden state with dropout: h_t = dropout(o_t * tanh(C_t))
            h_t[t + 1] = apply_dropout(elementwise_mult(o_t, tanh(C_t[t + 1])), dropout_rate);

            // Diagnostic output for first sequence at final timestep
            if (i == seq_length && t == seq_length - 1) {
                double f_mean = accumulate(f_t.begin(), f_t.end(), 0.0) / f_t.size(); // Mean of forget gate
                double i_mean = accumulate(i_t.begin(), i_t.end(), 0.0) / i_t.size(); // Mean of input gate
                //cout << "Forget gate mean: " << f_mean << "\n";
                //cout << "Input gate mean: " << i_mean << "\n";
                //cout << "Cell state (first 5): ";
                //for (int j = 0; j < min(5, hidden_size); ++j) cout << C_t[t + 1][j] << " ";
                //cout << endl;
            }
        }

        // Compute prediction from final hidden state
        double pred = bd;
        for (int j = 0; j < hidden_size; ++j) {
            pred += Wd[j] * h_t[seq_length][j];
        }
        predictions[i] = pred;
        double err = pred - X[i][0]; // Error against target (first feature)
        mse += err * err; // Accumulate squared error

        // Check for NaN or Inf in predictions
        if (isnan(pred) || isinf(pred)) {
            cerr << "NaN or Inf detected in predictions at i=" << i << endl;
            return numeric_limits<double>::quiet_NaN();
        }
    }

    // Compute average MSE
    double mse_result = mse / (n - seq_length);
    if (isnan(mse_result) || isinf(mse_result)) {
        // Check for NaN or Inf in MSE
        cerr << "NaN or Inf detected in MSE calculation" << endl;
        return numeric_limits<double>::quiet_NaN();
    }

    // Check for constant predictions
    if (predictions.size() > seq_length + 1) {
        bool all_same = true;
        for (size_t i = seq_length + 1; i < predictions.size(); ++i) {
            if (abs(predictions[i] - predictions[seq_length]) > 1e-6) {
                all_same = false;
                break;
            }
        }
        if (all_same) {
            cerr << "Warning: All predictions are constant (" << predictions[seq_length] << ")" << endl;
        }
    }

    return mse_result;
}

// Trains the LSTM model using Adam optimizer with early stopping
void LSTM::train(const vector<vector<double>>& X, int epochs) {
    // Input: Input data X (time series, size n x input_size), epochs (number of training iterations)
    // Output: None (updates model weights)
    // Logic: Splits data into train/test sets, performs forward and backward passes,
    //        updates weights using Adam optimizer, and applies early stopping based on test MSE
    int n = X.size();
    if (n <= seq_length) {
        // Validate dataset size
        cerr << "Error: Dataset size (" << n << ") is too small for seq_length (" << seq_length << ")" << endl;
        return;
    }

    // Split data into training (80%) and test (20%) sets
    int train_size = static_cast<int>(0.8 * n);
    int test_size = n - train_size;
    if (train_size <= seq_length) {
        // Validate training set size
        cerr << "Error: Training set size (" << train_size << ") is too small for seq_length (" << seq_length << ")" << endl;
        return;
    }

    X_all = X; // Store full dataset
    X_train = vector<vector<double>>(X_all.begin(), X_all.begin() + train_size); // Training data
    X_test = vector<vector<double>>(X_all.begin() + train_size, X_all.end()); // Test data

    double best_test_mse = numeric_limits<double>::max(); // Track best test MSE
    int patience = 20; // Early stopping patience
    int wait = 0; // Epochs since last improvement

    // Training loop
    for (int e = 0; e < epochs; ++e) {
        t++; // Increment time step for Adam bias correction
        vector<double> predictions;
        // Perform forward pass on training data
        double mse = forward(X_train, predictions);
        if (isnan(mse) || isinf(mse)) {
            // Stop if NaN or Inf detected
            cerr << "Training stopped due to NaN/Inf in Epoch " << e << endl;
            break;
        }
        //if (e % 10 == 0) {
            // Log training MSE every 10 epochs
            cout << "Epoch " << e << ", Training MSE: " << mse << endl;
        //}

        // Initialize gradients
        vector<vector<double>> dWf(hidden_size, vector<double>(input_size + hidden_size, 0.0));
        vector<vector<double>> dWi = dWf, dWc = dWf, dWo = dWf;
        vector<double> dbf(hidden_size, 0.0), dbi = dbf, dbc = dbf, dbo = dbf;
        vector<double> dWd(hidden_size, 0.0);
        double dbd = 0.0;

        // Backward pass for each training sequence
        for (int i = seq_length; i < train_size; ++i) {
            vector<vector<double>> h_t(seq_length + 1, vector<double>(hidden_size, 0.0)); // Hidden states
            vector<vector<double>> C_t(seq_length + 1, vector<double>(hidden_size, 0.0)); // Cell states
            vector<vector<double>> f_t(seq_length, vector<double>(hidden_size)); // Forget gates
            vector<vector<double>> i_t(seq_length, vector<double>(hidden_size)); // Input gates
            vector<vector<double>> c_tilde(seq_length, vector<double>(hidden_size)); // Cell candidates
            vector<vector<double>> o_t(seq_length, vector<double>(hidden_size)); // Output gates
            vector<vector<double>> concat(seq_length, vector<double>(input_size + hidden_size)); // Concatenated inputs

            // Forward pass to compute states and gates
            for (int t = 0; t < seq_length; ++t) {
                // Prepare input and concatenated vector
                for (int j = 0; j < input_size; ++j) {
                    concat[t][hidden_size + j] = X_train[i - seq_length + t][j];
                }
                for (int j = 0; j < hidden_size; ++j) {
                    concat[t][j] = h_t[t][j];
                }
                // Compute gates
                f_t[t] = sigmoid(vector_add(matrix_vector_mult(Wf, concat[t]), bf));
                i_t[t] = sigmoid(vector_add(matrix_vector_mult(Wi, concat[t]), bi));
                c_tilde[t] = tanh(vector_add(matrix_vector_mult(Wc, concat[t]), bc));
                o_t[t] = sigmoid(vector_add(matrix_vector_mult(Wo, concat[t]), bo));
                // Update cell and hidden states
                C_t[t + 1] = vector_add(elementwise_mult(f_t[t], C_t[t]), elementwise_mult(i_t[t], c_tilde[t]));
                h_t[t + 1] = apply_dropout(elementwise_mult(o_t[t], tanh(C_t[t + 1])), dropout_rate);
            }

            // Compute prediction and initial gradient
            double pred = bd;
            for (int j = 0; j < hidden_size; ++j) {
                pred += Wd[j] * h_t[seq_length][j];
            }
            double dpred = 2.0 * (pred - X_train[i][0]) / (train_size - seq_length); // Gradient of MSE

            // Initialize backpropagation gradients
            vector<double> dh_next(hidden_size, 0.0);
            vector<double> dC_next(hidden_size, 0.0);
            for (int j = 0; j < hidden_size; ++j) {
                dh_next[j] = dpred * Wd[j]; // Gradient through output weights
                dWd[j] += dpred * h_t[seq_length][j]; // Gradient for output weights
            }
            dbd += dpred; // Gradient for output bias

            // Backpropagate through time
            for (int t = seq_length - 1; t >= 0; --t) {
                // Compute gradients for output gate
                vector<double> do_t = elementwise_mult(dh_next, tanh(C_t[t + 1]));
                // Compute cell state gradient
                vector<double> dC_t = elementwise_mult(dh_next, o_t[t]);
                dC_t = elementwise_mult(dC_t, tanh_deriv(C_t[t + 1]));
                dC_t = vector_add(dC_t, dC_next);
                // Compute forget and input gate gradients
                vector<double> df_t = elementwise_mult(dC_t, C_t[t]);
                df_t = elementwise_mult(df_t, sigmoid_deriv(vector_add(matrix_vector_mult(Wf, concat[t]), bf)));
                vector<double> di_t = elementwise_mult(dC_t, c_tilde[t]);
                di_t = elementwise_mult(di_t, sigmoid_deriv(vector_add(matrix_vector_mult(Wi, concat[t]), bi)));
                // Compute cell candidate gradient
                vector<double> dc_tilde = elementwise_mult(dC_t, i_t[t]);
                dc_tilde = elementwise_mult(dc_tilde, tanh_deriv(vector_add(matrix_vector_mult(Wc, concat[t]), bc)));
                vector<double> dWo_z = sigmoid_deriv(vector_add(matrix_vector_mult(Wo, concat[t]), bo));

                // Clip gradients to prevent explosion
                const double clip_value = 1.0;
                for (int j = 0; j < hidden_size; ++j) {
                    dbf[j] += max(min(df_t[j], clip_value), -clip_value); // Accumulate forget bias gradient
                    dbi[j] += max(min(di_t[j], clip_value), -clip_value); // Accumulate input bias gradient
                    dbc[j] += max(min(dc_tilde[j], clip_value), -clip_value); // Accumulate cell bias gradient
                    dbo[j] += max(min(do_t[j], clip_value), -clip_value); // Accumulate output bias gradient
                    for (int k = 0; k < input_size + hidden_size; ++k) {
                        // Accumulate weight gradients
                        dWf[j][k] += max(min(df_t[j] * concat[t][k], clip_value), -clip_value);
                        dWi[j][k] += max(min(di_t[j] * concat[t][k], clip_value), -clip_value);
                        dWc[j][k] += max(min(dc_tilde[j] * concat[t][k], clip_value), -clip_value);
                        dWo[j][k] += max(min(do_t[j] * concat[t][k], clip_value), -clip_value);
                    }
                }

                // Compute next hidden state gradient
                dh_next = vector<double>(hidden_size, 0.0);
                for (int j = 0; j < hidden_size; ++j) {
                    for (int k = 0; k < input_size + hidden_size; ++k) {
                        dh_next[j] += Wf[j][k] * df_t[j] + Wi[j][k] * di_t[j] + Wc[j][k] * dc_tilde[j] + Wo[j][k] * do_t[j];
                    }
                    dh_next[j] = max(min(dh_next[j], clip_value), -clip_value); // Clip gradient
                }
                // Compute next cell state gradient
                dC_next = elementwise_mult(dC_t, f_t[t]);
                for (int j = 0; j < hidden_size; ++j) {
                    dC_next[j] = max(min(dC_next[j], clip_value), -clip_value); // Clip gradient
                }
            }
        }

        // Update weights using Adam optimizer
        const double clip_value = 1.0;
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < input_size + hidden_size; ++j) {
                // Update forget gate weights
                double grad = dWf[i][j] + weight_decay * Wf[i][j];
                grad = max(min(grad, clip_value), -clip_value);
                mWf[i][j] = beta1 * mWf[i][j] + (1 - beta1) * grad;
                vWf[i][j] = beta2 * vWf[i][j] + (1 - beta2) * grad * grad;
                double m_hat = mWf[i][j] / (1 - pow(beta1, t));
                double v_hat = vWf[i][j] / (1 - pow(beta2, t));
                Wf[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);

                // Update input gate weights
                grad = dWi[i][j] + weight_decay * Wi[i][j];
                grad = max(min(grad, clip_value), -clip_value);
                mWi[i][j] = beta1 * mWi[i][j] + (1 - beta1) * grad;
                vWi[i][j] = beta2 * vWi[i][j] + (1 - beta2) * grad * grad;
                m_hat = mWi[i][j] / (1 - pow(beta1, t));
                v_hat = vWi[i][j] / (1 - pow(beta2, t));
                Wi[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);

                // Update cell state weights
                grad = dWc[i][j] + weight_decay * Wc[i][j];
                grad = max(min(grad, clip_value), -clip_value);
                mWc[i][j] = beta1 * mWc[i][j] + (1 - beta1) * grad;
                vWc[i][j] = beta2 * vWc[i][j] + (1 - beta2) * grad * grad;
                m_hat = mWc[i][j] / (1 - pow(beta1, t));
                v_hat = vWc[i][j] / (1 - pow(beta2, t));
                Wc[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);

                // Update output gate weights
                grad = dWo[i][j] + weight_decay * Wo[i][j];
                grad = max(min(grad, clip_value), -clip_value);
                mWo[i][j] = beta1 * mWo[i][j] + (1 - beta1) * grad;
                vWo[i][j] = beta2 * vWo[i][j] + (1 - beta2) * grad * grad;
                m_hat = mWo[i][j] / (1 - pow(beta1, t));
                v_hat = vWo[i][j] / (1 - pow(beta2, t));
                Wo[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            }

            // Update forget gate bias
            double grad = dbf[i] + weight_decay * bf[i];
            grad = max(min(grad, clip_value), -clip_value);
            mbf[i] = beta1 * mbf[i] + (1 - beta1) * grad;
            vbf[i] = beta2 * vbf[i] + (1 - beta2) * grad * grad;
            double m_hat = mbf[i] / (1 - pow(beta1, t));
            double v_hat = vbf[i] / (1 - pow(beta2, t));
            bf[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);

            // Update input gate bias
            grad = dbi[i] + weight_decay * bi[i];
            grad = max(min(grad, clip_value), -clip_value);
            mbi[i] = beta1 * mbi[i] + (1 - beta1) * grad;
            vbi[i] = beta2 * vbi[i] + (1 - beta2) * grad * grad;
            m_hat = mbi[i] / (1 - pow(beta1, t));
            v_hat = vbi[i] / (1 - pow(beta2, t));
            bi[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);

            // Update cell state bias
            grad = dbc[i] + weight_decay * bc[i];
            grad = max(min(grad, clip_value), -clip_value);
            mbc[i] = beta1 * mbc[i] + (1 - beta1) * grad;
            vbc[i] = beta2 * vbc[i] + (1 - beta2) * grad * grad;
            m_hat = mbc[i] / (1 - pow(beta1, t));
            v_hat = vbc[i] / (1 - pow(beta2, t));
            bc[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);

            // Update output gate bias
            grad = dbo[i] + weight_decay * bo[i];
            grad = max(min(grad, clip_value), -clip_value);
            mbo[i] = beta1 * mbo[i] + (1 - beta1) * grad;
            vbo[i] = beta2 * vbo[i] + (1 - beta2) * grad * grad;
            m_hat = mbo[i] / (1 - pow(beta1, t));
            v_hat = vbo[i] / (1 - pow(beta2, t));
            bo[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);

            // Update output layer weights
            grad = dWd[i] + weight_decay * Wd[i];
            grad = max(min(grad, clip_value), -clip_value);
            mWd[i] = beta1 * mWd[i] + (1 - beta1) * grad;
            vWd[i] = beta2 * vWd[i] + (1 - beta2) * grad * grad;
            m_hat = mWd[i] / (1 - pow(beta1, t));
            v_hat = vWd[i] / (1 - pow(beta2, t));
            Wd[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }

        // Update output layer bias
        double grad = dbd + weight_decay * bd;
        grad = max(min(grad, clip_value), -clip_value);
        mbd = beta1 * mbd + (1 - beta1) * grad;
        vbd = beta2 * vbd + (1 - beta2) * grad * grad;
        double m_hat = mbd / (1 - pow(beta1, t));
        double v_hat = vbd / (1 - pow(beta2, t));
        bd -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);

        // Evaluate on test set
        vector<double> test_predictions;
        double test_mse = forward(X_test, test_predictions);
        if (isnan(test_mse) || isinf(test_mse)) {
            // Stop if NaN or Inf detected
            cerr << "Test MSE is NaN/Inf in Epoch " << e << endl;
            break;
        }
        //if (e % 10 == 0) {
            // Log test MSE every 10 epochs
            cout << "Epoch " << e << ", Test MSE: " << test_mse << endl;
        //}

        // Early stopping check
        if (test_mse < best_test_mse) {
            best_test_mse = test_mse; // Update best test MSE
            wait = 0; // Reset patience counter
            save_model("lstm_model_best.json"); // Save best model
        } else {
            wait++;
            if (wait >= patience) {
                // Stop training if no improvement for patience epochs
                cout << "Early stopping at epoch " << e << endl;
                break;
            }
        }
    }
}

// Makes a prediction for a single sequence
double LSTM::predict(const vector<vector<double>>& X) {
    // Input: Input sequence X (size at least seq_length x input_size)
    // Output: Predicted value (sigmoid-normalized)
    // Logic: Processes a single sequence through LSTM, computes prediction from final hidden state,
    //        and applies sigmoid to constrain output to [0,1]
    if (X.size() < static_cast<size_t>(seq_length)) {
        // Validate input sequence length
        cerr << "Error: Input sequence length (" << X.size() << ") is less than seq_length (" << seq_length << ")" << endl;
        return numeric_limits<double>::quiet_NaN();
    }

    vector<double> h_t(hidden_size, 0.0); // Initial hidden state
    vector<double> C_t(hidden_size, 0.0); // Initial cell state

    // Process each timestep in the sequence
    for (const auto& x : X) {
        vector<double> x_t(input_size);
        for (int j = 0; j < input_size; ++j) {
            x_t[j] = x[j]; // Extract input vector
        }

        // Concatenate hidden state and input
        vector<double> concat(input_size + hidden_size);
        for (int j = 0; j < hidden_size; ++j) concat[j] = h_t[j];
        for (int j = 0; j < input_size; ++j) concat[hidden_size + j] = x_t[j];

        // Compute LSTM gates
        vector<double> f_t = sigmoid(vector_add(matrix_vector_mult(Wf, concat), bf)); // Forget gate
        vector<double> i_t = sigmoid(vector_add(matrix_vector_mult(Wi, concat), bi)); // Input gate
        vector<double> c_tilde = tanh(vector_add(matrix_vector_mult(Wc, concat), bc)); // Cell candidate
        vector<double> o_t = sigmoid(vector_add(matrix_vector_mult(Wo, concat), bo)); // Output gate

        // Update cell and hidden states
        C_t = vector_add(elementwise_mult(f_t, C_t), elementwise_mult(i_t, c_tilde));
        h_t = elementwise_mult(o_t, tanh(C_t));
    }

    // Compute prediction from final hidden state
    double pred = bd;
    for (int j = 0; j < hidden_size; ++j) {
        pred += Wd[j] * h_t[j];
    }
    // Apply sigmoid to constrain output
    pred = 1.0 / (1.0 + exp(-pred));
    if (isnan(pred) || isinf(pred)) {
        // Check for NaN or Inf in prediction
        cerr << "NaN or Inf detected in prediction" << endl;
        return numeric_limits<double>::quiet_NaN();
    }

    // Log diagnostic information
    cout << "Prediction hidden state (first 5): ";
    for (int j = 0; j < min(5, hidden_size); ++j) {
        cout << h_t[j] << " ";
    }
    cout << "\nOutput weights (first 5): ";
    for (int j = 0; j < min(5, hidden_size); ++j) {
        cout << Wd[j] << " ";
    }
    cout << "\nBias: " << bd << "\nNormalized Prediction: " << pred << endl;

    return pred;
}

// Saves the model to a JSON file
void LSTM::save_model(const string& filename) {
    // Input: filename (path to save JSON file)
    // Output: None (saves model to file)
    // Logic: Serializes model parameters, weights, optimizer state, and data to JSON format
    json j;

    // Save model parameters
    j["model_data"]["input_size"] = input_size;
    j["model_data"]["hidden_size"] = hidden_size;
    j["model_data"]["seq_length"] = seq_length;
    j["model_data"]["learning_rate"] = learning_rate;
    j["model_data"]["beta1"] = beta1;
    j["model_data"]["beta2"] = beta2;
    j["model_data"]["epsilon"] = epsilon;
    j["model_data"]["weight_decay"] = weight_decay;

    // Save weights and biases
    j["weights"]["Wf"] = Wf;
    j["weights"]["Wi"] = Wi;
    j["weights"]["Wc"] = Wc;
    j["weights"]["Wo"] = Wo;
    j["weights"]["bf"] = bf;
    j["weights"]["bi"] = bi;
    j["weights"]["bc"] = bc;
    j["weights"]["bo"] = bo;
    j["weights"]["Wd"] = Wd;
    j["weights"]["bd"] = bd;

    // Save optimizer state
    j["optimizer"]["mWf"] = mWf;
    j["optimizer"]["mWi"] = mWi;
    j["optimizer"]["mWc"] = mWc;
    j["optimizer"]["mWo"] = mWo;
    j["optimizer"]["vWf"] = vWf;
    j["optimizer"]["vWi"] = vWi;
    j["optimizer"]["vWc"] = vWc;
    j["optimizer"]["vWo"] = vWo;
    j["optimizer"]["mbf"] = mbf;
    j["optimizer"]["mbi"] = mbi;
    j["optimizer"]["mbc"] = mbc;
    j["optimizer"]["mbo"] = mbo;
    j["optimizer"]["vbf"] = vbf;
    j["optimizer"]["vbi"] = vbi;
    j["optimizer"]["vbc"] = vbc;
    j["optimizer"]["vbo"] = vbo;
    j["optimizer"]["mWd"] = mWd;
    j["optimizer"]["vWd"] = vWd;
    j["optimizer"]["mbd"] = mbd;
    j["optimizer"]["vbd"] = vbd;
    j["optimizer"]["t"] = t;

    // Save data
    j["data"]["X_all"] = X_all;
    j["data"]["X_train"] = X_train;
    j["data"]["X_test"] = X_test;

    // Write JSON to file
    ofstream file(filename);
    if (file.is_open()) {
        file << j.dump(4);
        file.close();
        cout << "Model saved to " << filename << endl;
    } else {
        cerr << "Error: Could not open file " << filename << " for writing" << endl;
    }
}

// Loads the model from a JSON file
void LSTM::load_model(const json& j) {
    // Input: JSON object j containing model data
    // Output: None (restores model state)
    // Logic: Deserializes model parameters, weights, optimizer state, and data from JSON,
    //        with error handling for parsing and dimension mismatches
    try {
        // Load model parameters
        input_size = j["model_data"]["input_size"].get<int>();
        hidden_size = j["model_data"]["hidden_size"].get<int>();
        seq_length = j["model_data"]["seq_length"].get<int>();
        learning_rate = j["model_data"]["learning_rate"].get<double>();
        beta1 = j["model_data"]["beta1"].get<double>();
        beta2 = j["model_data"]["beta2"].get<double>();
        epsilon = j["model_data"]["epsilon"].get<double>();
        weight_decay = j["model_data"]["weight_decay"].get<double>();

        // Load weights and biases
        Wf = j["weights"]["Wf"].get<vector<vector<double>>>();
        Wi = j["weights"]["Wi"].get<vector<vector<double>>>();
        Wc = j["weights"]["Wc"].get<vector<vector<double>>>();
        Wo = j["weights"]["Wo"].get<vector<vector<double>>>();
        bf = j["weights"]["bf"].get<vector<double>>();
        bi = j["weights"]["bi"].get<vector<double>>();
        bc = j["weights"]["bc"].get<vector<double>>();
        bo = j["weights"]["bo"].get<vector<double>>();
        Wd = j["weights"]["Wd"].get<vector<double>>();
        bd = j["weights"]["bd"].get<double>();

        // Load optimizer state
        mWf = j["optimizer"]["mWf"].get<vector<vector<double>>>();
        mWi = j["optimizer"]["mWi"].get<vector<vector<double>>>();
        mWc = j["optimizer"]["mWc"].get<vector<vector<double>>>();
        mWo = j["optimizer"]["mWo"].get<vector<vector<double>>>();
        vWf = j["optimizer"]["vWf"].get<vector<vector<double>>>();
        vWi = j["optimizer"]["vWi"].get<vector<vector<double>>>();
        vWc = j["optimizer"]["vWc"].get<vector<vector<double>>>();
        vWo = j["optimizer"]["vWo"].get<vector<vector<double>>>();
        mbf = j["optimizer"]["mbf"].get<vector<double>>();
        mbi = j["optimizer"]["mbi"].get<vector<double>>();
        mbc = j["optimizer"]["mbc"].get<vector<double>>();
        mbo = j["optimizer"]["mbo"].get<vector<double>>();
        vbf = j["optimizer"]["vbf"].get<vector<double>>();
        vbi = j["optimizer"]["vbi"].get<vector<double>>();
        vbc = j["optimizer"]["vbc"].get<vector<double>>();
        vbo = j["optimizer"]["vbo"].get<vector<double>>();
        mWd = j["optimizer"]["mWd"].get<vector<double>>();
        vWd = j["optimizer"]["vWd"].get<vector<double>>();
        mbd = j["optimizer"]["mbd"].get<double>();
        vbd = j["optimizer"]["vbd"].get<double>();
        t = j["optimizer"]["t"].get<int>();

        // Load data
        X_all = j["data"]["X_all"].get<vector<vector<double>>>();
        X_train = j["data"]["X_train"].get<vector<vector<double>>>();
        X_test = j["data"]["X_test"].get<vector<vector<double>>>();

        // Validate weight dimensions
        if (Wf.size() != static_cast<size_t>(hidden_size) ||
            Wf[0].size() != static_cast<size_t>(input_size + hidden_size) ||
            Wd.size() != static_cast<size_t>(hidden_size)) {
            throw runtime_error("Loaded weight dimensions do not match model configuration");
        }

        // Log successful load
        cout << "Model loaded successfully with input_size=" << input_size
             << ", hidden_size=" << hidden_size << ", seq_length=" << seq_length << endl;
    } catch (const json::exception& e) {
        throw runtime_error("JSON parse error during model loading: " + string(e.what()));
    } catch (const exception& e) {
        throw runtime_error("Error during model loading: " + string(e.what()));
    }
}