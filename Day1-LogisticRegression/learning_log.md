
# Day 1 - Logistic Regression

## Basics of Neural Networks 

Logistic regression is a learning algorithm used in a supervised learning problem when the output 𝑦 are
all either zero or one. The goal of logistic regression is to minimize the error between its predictions and
training data.

The parameters used in Logistic regression are:
- The input features vector: 𝑥 ∈ ℝ𝑛𝑥, where 𝑛𝑥 is the number of features
- The training label: 𝑦 ∈ 0,1
- The weights: 𝑤 ∈ ℝ𝑛𝑥, where 𝑛𝑥 is the number of features
- The threshold: 𝑏 ∈ ℝ
- The output: 𝑦̂ = 𝜎(𝑤𝑇𝑥 + 𝑏)
- Sigmoid function: s = 𝜎(𝑤𝑇𝑥 + 𝑏) = 𝜎(𝑧)= 1/(1+ 𝑒−𝑧)

*Cost Function*: J(w, b) = -1/m * Σ[i=1 to m] [y(i) * log(y_hat(i)) + (1 - y(i)) * log(1 - y_hat(i))]

### Notes: 

- For m iterations (training loops), each loop has a loss function. The average of the loss function is the cost function for complete training. 
-  "loss function" refers to the error function per sample, while "cost function" refers to the aggregated error over the entire dataset. However, this distinction varies depending on context and preference.
- Gradient Descent: Used to find the global minima of Cost function. The purpose being minimizing the cost function, to find the optimal set of parameters (weights and biases) that minimize the error across the dataset. 
