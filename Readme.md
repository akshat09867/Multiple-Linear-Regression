# Implementation of Multiple Linear Regression

## Definition

Multiple Linear Regression is a statistical technique that models the linear relationship between a dependent variable and two or more independent variables. The goal is to find the best-fitting line (or hyperplane in higher dimensions) that minimizes the error between predicted and actual values.  This implementation uses Gradient Descent to find these optimal parameters.

---

## Libraries Used

- **NumPy**: For matrix operations and numerical computations, especially for vectorized calculations.
- **Pandas**: For reading, cleaning, and managing the dataset from a `.csv` file.
- **Matplotlib**: For clear and intuitive visualizations of the dataset and regression results, including cost function progress and predicted vs. actual values.
- **Scikit-learn**: Used for feature scaling with `StandardScaler`.

---

## Dataset

The code uses a CSV file named `Student_Performance.csv` (the path should be correctly provided).  It assumes the dataset contains features like 'Hours Studied', 'Previous Scores', 'Sample Question Papers Practiced', and 'Sleep Hours', with 'Performance Index' as the target variable.  You'll need to adapt the feature names if your dataset is different.

---

## Implementation

1. **Dataset Preparation**:
   - The dataset is loaded using Pandas.
   - Feature scaling is performed using `StandardScaler` from scikit-learn. This is crucial for gradient descent to converge efficiently when features have different scales.  The features are scaled before being used in the model.
   - The target variable ('Performance Index') is extracted.

2. **Model Equation**:
   Multiple Linear Regression is modeled as:
   Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε
   where:
- *Y*: Dependent variable (Performance Index).
- *X₁, X₂, ..., Xₙ*: Independent variables (Hours Studied, Previous Scores, etc.).
- *β₀*: Intercept.
- *β₁, β₂, ..., βₙ*: Coefficients for the independent variables.
- *ε*: Error term.

3. **Algorithm Steps**:

- **Initialization:** Weights (w) are initialized to zero, and the bias (b) is initialized to 0.
- **Cost Function:** The Mean Squared Error (MSE) is used as the cost function:

  ```
  J(w, b) = (1 / 2m) * Σ(h_w,b(xᵢ) - yᵢ)²
  ```

  where *m* is the number of training examples, and *h_w,b(xᵢ)* is the model's prediction for the i-th example.
- **Gradient Descent:** The algorithm iteratively updates the weights and bias to minimize the cost function:

  - **Compute Gradient:** The `compute_gradient` function calculates the partial derivatives of the cost function with respect to each weight and the bias.
  - **Update Parameters:** The weights and bias are updated as follows:

    ```
    wⱼ := wⱼ - α * ∂J(w, b) / ∂wⱼ
    b := b - α * ∂J(w, b) / ∂b
    ```

    where *α* is the learning rate.  This update is performed for a set number of `iterations`.
- **Iteration Tracking:** The cost is calculated and stored at each iteration (or every 100 iterations as in the code) to monitor the progress of gradient descent.

4. **Implementation in Code**:
- The `gradient_desent` function implements the gradient descent algorithm.
- The `cost_fun` function calculates the cost.
- The `compute_gradient` calculates the gradient.
- The `model` function makes predictions using the learned parameters.

5. **Visualization**:
 - **Cost vs. Iterations:** A plot of the cost function versus the number of iterations is generated to visualize the convergence of gradient descent.
 - **Predicted vs. Actual (for a single feature):** A scatter plot shows the actual 'Performance Index' values against 'Hours Studied'. The predicted values from the model are overlaid on this plot to visualize the model's fit.  Note that since this is multiple linear regression, plotting against just one feature gives a limited view of the model's overall performance.  Ideally, you'd need more sophisticated visualization techniques for higher dimensions.

---
