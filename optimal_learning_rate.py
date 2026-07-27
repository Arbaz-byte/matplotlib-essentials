import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ------------------------------
# 1. Generate synthetic regression data
# ------------------------------
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for better convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Add bias term (intercept) – we'll treat it as an additional feature
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_val = np.c_[np.ones(X_val.shape[0]), X_val]

# ------------------------------
# 2. Define model and training function
# ------------------------------
def linear_regression_gd(X, y, lr, epochs=500):
    """
    Train a linear regression model using batch gradient descent.
    Returns: final weights and list of training losses.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    losses = []
    
    for epoch in range(epochs):
        # Predictions
        y_pred = X @ weights
        # Gradient of MSE loss: (2/n) * X^T (y_pred - y)
        gradient = (2 / n_samples) * X.T @ (y_pred - y)
        # Update weights
        weights -= lr * gradient
        # Store loss
        mse = mean_squared_error(y, y_pred)
        losses.append(mse)
    return weights, losses

# ------------------------------
# 3. Search for the best learning rate
# ------------------------------
def find_best_lr(X_train, y_train, X_val, y_val, lr_range, epochs=500):
    """
    Perform grid search over learning rates and return the best one
    based on validation loss.
    """
    best_lr = None
    best_val_loss = float('inf')
    val_losses = []
    
    for lr in lr_range:
        # Train on training set
        weights, _ = linear_regression_gd(X_train, y_train, lr, epochs)
        # Evaluate on validation set
        y_pred_val = X_val @ weights
        val_loss = mean_squared_error(y_val, y_pred_val)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lr = lr
    
    return best_lr, val_losses

# Define a logarithmic range of learning rates
lr_range = np.logspace(-6, 0, 20)   # from 1e-6 to 1.0

print("Searching over learning rates...")
best_lr, val_losses = find_best_lr(X_train, y_train, X_val, y_val, lr_range, epochs=300)

print(f"Best learning rate: {best_lr:.6f}")
print(f"Best validation MSE: {np.min(val_losses):.6f}")

# ------------------------------
# 4. Visualize the results
# ------------------------------
plt.figure(figsize=(10, 6))
plt.semilogx(lr_range, val_losses, marker='o', linestyle='-', color='b')
plt.axvline(best_lr, color='r', linestyle='--', label=f'Best LR = {best_lr:.6f}')
plt.xlabel('Learning Rate')
plt.ylabel('Validation MSE')
plt.title('Learning Rate vs Validation Performance')
plt.grid(True)
plt.legend()
plt.show()

# ------------------------------
# 5. (Optional) Train final model with the best LR
# ------------------------------
final_weights, final_losses = linear_regression_gd(
    np.vstack([X_train, X_val]), 
    np.hstack([y_train, y_val]), 
    lr=best_lr, 
    epochs=500
)

print(f"Final training loss (MSE) with best LR: {final_losses[-1]:.6f}")
