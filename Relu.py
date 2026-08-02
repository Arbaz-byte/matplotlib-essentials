import numpy as np
import matplotlib.pyplot as plt

# Generate x values from -10 to 10
x = np.linspace(-10, 10, 1000)

# ReLU function: f(x) = max(0, x)
relu = np.maximum(0, x)

# Derivative of ReLU: f'(x) = 1 if x > 0 else 0
# (At x=0, the derivative is technically a subgradient; we set it to 0 for plotting)
relu_derivative = np.where(x > 0, 1, 0)

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ---- Plot ReLU function ----
ax1.plot(x, relu, color='blue', linewidth=2.5)
ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax1.axvline(0, color='black', linewidth=0.5, linestyle='--')
ax1.set_title('ReLU Activation Function', fontsize=14)
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-10, 10)
ax1.set_ylim(-1, 10)

# ---- Plot ReLU derivative ----
ax2.plot(x, relu_derivative, color='red', linewidth=2.5, drawstyle='steps-post')
ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax2.axvline(0, color='black', linewidth=0.5, linestyle='--')
ax2.set_title("Derivative of ReLU", fontsize=14)
ax2.set_xlabel('x')
ax2.set_ylabel("f'(x)")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-10, 10)
ax2.set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.show()
