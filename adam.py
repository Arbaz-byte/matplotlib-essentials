import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1.  ADAM OPTIMIZER (from scratch)
# ============================================================
class Adam:
    """
    Implements the Adam optimization algorithm.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size (α).
    beta1 : float, default=0.9
        Exponential decay rate for the first moment estimate.
    beta2 : float, default=0.999
        Exponential decay rate for the second moment estimate.
    epsilon : float, default=1e-8
        Small constant to prevent division by zero.
    """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.m = None       # first moment
        self.v = None       # second moment
        self.t = 0          # time step

    def update(self, params, grads):
        """
        Perform one Adam update step.

        Parameters
        ----------
        params : np.ndarray
            Current parameter vector (1D or 2D).
        grads : np.ndarray
            Gradient of the objective w.r.t. params (same shape).

        Returns
        -------
        updated_params : np.ndarray
            New parameter vector after the update.
        """
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Parameter update
        updated_params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return updated_params


# ============================================================
# 2.  TEST FUNCTION AND ITS GRADIENT
# ============================================================
# Choose any differentiable 2D function. Here we use the Rosenbrock
# function (banana-shaped valley) to show the optimizer's behaviour.
def rosenbrock(x, y, a=1, b=100):
    """Rosenbrock function: f(x,y) = (a - x)^2 + b*(y - x^2)^2"""
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_grad(x, y, a=1, b=100):
    """Gradient of the Rosenbrock function."""
    grad_x = -2*(a - x) - 4*b*x*(y - x**2)
    grad_y = 2*b*(y - x**2)
    return np.array([grad_x, grad_y])

# Alternatively, you can use a simple quadratic: f(x,y)=x^2 + y^2
# def quadratic(x, y):
#     return x**2 + y**2
# def quadratic_grad(x, y):
#     return np.array([2*x, 2*y])


# ============================================================
# 3.  RUN ADAM AND RECORD THE OPTIMISATION PATH
# ============================================================
def run_adam(start_point, objective, gradient, num_steps, **adam_kwargs):
    """
    Run Adam from a given starting point for a fixed number of steps.

    Parameters
    ----------
    start_point : tuple (x0, y0)
        Initial parameters.
    objective : callable
        Function f(x, y) to minimise.
    gradient : callable
        Function returning the gradient (gx, gy) at (x, y).
    num_steps : int
        Number of optimisation steps.
    **adam_kwargs : keyword arguments for Adam constructor.

    Returns
    -------
    history : list of (x, y) points visited.
    """
    adam = Adam(**adam_kwargs)
    point = np.array(start_point, dtype=float)
    history = [point.copy()]

    for _ in range(num_steps):
        grad = gradient(point[0], point[1])
        point = adam.update(point, grad)
        history.append(point.copy())

    return np.array(history)


# ============================================================
# 4.  VISUALISATION
# ============================================================
def plot_optimisation(history, objective, title="Adam Optimisation Path",
                       bounds=(-2, 2), levels=50, cmap='viridis'):
    """
    Plot the contour of the objective function and the path taken by Adam.
    """
    x_min, x_max = bounds
    y_min, y_max = bounds
    x_vals = np.linspace(x_min, x_max, 300)
    y_vals = np.linspace(y_min, y_max, 300)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = objective(X, Y)

    plt.figure(figsize=(8, 6))
    # Contour plot
    plt.contour(X, Y, Z, levels=levels, cmap=cmap, linewidths=0.5)
    # Optimisation path
    plt.plot(history[:, 0], history[:, 1], 'ro-', markersize=3, linewidth=1,
             label='Adam path')
    # Mark start and end
    plt.plot(history[0, 0], history[0, 1], 'go', markersize=8, label='Start')
    plt.plot(history[-1, 0], history[-1, 1], 'bs', markersize=8, label='End')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.colorbar(label='Objective value')
    plt.grid(alpha=0.3)
    plt.show()


# ============================================================
# 5.  MAIN: CHANGE PARAMETERS HERE AND SEE THE EFFECT
# ============================================================
if __name__ == "__main__":
    # -------- Set your hyperparameters here -----------------
    hyperparams = {
        'learning_rate': 0.002,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8
    }
    num_steps = 100
    start = (-1.5, 1.5)           # starting point for Rosenbrock
    # ---------------------------------------------------------

    # Run Adam
    history = run_adam(start, rosenbrock, rosenbrock_grad,
                       num_steps, **hyperparams)

    # Plot the result
    plot_optimisation(history, rosenbrock,
                      title=f"Adam: lr={hyperparams['learning_rate']}, "
                            f"β₁={hyperparams['beta1']}, β₂={hyperparams['beta2']}",
                      bounds=(-2, 2))

    # Optionally, print final value and last few steps
    print(f"Final point: {history[-1]}")
    print(f"Final objective: {rosenbrock(history[-1,0], history[-1,1]):.6f}")


"""Compare different learning rates
lrs = [0.001, 0.005, 0.01, 0.02]
for lr in lrs:
    h = run_adam(start, rosenbrock, rosenbrock_grad, num_steps,
                 learning_rate=lr, beta1=0.9, beta2=0.999)
    plot_optimisation(h, rosenbrock,
                      title=f"Adam: lr={lr}", bounds=(-2, 2))"""
