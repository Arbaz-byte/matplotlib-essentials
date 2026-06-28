import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

# ----------------------------------------------------------------------
# 1. Loss functions and their derivatives
# ----------------------------------------------------------------------
def mse(y_true, y_pred):
    return (y_true - y_pred) ** 2

def mse_deriv(y_true, y_pred):
    return -2 * (y_true - y_pred)

def mae(y_true, y_pred):
    return np.abs(y_true - y_pred)

def mae_deriv(y_true, y_pred):
    return np.sign(y_pred - y_true)   # derivative w.r.t y_pred

def huber(y_true, y_pred, delta=1.0):
    diff = y_true - y_pred
    abs_diff = np.abs(diff)
    return np.where(abs_diff <= delta,
                    0.5 * diff**2,
                    delta * (abs_diff - 0.5 * delta))

def huber_deriv(y_true, y_pred, delta=1.0):
    diff = y_true - y_pred
    abs_diff = np.abs(diff)
    return np.where(abs_diff <= delta,
                    -diff,
                    -delta * np.sign(diff))

def binary_crossentropy(y_true, y_pred):
    # y_true in {0,1}, y_pred in (0,1)
    # Clamp to avoid log(0)
    eps = 1e-10
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_deriv(y_true, y_pred):
    eps = 1e-10
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)

def hinge(y_true, y_pred):
    # y_true in {-1, 1}, y_pred is the raw score
    return np.maximum(0, 1 - y_true * y_pred)

def hinge_deriv(y_true, y_pred):
    # derivative = 0 if margin >1, else -y_true
    return np.where(1 - y_true * y_pred > 0, -y_true, 0)

def log_cosh(y_true, y_pred):
    diff = y_true - y_pred
    return np.log(np.cosh(diff))

def log_cosh_deriv(y_true, y_pred):
    diff = y_true - y_pred
    return -np.tanh(diff)

# ----------------------------------------------------------------------
# 2. Loss dictionary
#    (function, derivative, param_info: list of (label, arg, default, min, max, step))
# ----------------------------------------------------------------------
losses = {
    'MSE': (mse, mse_deriv, []),
    'MAE': (mae, mae_deriv, []),
    'Huber': (huber, huber_deriv, [('δ (delta)', 'delta', 1.0, 0.1, 5.0, 0.1)]),
    'Binary Cross-Entropy': (binary_crossentropy, binary_crossentropy_deriv, []),
    'Hinge': (hinge, hinge_deriv, []),
    'Log-Cosh': (log_cosh, log_cosh_deriv, [])
}

# ----------------------------------------------------------------------
# 3. Set up the interactive figure
# ----------------------------------------------------------------------
fig, (ax_loss, ax_deriv) = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(left=0.12, bottom=0.20, right=0.90, top=0.90)

# Prediction axis (we will vary y_pred)
y_pred = np.linspace(-3, 3, 500)

# Initial selection
current_name = 'MSE'
y_true_init = 1.0
params = {}   # will hold current parameters (e.g., delta)

# ---- Loss plot ----
line_loss, = ax_loss.plot(y_pred, np.zeros_like(y_pred), 'b-', lw=2)
ax_loss.set_xlabel('Prediction (y_pred)')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('Loss Function')
ax_loss.axhline(0, color='gray', lw=0.5)
ax_loss.axvline(0, color='gray', lw=0.5)
ax_loss.grid(True, alpha=0.3)

# ---- Derivative plot ----
line_deriv, = ax_deriv.plot(y_pred, np.zeros_like(y_pred), 'r-', lw=2)
ax_deriv.set_xlabel('Prediction (y_pred)')
ax_deriv.set_ylabel("dLoss / d(y_pred)")
ax_deriv.set_title('Derivative w.r.t. Prediction')
ax_deriv.axhline(0, color='gray', lw=0.5)
ax_deriv.axvline(0, color='gray', lw=0.5)
ax_deriv.grid(True, alpha=0.3)

# ---- Radio buttons ----
ax_radio = plt.axes([0.03, 0.30, 0.12, 0.50])
radio = RadioButtons(ax_radio, list(losses.keys()), active=0)

# ---- Sliders ----
# Slider for y_true (common to all)
ax_slider_ytrue = plt.axes([0.20, 0.12, 0.55, 0.03])
slider_ytrue = Slider(ax_slider_ytrue, 'y_true', -2.0, 2.0, valinit=y_true_init, valstep=0.05)

# Sliders for additional parameters (will be created dynamically)
extra_sliders = {}
extra_slider_axes = []

def create_extra_sliders(param_info):
    """Create sliders for extra parameters (e.g., delta for Huber)."""
    # Clear previous extra sliders
    for ax in extra_slider_axes:
        ax.remove()
    extra_slider_axes.clear()
    extra_sliders.clear()
    # Position them just below y_true slider
    for i, (label, arg, default, min_val, max_val, step) in enumerate(param_info):
        y_pos = 0.07 - i * 0.04
        ax = plt.axes([0.20, y_pos, 0.55, 0.03])
        extra_slider_axes.append(ax)
        slider = Slider(ax, label, min_val, max_val, valinit=default, valstep=step)
        extra_sliders[arg] = slider
        slider.on_changed(update_plot)

def update_plot(val=None):
    """Redraw loss and derivative based on current selections."""
    # Get current y_true and parameters
    y_true = slider_ytrue.val
    # Build parameter dict for the selected loss
    func, deriv_func, param_info = losses[current_name]
    kwargs = {arg: extra_sliders[arg].val for arg in extra_sliders}
    # Compute loss and derivative
    loss_vals = func(y_true, y_pred, **kwargs)
    deriv_vals = deriv_func(y_true, y_pred, **kwargs)
    # Update curves
    line_loss.set_ydata(loss_vals)
    line_deriv.set_ydata(deriv_vals)
    # Auto-scale y limits (skip for binary cross-entropy near y_pred=0)
    y_min, y_max = np.min(loss_vals), np.max(loss_vals)
    margin = 0.1 * (y_max - y_min) if y_max != y_min else 0.5
    ax_loss.set_ylim(y_min - margin, y_max + margin)
    d_min, d_max = np.min(deriv_vals), np.max(deriv_vals)
    d_margin = 0.1 * (d_max - d_min) if d_max != d_min else 0.5
    ax_deriv.set_ylim(d_min - d_margin, d_max + d_margin)
    fig.canvas.draw_idle()

def radio_update(label):
    global current_name
    current_name = label
    # Recreate extra sliders if needed
    func, deriv_func, param_info = losses[current_name]
    create_extra_sliders(param_info)
    update_plot()

# Connect callbacks
radio.on_clicked(radio_update)
slider_ytrue.on_changed(update_plot)

# Initial setup
create_extra_sliders([])   # no extra sliders initially
update_plot()

# ----------------------------------------------------------------------
# 4. Show
# ----------------------------------------------------------------------
plt.show()