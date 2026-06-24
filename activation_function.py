import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

# ----------------------------------------------------------------------
# 1. Activation functions and their derivatives
# ----------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def leaky_relu_deriv(x, alpha=0.01):
    return np.where(x >= 0, 1.0, alpha)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def elu_deriv(x, alpha=1.0):
    return np.where(x >= 0, 1.0, elu(x, alpha) + alpha)

def selu(x):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def selu_deriv(x):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * np.where(x >= 0, 1.0, alpha * np.exp(x))

def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

def swish_deriv(x, beta=1.0):
    s = sigmoid(beta * x)
    return s + beta * x * s * (1 - s)

def gelu(x):
    # Approximate GELU used in practice
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def gelu_deriv(x):
    c = np.sqrt(2 / np.pi)
    term = x + 0.044715 * x**3
    tanh_term = np.tanh(c * term)
    sech2 = 1 - tanh_term**2
    d_term = 1 + 3 * 0.044715 * x**2
    return 0.5 * (1 + tanh_term) + 0.5 * x * sech2 * c * d_term

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_deriv(x):
    return sigmoid(x)

# ----------------------------------------------------------------------
# 2. Dictionary mapping names to (function, derivative, params)
#    params: dictionary of default parameter values (if any)
# ----------------------------------------------------------------------
activations = {
    'Sigmoid':    (sigmoid,      sigmoid_deriv,      {}),
    'Tanh':       (tanh,         tanh_deriv,         {}),
    'ReLU':       (relu,         relu_deriv,         {}),
    'Leaky ReLU': (leaky_relu,   leaky_relu_deriv,   {'alpha': 0.01}),
    'ELU':        (elu,          elu_deriv,          {'alpha': 1.0}),
    'SELU':       (selu,         selu_deriv,         {}),
    'Swish':      (swish,        swish_deriv,        {'beta': 1.0}),
    'GELU':       (gelu,         gelu_deriv,         {}),
    'Softplus':   (softplus,     softplus_deriv,     {})
}

# ----------------------------------------------------------------------
# 3. Set up the interactive figure
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(left=0.12, bottom=0.20, right=0.90)

x = np.linspace(-5, 5, 1000)

# Initial selection
current_name = 'ReLU'
func, deriv_func, params = activations[current_name]
y = func(x)
dy = deriv_func(x)

# ---- Left: activation ----
line_act, = ax1.plot(x, y, 'b-', lw=2, label='f(x)')
ax1.axhline(0, color='gray', lw=0.5)
ax1.axvline(0, color='gray', lw=0.5)
ax1.set_xlim(-5, 5)
ax1.set_ylim(-2, 4)
ax1.set_title('Activation Function')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# ---- Right: derivative ----
line_deriv, = ax2.plot(x, dy, 'r-', lw=2, label="f'(x)")
ax2.axhline(0, color='gray', lw=0.5)
ax2.axvline(0, color='gray', lw=0.5)
ax2.set_xlim(-5, 5)
ax2.set_ylim(-1, 2)
ax2.set_title('Derivative')
ax2.set_xlabel('x')
ax2.set_ylabel("f'(x)")
ax2.grid(True, alpha=0.3)
ax2.legend()

# ---- Radio buttons (left side) ----
ax_radio = plt.axes([0.03, 0.25, 0.12, 0.50])
radio = RadioButtons(ax_radio, list(activations.keys()), active=2)  # ReLU index 2

# ---- Slider (below the plots) ----
ax_slider = plt.axes([0.15, 0.05, 0.60, 0.03])
slider = Slider(ax_slider, 'Param', 0.001, 5.0, valinit=1.0, valstep=0.01)
slider.label.set_text('alpha / beta')
slider.set_val(1.0)
slider.set_active(False)   # hidden initially (no param for ReLU)

# Keep track of current parameter name
param_name = ''

# ----------------------------------------------------------------------
# 4. Update functions
# ----------------------------------------------------------------------
def update_activation(val=None):
    global current_name, param_name
    current_name = radio.value_selected
    func, deriv_func, params = activations[current_name]
    
    if params:
        param_name = list(params.keys())[0]
        slider.label.set_text(param_name)
        slider.set_val(params[param_name])
        slider.set_active(True)
    else:
        slider.set_active(False)
        slider.label.set_text('(no param)')
    
    # Recompute with current slider value
    param_value = slider.val if params else None
    if param_value is not None:
        y = func(x, **{param_name: param_value})
        dy = deriv_func(x, **{param_name: param_value})
    else:
        y = func(x)
        dy = deriv_func(x)
    
    # Update plot data
    line_act.set_ydata(y)
    line_deriv.set_ydata(dy)
    
    # Auto-scale y-limits
    y_min, y_max = np.min(y), np.max(y)
    y_margin = 0.2 * (y_max - y_min) if y_max != y_min else 0.5
    ax1.set_ylim(y_min - y_margin, y_max + y_margin)
    
    dy_min, dy_max = np.min(dy), np.max(dy)
    d_margin = 0.2 * (dy_max - dy_min) if dy_max != dy_min else 0.5
    ax2.set_ylim(dy_min - d_margin, dy_max + d_margin)
    
    fig.canvas.draw_idle()

def slider_update(val):
    # Called when slider is dragged
    func, deriv_func, params = activations[current_name]
    if params:
        param_value = slider.val
        y = func(x, **{param_name: param_value})
        dy = deriv_func(x, **{param_name: param_value})
        line_act.set_ydata(y)
        line_deriv.set_ydata(dy)
        # Update limits
        y_min, y_max = np.min(y), np.max(y)
        y_margin = 0.2 * (y_max - y_min) if y_max != y_min else 0.5
        ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        dy_min, dy_max = np.min(dy), np.max(dy)
        d_margin = 0.2 * (dy_max - dy_min) if dy_max != dy_min else 0.5
        ax2.set_ylim(dy_min - d_margin, dy_max + d_margin)
        fig.canvas.draw_idle()

# Connect events
radio.on_clicked(update_activation)
slider.on_changed(slider_update)

# Initial draw
update_activation()

# ----------------------------------------------------------------------
# 5. Show the interactive window
# ----------------------------------------------------------------------
plt.show()