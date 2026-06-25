import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
from scipy.stats import (
    norm, bernoulli, binom, poisson,
    uniform, expon, gamma, beta, laplace
)

# ----------------------------------------------------------------------
# 1. Define distributions with proper parameter mapping
#    Each entry: (dist_object, params_info, is_discrete)
#    params_info: list of (display_label, arg_name, default, min_val, max_val, step)
#    step = None for continuous, 1 for discrete integers
# ----------------------------------------------------------------------
distributions = {
    'Normal': (
        norm,
        [
            ('μ (mean)', 'loc', 0.0, -5.0, 5.0, None),
            ('σ (std)', 'scale', 1.0, 0.01, 5.0, None)
        ],
        False
    ),
    'Bernoulli': (
        bernoulli,
        [('p', 'p', 0.5, 0.0, 1.0, 0.01)],
        True
    ),
    'Binomial': (
        binom,
        [
            ('n (trials)', 'n', 10, 1, 50, 1),
            ('p', 'p', 0.5, 0.0, 1.0, 0.01)
        ],
        True
    ),
    'Poisson': (
        poisson,
        [('λ (rate)', 'mu', 3.0, 0.1, 20.0, 0.1)],
        True
    ),
    'Uniform': (
        uniform,
        [
            ('min', 'loc', -2.0, -5.0, 5.0, 0.1),
            ('max', 'scale', 4.0, 0.1, 10.0, 0.1)
        ],
        False
    ),
    'Exponential': (
        expon,
        [('scale (1/λ)', 'scale', 1.0, 0.01, 10.0, 0.01)],
        False
    ),
    'Gamma': (
        gamma,
        [
            ('shape (a)', 'a', 2.0, 0.1, 10.0, 0.1),
            ('scale (θ)', 'scale', 1.0, 0.01, 5.0, 0.01)
        ],
        False
    ),
    'Beta': (
        beta,
        [
            ('α', 'a', 2.0, 0.1, 10.0, 0.1),
            ('β', 'b', 5.0, 0.1, 10.0, 0.1)
        ],
        False
    ),
    'Laplace': (
        laplace,
        [
            ('location', 'loc', 0.0, -5.0, 5.0, 0.1),
            ('scale', 'scale', 1.0, 0.01, 5.0, 0.01)
        ],
        False
    )
}

# ----------------------------------------------------------------------
# 2. Set up the interactive figure
# ----------------------------------------------------------------------
fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(left=0.12, bottom=0.25, right=0.90, top=0.90)

# Global variables
current_name = 'Normal'
sliders = {}        # arg_name -> Slider object
slider_axes = []    # list of axes (to remove on update)

# Helper to get parameter dictionary from current sliders
def get_current_params():
    return {arg: sliders[arg].val for arg in sliders}

# Helper to get x range for given distribution and parameters
def get_plot_data(dist_obj, params, is_discrete, num_points=1000):
    if is_discrete:
        # Use support or heuristic
        try:
            low, high = dist_obj.support(**params)
            # For binomial, high might be n, we include all integers
            x = np.arange(int(low), int(high) + 1)
        except:
            # Fallback based on common names
            if 'n' in params:
                n = int(params['n'])
                x = np.arange(0, n + 1)
            elif 'mu' in params:
                mu = params['mu']
                x = np.arange(0, int(mu * 3) + 10)
            else:
                x = np.arange(0, 20)
        return x
    else:
        # Continuous: use quantiles to get a range covering 99.9% mass
        try:
            low = dist_obj.ppf(0.001, **params)
            high = dist_obj.ppf(0.999, **params)
        except:
            low = -5.0
            high = 5.0
        if not np.isfinite(low):
            low = -10.0
        if not np.isfinite(high):
            high = 10.0
        return np.linspace(low, high, num_points)

# ----------------------------------------------------------------------
# 3. Update functions (must be defined before creating sliders)
# ----------------------------------------------------------------------
def update_plot():
    """Redraw PDF/PMF and CDF with current parameter values."""
    # Get current parameters
    current_params = get_current_params()
    # Get distribution info
    dist_obj, params_info, is_discrete = distributions[current_name]
    # Generate x range
    x_new = get_plot_data(dist_obj, current_params, is_discrete)
    # Compute PDF/PMF and CDF
    if is_discrete:
        y_pdf = dist_obj.pmf(x_new, **current_params)
        y_cdf = dist_obj.cdf(x_new, **current_params)
    else:
        y_pdf = dist_obj.pdf(x_new, **current_params)
        y_cdf = dist_obj.cdf(x_new, **current_params)
    # Clear and redraw
    ax_pdf.clear()
    ax_cdf.clear()
    if is_discrete:
        ax_pdf.stem(x_new, y_pdf, basefmt=' ', markerfmt='bo', linefmt='b-')
        ax_cdf.step(x_new, y_cdf, where='post', color='r', lw=2)
    else:
        ax_pdf.plot(x_new, y_pdf, 'b-', lw=2)
        ax_cdf.plot(x_new, y_cdf, 'r-', lw=2)
    # Styling
    ax_pdf.axhline(0, color='gray', lw=0.5)
    ax_pdf.set_title('Probability Density / Mass Function')
    ax_pdf.set_xlabel('x')
    ax_pdf.set_ylabel('f(x) or P(X=x)')
    ax_pdf.grid(True, alpha=0.3)
    ax_cdf.axhline(0, color='gray', lw=0.5)
    ax_cdf.axhline(1, color='gray', lw=0.5, linestyle='--', alpha=0.5)
    ax_cdf.set_title('Cumulative Distribution Function')
    ax_cdf.set_xlabel('x')
    ax_cdf.set_ylabel('F(x)')
    ax_cdf.grid(True, alpha=0.3)
    fig.canvas.draw_idle()

def slider_update(val):
    """Callback for any slider change."""
    update_plot()

def radio_update(label):
    """Callback when a different distribution is selected."""
    global current_name, params_info, is_discrete
    current_name = label
    dist_obj, params_info, is_discrete = distributions[current_name]
    # Recreate sliders
    create_sliders(params_info)
    # Force update
    update_plot()

# ----------------------------------------------------------------------
# 4. Slider creation (now after slider_update is defined)
# ----------------------------------------------------------------------
def create_sliders(info_list):
    global slider_axes, sliders
    # Remove previous sliders
    for ax in slider_axes:
        ax.remove()
    slider_axes.clear()
    sliders.clear()
    # Create new sliders
    for i, (label, arg_name, default, min_val, max_val, step) in enumerate(info_list):
        y_pos = 0.12 - i * 0.05
        ax_slider = plt.axes([0.20, y_pos, 0.60, 0.03])
        slider_axes.append(ax_slider)
        # Create slider with optional step
        slider = Slider(ax_slider, label, min_val, max_val,
                        valinit=default, valstep=step)
        sliders[arg_name] = slider
        slider.on_changed(slider_update)

# ----------------------------------------------------------------------
# 5. Initialize UI
# ----------------------------------------------------------------------
# Radio buttons
ax_radio = plt.axes([0.03, 0.30, 0.12, 0.50])
radio = RadioButtons(ax_radio, list(distributions.keys()), active=0)
radio.on_clicked(radio_update)

# Create initial sliders for 'Normal'
dist_obj, params_info, is_discrete = distributions[current_name]
create_sliders(params_info)

# Initial plot
update_plot()

# ----------------------------------------------------------------------
# 6. Show the interactive window
# ----------------------------------------------------------------------
plt.show()