import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle

# ------------------------------------------------------------
# 1. Logistic map function
# ------------------------------------------------------------
def logistic(r, x):
    return r * x * (1 - x)

def iterate(r, x0, n):
    """Return n iterations starting from x0."""
    xs = np.zeros(n)
    xs[0] = x0
    for i in range(1, n):
        xs[i] = logistic(r, xs[i-1])
    return xs

# ------------------------------------------------------------
# 2. Bifurcation diagram (precomputed for speed)
# ------------------------------------------------------------
def compute_bifurcation(r_min=2.8, r_max=4.0, steps=400, trans=200, keep=200):
    """Return arrays r_vals and x_vals for bifurcation diagram."""
    r_vals = np.linspace(r_min, r_max, steps)
    x_vals = []
    for r in r_vals:
        x = 0.5  # arbitrary start
        # transient
        for _ in range(trans):
            x = logistic(r, x)
        # keep subsequent values
        for _ in range(keep):
            x = logistic(r, x)
            x_vals.append((r, x))
    return np.array(x_vals)

# Compute (takes a moment but fine)
print("Computing bifurcation diagram...")
bif_data = compute_bifurcation()
print("Done.")

# ------------------------------------------------------------
# 3. Set up the figure
# ------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(left=0.10, bottom=0.25, top=0.95, right=0.95)

# ---- Main bifurcation plot (background) ----
ax_bif = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax_bif.scatter(bif_data[:,0], bif_data[:,1], s=0.1, c='blue', alpha=0.5)
ax_bif.set_xlim(2.8, 4.0)
ax_bif.set_ylim(0, 1)
ax_bif.set_xlabel('r')
ax_bif.set_ylabel('x')
ax_bif.set_title('Bifurcation Diagram of the Logistic Map')
ax_bif.grid(True, alpha=0.3)

# ---- Time series plot (top right) ----
ax_time = plt.subplot2grid((2, 2), (0, 1))
ax_time.set_xlim(0, 100)
ax_time.set_ylim(0, 1)
ax_time.set_xlabel('Iteration n')
ax_time.set_ylabel('x_n')
ax_time.set_title('Time Series')
ax_time.grid(True, alpha=0.3)
line_time, = ax_time.plot([], [], 'r-', lw=1)

# ---- Cobweb plot (bottom right) ----
ax_cob = plt.subplot2grid((2, 2), (1, 1))
ax_cob.set_xlim(0, 1)
ax_cob.set_ylim(0, 1)
ax_cob.set_xlabel('x_n')
ax_cob.set_ylabel('x_{n+1}')
ax_cob.set_title('Cobweb Plot')
ax_cob.grid(True, alpha=0.3)
# Plot parabola and diagonal (static)
x_plot = np.linspace(0, 1, 200)
line_para, = ax_cob.plot(x_plot, x_plot, 'k--', lw=0.5)  # diagonal
line_curve, = ax_cob.plot([], [], 'b-', lw=1)             # parabola (will update)
line_cob, = ax_cob.plot([], [], 'r-', lw=0.8, alpha=0.7)  # cobweb steps

# ---- Slider for r ----
ax_slider_r = plt.axes([0.15, 0.12, 0.60, 0.03])
slider_r = Slider(ax_slider_r, 'r', 2.8, 4.0, valinit=3.9, valstep=0.001)

# ---- Text box to show current r and Feigenbaum info ----
ax_text = plt.axes([0.80, 0.10, 0.15, 0.10])
ax_text.axis('off')
text_info = ax_text.text(0, 0.5, '', fontsize=10, verticalalignment='center')

# ---- Button to compute Feigenbaum constant ----
ax_btn = plt.axes([0.50, 0.05, 0.12, 0.04])
btn_feig = Button(ax_btn, 'Estimate δ')

# ------------------------------------------------------------
# 4. Update functions
# ------------------------------------------------------------
def update_plots(val=None):
    r = slider_r.val
    # Time series (first 100 iterations)
    n_iters = 100
    x0 = 0.5
    xs = iterate(r, x0, n_iters)
    line_time.set_data(np.arange(n_iters), xs)
    ax_time.relim()
    ax_time.autoscale_view(scalex=False)

    # Cobweb plot: update parabola and cobweb
    # Parabola: y = r * x * (1 - x)
    y_curve = logistic(r, x_plot)
    line_curve.set_data(x_plot, y_curve)
    # Cobweb: draw the staircase
    steps = 50
    x_steps = np.zeros(2*steps)
    y_steps = np.zeros(2*steps)
    x = x0
    for i in range(steps):
        x_steps[2*i] = x
        y_steps[2*i] = logistic(r, x)
        x_steps[2*i+1] = y_steps[2*i]
        y_steps[2*i+1] = y_steps[2*i]
        x = y_steps[2*i]
    line_cob.set_data(x_steps, y_steps)
    ax_cob.relim()
    ax_cob.autoscale_view()

    # Update text
    text_info.set_text(f'r = {r:.3f}\n')
    fig.canvas.draw_idle()

# ------------------------------------------------------------
# 5. Feigenbaum constant estimation
# ------------------------------------------------------------
def estimate_feigenbaum():
    """Detect period-doubling bifurcation points and estimate δ."""
    # We'll scan r from 3.0 to 4.0 and look for period doublings
    # by analyzing the time series at each r.
    # Simple approach: find r where period doubles (bifurcation)
    r_vals = np.linspace(3.0, 4.0, 2000)
    bifurcations = []
    # We'll use a method: for each r, compute steady state and detect period.
    # We'll look at the last few iterates and see if they are periodic.
    # More robust: use the fact that the number of distinct values doubles.
    # We'll use a heuristic: look at the set of attractor points after transients.
    trans = 500
    keep = 200
    prev_period = 1
    for r in r_vals:
        x = 0.5
        # transient
        for _ in range(trans):
            x = logistic(r, x)
        # collect
        vals = []
        for _ in range(keep):
            x = logistic(r, x)
            vals.append(x)
        # estimate period by counting unique values (with tolerance)
        # Use clustering
        vals = np.array(vals)
        # Find unique values by rounding to 4 decimals
        unique = np.unique(np.round(vals, 4))
        period = len(unique)
        if period > prev_period:
            # bifurcation detected
            bifurcations.append(r)
            prev_period = period
    # Compute Feigenbaum delta from bifurcations (at least 3)
    if len(bifurcations) >= 3:
        # take last three
        r1, r2, r3 = bifurcations[-3], bifurcations[-2], bifurcations[-1]
        delta = (r2 - r1) / (r3 - r2)
    else:
        delta = np.nan
    # Display in text
    txt = f'Feigenbaum δ ≈ {delta:.4f}\n' if not np.isnan(delta) else 'Need more bifurcations.'
    # Also print
    print(txt)
    # Update text
    current_text = text_info.get_text()
    text_info.set_text(current_text + txt)
    fig.canvas.draw_idle()

# ------------------------------------------------------------
# 6. Connect callbacks and initial update
# ------------------------------------------------------------
slider_r.on_changed(update_plots)
btn_feig.on_clicked(estimate_feigenbaum)

# Initial update
update_plots()

# ------------------------------------------------------------
# 7. Show
# ------------------------------------------------------------
plt.show()