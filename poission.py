import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.stats import poisson, norm

# ------------------------------------------------------------
# 1. Set up the figure
# ------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(left=0.08, bottom=0.25, top=0.92, right=0.95)

# Create three subplots: PMF (top left), CDF (top right), histogram (bottom)
ax_pmf = plt.subplot2grid((2, 2), (0, 0))
ax_cdf = plt.subplot2grid((2, 2), (0, 1))
ax_hist = plt.subplot2grid((2, 2), (1, 0), colspan=2)

# Initial parameters
lambda_init = 5.0
n_samples_init = 500

# Generate initial data
np.random.seed(42)
x_max = int(lambda_init * 3) + 10   # range for PMF
x_vals = np.arange(0, x_max)
pmf_vals = poisson.pmf(x_vals, lambda_init)
cdf_vals = poisson.cdf(x_vals, lambda_init)
samples = poisson.rvs(lambda_init, size=n_samples_init)

# ---- PMF plot ----
pmf_bars = ax_pmf.bar(x_vals, pmf_vals, alpha=0.6, color='skyblue', label='PMF')
ax_pmf.set_xlabel('k')
ax_pmf.set_ylabel('P(X=k)')
ax_pmf.set_title('Probability Mass Function')
ax_pmf.grid(True, alpha=0.3)
# Normal approximation line (initially hidden)
x_norm = np.linspace(0, x_max, 200)
norm_pdf = norm.pdf(x_norm, lambda_init, np.sqrt(lambda_init))
norm_line, = ax_pmf.plot(x_norm, norm_pdf, 'r-', lw=2, label='Normal approx.', visible=False)
ax_pmf.legend()

# ---- CDF plot ----
cdf_step = ax_cdf.step(x_vals, cdf_vals, where='post', color='green', lw=2, label='CDF')
ax_cdf.set_xlabel('k')
ax_cdf.set_ylabel('F(k)')
ax_cdf.set_title('Cumulative Distribution Function')
ax_cdf.grid(True, alpha=0.3)
ax_cdf.legend()

# ---- Histogram plot ----
hist_counts, bin_edges, _ = ax_hist.hist(samples, bins=range(0, x_max+1), 
                                          density=True, alpha=0.5, color='lightcoral', label='Samples')
# Overlay theoretical PMF
ax_hist.plot(x_vals, pmf_vals, 'bo-', lw=2, label='Theoretical PMF')
ax_hist.set_xlabel('k')
ax_hist.set_ylabel('Density')
ax_hist.set_title('Histogram of Samples')
ax_hist.grid(True, alpha=0.3)
ax_hist.legend()

# ------------------------------------------------------------
# 2. Widgets
# ------------------------------------------------------------
ax_slider_lambda = plt.axes([0.15, 0.12, 0.40, 0.03])
ax_slider_n = plt.axes([0.60, 0.12, 0.30, 0.03])
ax_check = plt.axes([0.15, 0.05, 0.15, 0.04])
ax_button = plt.axes([0.60, 0.05, 0.10, 0.04])

slider_lambda = Slider(ax_slider_lambda, 'λ', 0.1, 20.0, valinit=lambda_init, valstep=0.1)
slider_n = Slider(ax_slider_n, 'Sample size', 10, 1000, valinit=n_samples_init, valstep=10)

# Checkbox for normal approximation
check = CheckButtons(ax_check, ['Show normal approx.'], [False])
normal_visible = False

# Resample button
btn_resample = Button(ax_button, 'Resample')

# ------------------------------------------------------------
# 3. Update function
# ------------------------------------------------------------
def update_plots(val=None):
    # Get current parameters
    lam = slider_lambda.val
    n = int(slider_n.val)
    
    # Update PMF and CDF
    x_max = int(lam * 3) + 10
    x_vals = np.arange(0, x_max)
    pmf_vals = poisson.pmf(x_vals, lam)
    cdf_vals = poisson.cdf(x_vals, lam)
    
    # Update PMF bars
    ax_pmf.clear()
    ax_pmf.bar(x_vals, pmf_vals, alpha=0.6, color='skyblue')
    # Redraw normal approximation if visible
    if normal_visible:
        x_norm = np.linspace(0, x_max, 200)
        norm_pdf = norm.pdf(x_norm, lam, np.sqrt(lam))
        ax_pmf.plot(x_norm, norm_pdf, 'r-', lw=2, label='Normal approx.')
    ax_pmf.set_xlabel('k')
    ax_pmf.set_ylabel('P(X=k)')
    ax_pmf.set_title('Probability Mass Function')
    ax_pmf.grid(True, alpha=0.3)
    ax_pmf.legend()
    
    # Update CDF
    ax_cdf.clear()
    ax_cdf.step(x_vals, cdf_vals, where='post', color='green', lw=2)
    ax_cdf.set_xlabel('k')
    ax_cdf.set_ylabel('F(k)')
    ax_cdf.set_title('Cumulative Distribution Function')
    ax_cdf.grid(True, alpha=0.3)
    
    # Resample new data (only if button not pressed, we will call separately)
    # Here we just update the theoretical curves; samples are updated only on button or slider_n change?
    # We'll regenerate samples on any update to reflect current n and lam
    samples = poisson.rvs(lam, size=n)
    # Update histogram
    ax_hist.clear()
    ax_hist.hist(samples, bins=range(0, x_max+1), density=True, alpha=0.5, color='lightcoral')
    ax_hist.plot(x_vals, pmf_vals, 'bo-', lw=2, label='Theoretical PMF')
    ax_hist.set_xlabel('k')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('Histogram of Samples')
    ax_hist.grid(True, alpha=0.3)
    ax_hist.legend()
    
    fig.canvas.draw_idle()

def resample_data(event):
    # Force new samples with current parameters
    update_plots(None)

def toggle_normal(label):
    global normal_visible
    normal_visible = not normal_visible
    update_plots(None)

# Connect widgets
slider_lambda.on_changed(update_plots)
slider_n.on_changed(update_plots)
check.on_clicked(toggle_normal)
btn_resample.on_clicked(resample_data)

# ------------------------------------------------------------
# 4. Initial draw
# ------------------------------------------------------------
update_plots()

plt.show()