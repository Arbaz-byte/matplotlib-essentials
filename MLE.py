import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import norm

# ------------------------------------------------------------
# 1. Set up the interactive figure
# ------------------------------------------------------------
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=0.10, bottom=0.25, top=0.95, right=0.95)

# Grid: data (left), likelihood (right)
ax_data = plt.subplot2grid((1, 2), (0, 0))
ax_lik = plt.subplot2grid((1, 2), (0, 1))

# Initial parameters
true_mu_init = 2.0
sigma_init = 1.0
n_samples_init = 20

# Generate initial data
np.random.seed(42)
data = np.random.normal(true_mu_init, sigma_init, n_samples_init)

# ---- Data plot ----
ax_data.hist(data, bins=15, density=True, alpha=0.6, color='skyblue', edgecolor='black')
# Overlay true distribution
x_range = np.linspace(true_mu_init - 4*sigma_init, true_mu_init + 4*sigma_init, 200)
true_pdf = norm.pdf(x_range, true_mu_init, sigma_init)
line_true_pdf, = ax_data.plot(x_range, true_pdf, 'r-', lw=2, label='True PDF')
ax_data.axvline(true_mu_init, color='r', linestyle='--', label=f'True μ = {true_mu_init:.2f}')
ax_data.set_xlabel('x')
ax_data.set_ylabel('Density')
ax_data.set_title('Data & True Distribution')
ax_data.legend()
ax_data.grid(True, alpha=0.3)

# ---- Likelihood plot ----
mu_vals = np.linspace(-2, 6, 200)   # range for mu
# Compute likelihood (product of normal densities) and log-likelihood
def compute_likelihood(mu, data, sigma):
    # product of norm.pdf for each data point
    return np.prod(norm.pdf(data, mu, sigma), axis=0) if len(data) > 0 else 1.0

def compute_log_likelihood(mu, data, sigma):
    return np.sum(norm.logpdf(data, mu, sigma), axis=0)

# Initialize with current data
like_vals = np.array([compute_likelihood(mu, data, sigma_init) for mu in mu_vals])
log_like_vals = np.array([compute_log_likelihood(mu, data, sigma_init) for mu in mu_vals])
# Normalize likelihood for plotting (optional)
like_vals = like_vals / np.max(like_vals)  # scale to [0,1]

line_lik, = ax_lik.plot(mu_vals, like_vals, 'b-', lw=2, label='Likelihood (scaled)')
line_loglik, = ax_lik.plot(mu_vals, log_like_vals, 'g-', lw=2, label='Log-Likelihood')
# Mark true value
ax_lik.axvline(true_mu_init, color='r', linestyle='--', label=f'True μ = {true_mu_init:.2f}')
# Mark MLE (sample mean)
mle_est = np.mean(data)
ax_lik.axvline(mle_est, color='k', linestyle='-', label=f'MLE = {mle_est:.2f}')
ax_lik.set_xlabel('μ')
ax_lik.set_ylabel('Likelihood / Log-Likelihood')
ax_lik.set_title('Likelihood Function')
ax_lik.legend()
ax_lik.grid(True, alpha=0.3)

# Also add a vertical line for MLE (will update)
mle_line = ax_lik.axvline(mle_est, color='k', linestyle='-', label='MLE')

# ------------------------------------------------------------
# 2. Sliders
# ------------------------------------------------------------
ax_slider_mu = plt.axes([0.15, 0.12, 0.25, 0.03])
ax_slider_sigma = plt.axes([0.50, 0.12, 0.25, 0.03])
ax_slider_n = plt.axes([0.15, 0.07, 0.25, 0.03])
# Button to regenerate data with same parameters
ax_button = plt.axes([0.50, 0.07, 0.12, 0.04])

slider_mu = Slider(ax_slider_mu, 'True μ', -2, 5, valinit=true_mu_init, valstep=0.05)
slider_sigma = Slider(ax_slider_sigma, 'σ (sigma)', 0.2, 3.0, valinit=sigma_init, valstep=0.05)
slider_n = Slider(ax_slider_n, 'Sample size', 5, 100, valinit=n_samples_init, valstep=1)

# ------------------------------------------------------------
# 3. Update function
# ------------------------------------------------------------
def update_plots(val=None):
    # Get current slider values
    true_mu = slider_mu.val
    sigma = slider_sigma.val
    n = int(slider_n.val)
    
    # Generate new data
    np.random.seed(None)   # random seed each time
    data = np.random.normal(true_mu, sigma, n)
    
    # Update data histogram
    ax_data.clear()
    ax_data.hist(data, bins=min(n, 20), density=True, alpha=0.6, color='skyblue', edgecolor='black')
    x_range = np.linspace(true_mu - 4*sigma, true_mu + 4*sigma, 200)
    true_pdf = norm.pdf(x_range, true_mu, sigma)
    ax_data.plot(x_range, true_pdf, 'r-', lw=2, label='True PDF')
    ax_data.axvline(true_mu, color='r', linestyle='--', label=f'True μ = {true_mu:.2f}')
    ax_data.set_xlabel('x')
    ax_data.set_ylabel('Density')
    ax_data.set_title('Data & True Distribution')
    ax_data.legend()
    ax_data.grid(True, alpha=0.3)
    
    # Recompute likelihood and log-likelihood
    mu_vals = np.linspace(true_mu - 4, true_mu + 4, 200) if sigma < 3 else np.linspace(-4, 8, 200)
    like_vals = np.array([compute_likelihood(mu, data, sigma) for mu in mu_vals])
    log_like_vals = np.array([compute_log_likelihood(mu, data, sigma) for mu in mu_vals])
    # Normalize likelihood for better visualization
    like_vals = like_vals / np.max(like_vals) if np.max(like_vals) > 0 else like_vals
    
    # Clear likelihood axis and replot
    ax_lik.clear()
    ax_lik.plot(mu_vals, like_vals, 'b-', lw=2, label='Likelihood (scaled)')
    ax_lik.plot(mu_vals, log_like_vals, 'g-', lw=2, label='Log-Likelihood')
    ax_lik.axvline(true_mu, color='r', linestyle='--', label=f'True μ = {true_mu:.2f}')
    mle_est = np.mean(data)
    ax_lik.axvline(mle_est, color='k', linestyle='-', label=f'MLE = {mle_est:.2f}')
    ax_lik.set_xlabel('μ')
    ax_lik.set_ylabel('Likelihood / Log-Likelihood')
    ax_lik.set_title('Likelihood Function')
    ax_lik.legend()
    ax_lik.grid(True, alpha=0.3)
    
    fig.canvas.draw_idle()

# ------------------------------------------------------------
# 4. Random regeneration button
# ------------------------------------------------------------
def randomize(event):
    # Resample data with current parameters (already done in update)
    update_plots()

button = plt.Button(ax_button, 'Resample')
button.on_clicked(randomize)

# Connect sliders
slider_mu.on_changed(update_plots)
slider_sigma.on_changed(update_plots)
slider_n.on_changed(update_plots)

# Initial plot
update_plots()

plt.show()