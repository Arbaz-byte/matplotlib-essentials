import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli

# Parameters
mu, sigma = 0, 1          # Gaussian mean and standard deviation
p = 0.3                   # Bernoulli probability of success

# --- Gaussian distribution ---
x_gauss = np.linspace(-4, 4, 500)
pdf_gauss = norm.pdf(x_gauss, mu, sigma)

# --- Bernoulli distribution ---
x_bern = [0, 1]
pmf_bern = bernoulli.pmf(x_bern, p)

# Create side‑by‑side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot Gaussian
ax1.plot(x_gauss, pdf_gauss, 'b-', lw=2)
ax1.fill_between(x_gauss, pdf_gauss, alpha=0.3)
ax1.set_title(f'Gaussian (μ={mu}, σ={sigma})')
ax1.set_xlabel('x')
ax1.set_ylabel('Probability Density')
ax1.grid(True, alpha=0.3)

# Plot Bernoulli
ax2.bar(x_bern, pmf_bern, color='orange', width=0.4)
ax2.set_xticks(x_bern)
ax2.set_xticklabels(['0 (Fail)', '1 (Success)'])
ax2.set_title(f'Bernoulli (p={p})')
ax2.set_xlabel('Outcome')
ax2.set_ylabel('Probability Mass')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
