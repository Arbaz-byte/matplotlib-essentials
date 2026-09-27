"""
Relativistic Time-Dilation Explorer
-----------------------------------
Drag the sliders to change speed / proper time / distance and watch
all four panels update live.

Needs:  numpy, matplotlib
Run:    python time_dilation.py
In Jupyter: put  %matplotlib widget  (ipympl) or  %matplotlib tk  at the top.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
c = 299792458.0          # speed of light [m/s]
TAU_MUON = 2.2e-6        # muon proper mean lifetime [s]


def lorentz(beta):
    """gamma = 1/sqrt(1 - beta^2).  Works for scalars and arrays."""
    beta = np.clip(np.asarray(beta, float), 0.0, 1.0 - 1e-12)
    return 1.0 / np.sqrt(1.0 - beta ** 2)


# ----------------------------------------------------------------------
# Figure layout
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(13.5, 9))
try:
    fig.canvas.manager.set_window_title("Special-Relativity Time Dilation Lab")
except Exception:
    pass

gs = fig.add_gridspec(2, 2, left=0.075, right=0.97, top=0.90,
                      bottom=0.30, hspace=0.45, wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])   # gamma vs beta
ax2 = fig.add_subplot(gs[0, 1])   # clock comparison
ax3 = fig.add_subplot(gs[1, 0])   # muon survival
ax4 = fig.add_subplot(gs[1, 1])   # trip to a star

fig.suptitle("Time Dilation Explorer  —  drag the sliders", fontsize=15, y=0.965)

# ----------------------------------------------------------------------
# Panel 1 : Lorentz factor vs speed
# ----------------------------------------------------------------------
beta_grid = np.linspace(0.0, 0.99999, 1500)

ax1.plot(beta_grid, lorentz(beta_grid), color='C0', lw=2)
ax1.set_yscale('log')
ax1.set_xlim(0, 1.0)
ax1.set_ylim(1, 400)
ax1.set_xlabel(r'$\beta = v/c$')
ax1.set_ylabel(r'$\gamma$   (Lorentz factor)')
ax1.grid(True, which='both', alpha=0.3)
ax1.axhline(1.0, color='k', lw=0.6, alpha=0.4)
dot1, = ax1.plot([], [], 'o', color='red', ms=9, zorder=5)

# ----------------------------------------------------------------------
# Panel 2 : coordinate time vs proper time
# ----------------------------------------------------------------------
TAU_MAX = 50.0                                   # years on the x-axis
tau_axis = np.linspace(0, TAU_MAX, 300)

for b_ref in (0.5, 0.9, 0.99):                   # faint reference slopes
    ax2.plot(tau_axis, lorentz(b_ref) * tau_axis, color='grey', lw=0.9, alpha=0.45)
ax2.plot(tau_axis, tau_axis, 'k--', lw=1.0, alpha=0.6)   # no-dilation line

line2, = ax2.plot([], [], color='C1', lw=2.5, label='your speed')
pt2, = ax2.plot([], [], 'o', color='C1', ms=8)
ax2.set_xlim(0, TAU_MAX)
ax2.set_xlabel(r'traveler proper time  $\tau$  (years)')
ax2.set_ylabel(r'Earth coordinate time  $t$  (years)')
ax2.grid(alpha=0.3)

# ----------------------------------------------------------------------
# Panel 3 : cosmic-ray muon survival
# ----------------------------------------------------------------------
d_km = np.linspace(0, 20, 500)
d_m = d_km * 1000.0

line3_sr, = ax3.plot([], [], color='C2', lw=2.5, label='relativistic (time dilation)')
line3_cl, = ax3.plot([], [], color='C3', lw=1.5, ls='--', label='classical (no dilation)')
ax3.set_xlim(0, 20)
ax3.set_ylim(0, 1.02)
ax3.set_xlabel('distance travelled in the lab frame  (km)')
ax3.set_ylabel('fraction of muons surviving')
ax3.grid(alpha=0.3)
ax3.legend(loc='upper right', fontsize=8)

# ----------------------------------------------------------------------
# Panel 4 : trip to a star (redrawn every update)
# ----------------------------------------------------------------------
ax4.set_ylabel('elapsed time (years)')

# ----------------------------------------------------------------------
# Sliders
# ----------------------------------------------------------------------
ax_b = fig.add_axes([0.16, 0.185, 0.68, 0.028])
ax_t = fig.add_axes([0.16, 0.125, 0.68, 0.028])
ax_d = fig.add_axes([0.16, 0.065, 0.68, 0.028])

sl_beta = Slider(ax_b, r'speed   $\beta = v/c$', 0.01, 0.9999, valinit=0.90, valfmt='%.4f')
sl_tau = Slider(ax_t, r'proper time  $\tau$  (yr)', 0.1, TAU_MAX, valinit=10.0, valfmt='%.1f')
sl_dist = Slider(ax_d, 'trip distance  D  (light-years)', 0.1, 100.0, valinit=10.0, valfmt='%.1f')


# ----------------------------------------------------------------------
# Update function
# ----------------------------------------------------------------------
def update(_=None):
    b = float(sl_beta.val)
    tau = float(sl_tau.val)
    D = float(sl_dist.val)
    g = float(lorentz(b))

    # ---- 1 : Lorentz factor -----------------------------------------
    dot1.set_data([b], [g])
    ax1.set_title(f'1.  Lorentz factor   γ = {g:.4f}   (β = {b:.4f})', fontsize=10)

    # ---- 2 : clock comparison ---------------------------------------
    line2.set_data(tau_axis, g * tau_axis)
    ax2.set_ylim(0, max(g * TAU_MAX, TAU_MAX) * 1.05)
    pt2.set_data([tau], [g * tau])
    ax2.set_title(f'2.  traveler ages {tau:.1f} yr  →  Earth ages {g * tau:.2f} yr',
                  fontsize=10)

    # ---- 3 : muon survival ------------------------------------------
    t_lab = d_m / (b * c)                       # lab-frame travel time
    line3_sr.set_data(d_km, np.exp(-t_lab / (g * TAU_MUON)))   # dilated clock
    line3_cl.set_data(d_km, np.exp(-t_lab / TAU_MUON))         # Newtonian
    ax3.set_title(f'3.  muon half-life distance = '
                  f'{b * c * g * TAU_MUON / 1000:.2f} km', fontsize=10)

    # ---- 4 : trip to a star -----------------------------------------
    t_earth = D / b                # years (D in light-years, v = βc)
    t_ship = t_earth / g

    ax4.clear()
    bars = ax4.bar(['Earth frame', 'traveler'], [t_earth, t_ship],
                   color=['C0', 'C1'], width=0.5)
    ax4.set_ylabel('elapsed time (years)')
    ax4.set_title(f'4.  trip of {D:.1f} light-years at β = {b:.4f}', fontsize=10)
    ax4.grid(alpha=0.3, axis='y')
    for r, v in zip(bars, [t_earth, t_ship]):
        ax4.text(r.get_x() + r.get_width() / 2, v, f'{v:.2f} yr',
                 ha='center', va='bottom', fontsize=9)
    ax4.set_ylim(0, max(t_earth, t_ship) * 1.25 + 1e-9)
    ax4.text(0.5, 0.80, f'aging ratio   1 : {g:.3f}',
             transform=ax4.transAxes, ha='center', fontsize=10,
             bbox=dict(fc='lightyellow', ec='grey'))

    fig.canvas.draw_idle()


for s in (sl_beta, sl_tau, sl_dist):
    s.on_changed(update)

update()
plt.show()
