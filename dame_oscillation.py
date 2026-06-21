import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------------------------
# 1. Generate the damped oscillation signal and its derivative (for phase)
# ----------------------------------------------------------------------
def damped_signal(t, A, beta, f):
    """A * exp(-beta*t) * sin(2*pi*f*t)"""
    return A * np.exp(-beta * t) * np.sin(2 * np.pi * f * t)

def signal_derivative(t, A, beta, f):
    """Analytical derivative of the damped sine (for velocity/phase plot)."""
    omega = 2 * np.pi * f
    # d/dt [A e^{-βt} sin(ωt)] = A e^{-βt} (ω cos(ωt) - β sin(ωt))
    return A * np.exp(-beta * t) * (omega * np.cos(omega * t) - beta * np.sin(omega * t))

# ----------------------------------------------------------------------
# 2. Set up the figure with 3 subplots
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(left=0.08, bottom=0.25, top=0.92, right=0.95)

# Grid: time (top), spectrum (bottom-left), phase (bottom-right)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
ax_time = fig.add_subplot(gs[0, :])          # time domain takes full top row
ax_spec = fig.add_subplot(gs[1, 0])
ax_phase = fig.add_subplot(gs[1, 1])

# Initial parameters
A_init = 1.0
beta_init = 0.3
f_init = 1.5
total_time = 15.0
num_points = 3000

t = np.linspace(0, total_time, num_points)

# Compute initial signal and its derivative
theta = damped_signal(t, A_init, beta_init, f_init)
omega = signal_derivative(t, A_init, beta_init, f_init)

# ---- Time domain plot ----
line_time, = ax_time.plot(t, theta, 'r-', lw=2, label='θ(t)')
point_time, = ax_time.plot([], [], 'ro', ms=6)   # current position
# Also plot envelope for clarity
env_pos = A_init * np.exp(-beta_init * t)
env_neg = -env_pos
line_env_pos, = ax_time.plot(t, env_pos, 'k--', lw=1, alpha=0.5)
line_env_neg, = ax_time.plot(t, env_neg, 'k--', lw=1, alpha=0.5)
ax_time.set_xlim(0, total_time)
ax_time.set_ylim(-1.5, 1.5)
ax_time.set_xlabel('Time (s)')
ax_time.set_ylabel('Amplitude')
ax_time.set_title('Time Domain – Damped Oscillation')
ax_time.legend(loc='upper right')
ax_time.grid(True, alpha=0.3)

# ---- Frequency spectrum (FFT) ----
def compute_spectrum(t, y):
    dt = t[1] - t[0]
    fft_vals = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), dt)
    pos_mask = freqs > 0
    return freqs[pos_mask], np.abs(fft_vals[pos_mask])

freqs, mag = compute_spectrum(t, theta)
line_spec, = ax_spec.plot(freqs, mag, 'b-', lw=2)
ax_spec.set_xlim(0, 4.0)
ax_spec.set_ylim(0, 1.1 * np.max(mag) if len(mag) > 0 else 1.0)
ax_spec.set_xlabel('Frequency (Hz)')
ax_spec.set_ylabel('|FFT|')
ax_spec.set_title('Fourier Spectrum')
ax_spec.grid(True, alpha=0.3)

# ---- Phase portrait (θ vs. ω) ----
line_phase, = ax_phase.plot(theta, omega, 'g-', lw=2, alpha=0.7)
point_phase, = ax_phase.plot([], [], 'go', ms=6)
ax_phase.set_xlabel('θ (position)')
ax_phase.set_ylabel('ω (velocity)')
ax_phase.set_title('Phase Portrait (θ vs. dθ/dt)')
ax_phase.grid(True, alpha=0.3)
ax_phase.axhline(0, color='gray', lw=0.5)
ax_phase.axvline(0, color='gray', lw=0.5)
# Set limits dynamically after initial plot
ax_phase.set_xlim(1.1 * np.min(theta), 1.1 * np.max(theta))
ax_phase.set_ylim(1.1 * np.min(omega), 1.1 * np.max(omega))

# ----------------------------------------------------------------------
# 3. Sliders and button
# ----------------------------------------------------------------------
ax_slider_A = plt.axes([0.15, 0.12, 0.55, 0.03])
ax_slider_beta = plt.axes([0.15, 0.07, 0.55, 0.03])
ax_slider_f = plt.axes([0.15, 0.02, 0.55, 0.03])

slider_A = Slider(ax_slider_A, 'Amplitude A', 0.1, 2.0, valinit=A_init, valstep=0.05)
slider_beta = Slider(ax_slider_beta, 'Damping β', 0.0, 1.5, valinit=beta_init, valstep=0.01)
slider_f = Slider(ax_slider_f, 'Frequency f (Hz)', 0.2, 5.0, valinit=f_init, valstep=0.1)

# Button for Play/Pause
ax_button = plt.axes([0.78, 0.06, 0.10, 0.05])
btn_play = Button(ax_button, 'Play')
anim_running = False
anim = None

# ----------------------------------------------------------------------
# 4. Data storage and update functions
# ----------------------------------------------------------------------
# Global variables for current signal and animation index
current_data = {
    't': t,
    'theta': theta,
    'omega': omega,
    'freqs': freqs,
    'mag': mag,
    'A': A_init,
    'beta': beta_init,
    'f': f_init
}
current_frame = 0
total_frames = len(t)

def update_signal():
    """Recompute signal and all plots (static parts) based on slider values."""
    global current_data, current_frame, total_frames

    A = slider_A.val
    beta = slider_beta.val
    f = slider_f.val

    # Recompute signal and derivative
    theta_new = damped_signal(t, A, beta, f)
    omega_new = signal_derivative(t, A, beta, f)
    freqs_new, mag_new = compute_spectrum(t, theta_new)

    # Store in global
    current_data['theta'] = theta_new
    current_data['omega'] = omega_new
    current_data['mag'] = mag_new
    current_data['freqs'] = freqs_new
    current_data['A'] = A
    current_data['beta'] = beta
    current_data['f'] = f

    # Update time plot (full trace)
    line_time.set_ydata(theta_new)
    # Update envelope
    env = A * np.exp(-beta * t)
    line_env_pos.set_ydata(env)
    line_env_neg.set_ydata(-env)
    ax_time.set_ylim(1.1 * np.min(theta_new), 1.1 * np.max(theta_new))

    # Update spectrum
    ax_spec.clear()
    ax_spec.plot(freqs_new, mag_new, 'b-', lw=2)
    ax_spec.set_xlim(0, max(2.0, 1.5 * f))
    if len(mag_new) > 0:
        ax_spec.set_ylim(0, 1.1 * np.max(mag_new))
    ax_spec.set_xlabel('Frequency (Hz)')
    ax_spec.set_ylabel('|FFT|')
    ax_spec.set_title('Fourier Spectrum')
    ax_spec.grid(True, alpha=0.3)

    # Update phase portrait (full trace)
    ax_phase.clear()
    ax_phase.plot(theta_new, omega_new, 'g-', lw=2, alpha=0.7)
    ax_phase.set_xlabel('θ (position)')
    ax_phase.set_ylabel('ω (velocity)')
    ax_phase.set_title('Phase Portrait')
    ax_phase.grid(True, alpha=0.3)
    ax_phase.axhline(0, color='gray', lw=0.5)
    ax_phase.axvline(0, color='gray', lw=0.5)
    ax_phase.set_xlim(1.1 * np.min(theta_new), 1.1 * np.max(theta_new))
    ax_phase.set_ylim(1.1 * np.min(omega_new), 1.1 * np.max(omega_new))

    # Reset animation indices
    current_frame = 0
    total_frames = len(t)
    # Clear the animated markers
    point_time.set_data([], [])
    point_phase.set_data([], [])

    fig.canvas.draw_idle()

# ----------------------------------------------------------------------
# 5. Animation update function (grows the trace)
# ----------------------------------------------------------------------
def init_anim():
    point_time.set_data([], [])
    point_phase.set_data([], [])
    return point_time, point_phase

def update_anim(frame):
    if frame >= total_frames:
        return point_time, point_phase

    # Slice data up to current frame
    theta_slice = current_data['theta'][:frame+1]
    omega_slice = current_data['omega'][:frame+1]
    t_slice = t[:frame+1]

    # Update the line data for time and phase?
    # We want the full trace to remain, and just move the point.
    # The line_time already has all data, we just update the point.
    point_time.set_data([t[frame]], [current_data['theta'][frame]])
    point_phase.set_data([current_data['theta'][frame]], [current_data['omega'][frame]])

    return point_time, point_phase

# ----------------------------------------------------------------------
# 6. Callbacks for sliders and button
# ----------------------------------------------------------------------
def on_slider_change(val):
    global anim_running, anim
    update_signal()
    if anim_running:
        # Restart animation if playing
        anim.event_source.stop()
        anim = FuncAnimation(fig, update_anim, init_func=init_anim,
                             frames=total_frames, interval=30, blit=True, repeat=False)
        anim_running = True

slider_A.on_changed(on_slider_change)
slider_beta.on_changed(on_slider_change)
slider_f.on_changed(on_slider_change)

def toggle_animation(event):
    global anim_running, anim
    if anim_running:
        anim.event_source.stop()
        btn_play.label.set_text('Play')
        anim_running = False
    else:
        # Reset to beginning if at end
        if current_frame >= total_frames - 1:
            current_frame = 0
        anim = FuncAnimation(fig, update_anim, init_func=init_anim,
                             frames=total_frames, interval=30, blit=True, repeat=False)
        anim_running = True
        btn_play.label.set_text('Pause')

btn_play.on_clicked(toggle_animation)

# ----------------------------------------------------------------------
# 7. Initial draw and start
# ----------------------------------------------------------------------
update_signal()
plt.show()