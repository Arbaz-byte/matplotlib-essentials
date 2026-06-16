import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------------------------
# 1. Fourier coefficients for different signal types
# ----------------------------------------------------------------------
def fourier_coeffs(signal_type, N, T, A):
    """
    Return complex coefficients c_n for n = -N ... N.
    signal_type: 'square', 'sawtooth', 'triangle'
    """
    n_vals = np.arange(-N, N+1)
    c = np.zeros(len(n_vals), dtype=complex)
    omega0 = 2 * np.pi / T

    for i, n in enumerate(n_vals):
        if n == 0:
            if signal_type == 'square':
                c[i] = 0.0
            elif signal_type == 'sawtooth':
                c[i] = 0.0
            elif signal_type == 'triangle':
                c[i] = A / 2.0
        else:
            if signal_type == 'square':
                if n % 2 == 1:  # odd harmonics
                    c[i] = (2 * A) / (1j * np.pi * n)
                else:
                    c[i] = 0.0
            elif signal_type == 'sawtooth':
                # sawtooth: f(t) = 2A*(t/T - floor(t/T+0.5))  (centered)
                c[i] = (1j * A) / (np.pi * n) * (-1)**(n+1)
            elif signal_type == 'triangle':
                if n % 2 == 1:
                    c[i] = (4 * A) / (np.pi**2 * n**2)
                else:
                    c[i] = 0.0
    return n_vals, c

def reconstruct(t, n_vals, c, T):
    """Sum c_n * exp(j n omega0 t)"""
    omega0 = 2 * np.pi / T
    f = np.zeros_like(t, dtype=complex)
    for n, cn in zip(n_vals, c):
        f += cn * np.exp(1j * n * omega0 * t)
    return np.real(f)

def original_signal(t, signal_type, T, A):
    """Return the ideal waveform for comparison."""
    if signal_type == 'square':
        return A * np.sign(np.sin(2 * np.pi * t / T))
    elif signal_type == 'sawtooth':
        return 2 * A * (t / T - np.floor(t / T + 0.5))
    elif signal_type == 'triangle':
        return A * (2 * np.abs(2 * (t / T - np.floor(t / T + 0.5))) - 1)

# ----------------------------------------------------------------------
# 2. Set up the figure and widgets
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(left=0.1, bottom=0.25, top=0.92, right=0.95)

# Create three subplots
ax_time = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax_mag  = plt.subplot2grid((2, 2), (1, 0))
ax_phase= plt.subplot2grid((2, 2), (1, 1))

# Initial parameters
signal_type = 'square'
T = 2.0          # period
A = 1.0          # amplitude
N_max = 50       # maximum number of harmonics (each side)
N_init = 1

# Time axis
t = np.linspace(-T, T, 1000)

# Compute initial coefficients and reconstruction
n_vals, c = fourier_coeffs(signal_type, N_init, T, A)
f_approx = reconstruct(t, n_vals, c, T)
f_orig = original_signal(t, signal_type, T, A)

# ---- Time domain plot ----
line_orig, = ax_time.plot(t, f_orig, 'k--', label='Original', lw=1.5)
line_approx, = ax_time.plot(t, f_approx, 'r-', label='Approximation', lw=2)
ax_time.set_xlabel('Time')
ax_time.set_ylabel('Amplitude')
ax_time.legend(loc='upper right')
ax_time.grid(True, alpha=0.3)
ax_time.set_title('Time Domain')

# ---- Magnitude spectrum ----
n_display = n_vals
mag = np.abs(c)
stem_mag = ax_mag.stem(n_display, mag, basefmt=' ')
ax_mag.set_xlabel('Harmonic index n')
ax_mag.set_ylabel('|c_n|')
ax_mag.grid(True, alpha=0.3)
ax_mag.set_title('Magnitude Spectrum')

# ---- Phase spectrum ----
phase = np.angle(c)
stem_phase = ax_phase.stem(n_display, phase, basefmt=' ')
ax_phase.set_xlabel('Harmonic index n')
ax_phase.set_ylabel('Phase (rad)')
ax_phase.grid(True, alpha=0.3)
ax_phase.set_title('Phase Spectrum')

# ---- Sliders ----
ax_slider_N = plt.axes([0.15, 0.12, 0.60, 0.03])
slider_N = Slider(ax_slider_N, 'Number of terms (N)', 0, N_max, valinit=N_init, valstep=1)

ax_slider_T = plt.axes([0.15, 0.07, 0.60, 0.03])
slider_T = Slider(ax_slider_T, 'Period T', 0.5, 5.0, valinit=T)

ax_slider_A = plt.axes([0.15, 0.02, 0.60, 0.03])
slider_A = Slider(ax_slider_A, 'Amplitude A', 0.1, 2.0, valinit=A)

# ---- Radio buttons for signal type ----
ax_radio = plt.axes([0.80, 0.05, 0.12, 0.15])
radio = RadioButtons(ax_radio, ('square', 'sawtooth', 'triangle'), active=0)

# ---- Button for animation ----
ax_button = plt.axes([0.80, 0.02, 0.10, 0.04])
btn_animate = Button(ax_button, 'Animate')
# Status of animation
anim_running = False
anim = None

# ----------------------------------------------------------------------
# 3. Update function
# ----------------------------------------------------------------------
def update_plot(val=None):
    """Redraw all plots based on current widget values."""
    global N_init, T, A, signal_type, n_vals, c, f_approx, f_orig

    # Read current values
    N = int(slider_N.val)
    T = slider_T.val
    A = slider_A.val
    signal_type = radio.value_selected

    # Compute coefficients and reconstruction
    n_vals, c = fourier_coeffs(signal_type, N, T, A)
    f_approx = reconstruct(t, n_vals, c, T)
    f_orig = original_signal(t, signal_type, T, A)

    # Update time plot
    line_orig.set_ydata(f_orig)
    line_approx.set_ydata(f_approx)
    ax_time.relim()
    ax_time.autoscale_view()

    # Update magnitude spectrum
    ax_mag.clear()
    ax_mag.stem(n_vals, np.abs(c), basefmt=' ')
    ax_mag.set_xlabel('Harmonic index n')
    ax_mag.set_ylabel('|c_n|')
    ax_mag.grid(True, alpha=0.3)
    ax_mag.set_title('Magnitude Spectrum')

    # Update phase spectrum
    ax_phase.clear()
    ax_phase.stem(n_vals, np.angle(c), basefmt=' ')
    ax_phase.set_xlabel('Harmonic index n')
    ax_phase.set_ylabel('Phase (rad)')
    ax_phase.grid(True, alpha=0.3)
    ax_phase.set_title('Phase Spectrum')

    fig.canvas.draw_idle()

# Connect sliders and radio buttons
slider_N.on_changed(update_plot)
slider_T.on_changed(update_plot)
slider_A.on_changed(update_plot)
radio.on_clicked(update_plot)

# ----------------------------------------------------------------------
# 4. Animation (grow N automatically)
# ----------------------------------------------------------------------
def animate_frame(frame):
    """Called by FuncAnimation; increments N by 1 each frame."""
    current_N = int(slider_N.val)
    if current_N < N_max:
        slider_N.set_val(current_N + 1)   # this triggers update_plot
    else:
        # If at max, stop the animation (optional)
        pass
    return []

def toggle_animation(event):
    global anim_running, anim
    if anim_running:
        anim.event_source.stop()
        btn_animate.label.set_text('Animate')
        anim_running = False
    else:
        # Reset to N=0 if we are at max and start
        if int(slider_N.val) == N_max:
            slider_N.set_val(0)
        anim = FuncAnimation(fig, animate_frame, interval=200, blit=False, cache_frame_data=False)
        anim_running = True
        btn_animate.label.set_text('Stop')

btn_animate.on_clicked(toggle_animation)

# ----------------------------------------------------------------------
# 5. Show the interactive window
# ----------------------------------------------------------------------
plt.show()