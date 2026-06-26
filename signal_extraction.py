import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import find_peaks

# ------------------------------------------------------------
# 1. Parameters
# ------------------------------------------------------------
fs = 500.0              # sampling frequency (Hz)
duration = 2.0          # seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Initial component parameters (three sinusoids)
freqs = [10.0, 25.0, 45.0]   # Hz
amps = [1.0, 0.8, 0.5]

# ------------------------------------------------------------
# 2. Signal generation & FFT utilities
# ------------------------------------------------------------
def generate_mixed(freqs, amps):
    signal = np.zeros_like(t)
    for f, A in zip(freqs, amps):
        signal += A * np.sin(2 * np.pi * f * t)
    return signal

def compute_fft(signal):
    N = len(signal)
    fft_vals = np.fft.fft(signal)
    freqs_fft = np.fft.fftfreq(N, 1/fs)
    return freqs_fft, fft_vals

def separate_components(signal, bandwidth):
    """Returns list of (component_signal, frequency) for each detected peak."""
    freqs_fft, fft_vals = compute_fft(signal)
    N = len(signal)
    pos_mask = freqs_fft > 0
    mag = np.abs(fft_vals[pos_mask])
    freq_pos = freqs_fft[pos_mask]

    # Peak detection
    peaks, _ = find_peaks(mag, height=0.1 * np.max(mag),
                          distance=int(bandwidth / (fs / N)))
    peak_freqs = freq_pos[peaks]

    components = []
    for pf in peak_freqs:
        mask = np.zeros_like(freqs_fft, dtype=bool)
        mask[np.abs(freqs_fft - pf) < bandwidth/2] = True
        mask[np.abs(freqs_fft + pf) < bandwidth/2] = True
        filtered_fft = fft_vals * mask
        comp = np.fft.ifft(filtered_fft).real
        components.append((comp, pf))
    return components

# ------------------------------------------------------------
# 3. Set up the figure (top mixed + 3 component subplots)
# ------------------------------------------------------------
fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(left=0.08, bottom=0.30, top=0.95, right=0.95)

# Grid: top row (mixed), bottom row (three components)
ax_mixed = plt.subplot2grid((2, 3), (0, 0), colspan=3)
ax_comp1 = plt.subplot2grid((2, 3), (1, 0))
ax_comp2 = plt.subplot2grid((2, 3), (1, 1))
ax_comp3 = plt.subplot2grid((2, 3), (1, 2))

component_axes = [ax_comp1, ax_comp2, ax_comp3]

# ---- Mixed signal plot ----
signal = generate_mixed(freqs, amps)
line_mixed, = ax_mixed.plot(t, signal, 'b-', lw=1.5)
ax_mixed.set_xlim(0, duration)
ax_mixed.set_ylim(-2.5, 2.5)
ax_mixed.set_title('Mixed Signal (Top)')
ax_mixed.set_xlabel('Time (s)')
ax_mixed.set_ylabel('Amplitude')
ax_mixed.grid(True, alpha=0.3)

# ---- Component plots (initially empty) ----
comp_lines = []
for ax in component_axes:
    line, = ax.plot([], [], 'r-', lw=1.5)
    comp_lines.append(line)
    ax.set_xlim(0, duration)
    ax.set_ylim(-2, 2)
    ax.set_title('Component')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)

# ------------------------------------------------------------
# 4. Sliders for frequencies and amplitudes
# ------------------------------------------------------------
ax_slider_f1 = plt.axes([0.12, 0.22, 0.25, 0.02])
ax_slider_f2 = plt.axes([0.40, 0.22, 0.25, 0.02])
ax_slider_f3 = plt.axes([0.68, 0.22, 0.25, 0.02])
ax_slider_A1 = plt.axes([0.12, 0.17, 0.25, 0.02])
ax_slider_A2 = plt.axes([0.40, 0.17, 0.25, 0.02])
ax_slider_A3 = plt.axes([0.68, 0.17, 0.25, 0.02])
ax_slider_bw = plt.axes([0.12, 0.12, 0.25, 0.02])

slider_f1 = Slider(ax_slider_f1, 'f1 (Hz)', 1, 50, valinit=freqs[0], valstep=0.5)
slider_f2 = Slider(ax_slider_f2, 'f2 (Hz)', 1, 50, valinit=freqs[1], valstep=0.5)
slider_f3 = Slider(ax_slider_f3, 'f3 (Hz)', 1, 50, valinit=freqs[2], valstep=0.5)
slider_A1 = Slider(ax_slider_A1, 'A1', 0.0, 2.0, valinit=amps[0], valstep=0.05)
slider_A2 = Slider(ax_slider_A2, 'A2', 0.0, 2.0, valinit=amps[1], valstep=0.05)
slider_A3 = Slider(ax_slider_A3, 'A3', 0.0, 2.0, valinit=amps[2], valstep=0.05)
slider_bw = Slider(ax_slider_bw, 'Bandwidth (Hz)', 0.5, 10.0, valinit=2.0, valstep=0.5)

# ------------------------------------------------------------
# 5. Button
# ------------------------------------------------------------
ax_button = plt.axes([0.50, 0.07, 0.12, 0.05])
btn_sep = Button(ax_button, 'Separate')

# ------------------------------------------------------------
# 6. Update functions
# ------------------------------------------------------------
def update_mixed(val=None):
    # Read slider values
    freqs[0] = slider_f1.val
    freqs[1] = slider_f2.val
    freqs[2] = slider_f3.val
    amps[0] = slider_A1.val
    amps[1] = slider_A2.val
    amps[2] = slider_A3.val
    # Update mixed signal
    signal = generate_mixed(freqs, amps)
    line_mixed.set_ydata(signal)
    ax_mixed.relim()
    ax_mixed.autoscale_view(scalex=False)
    fig.canvas.draw_idle()

def separate_callback(event):
    # Get current signal and bandwidth
    signal = generate_mixed(freqs, amps)
    bw = slider_bw.val
    components = separate_components(signal, bw)

    # Clear all component axes and replot
    for ax in component_axes:
        ax.clear()
        ax.set_xlim(0, duration)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)

    # Plot up to 3 components
    for i, (comp, pf) in enumerate(components):
        if i >= 3:   # we only have 3 subplots
            break
        ax = component_axes[i]
        ax.plot(t, comp, 'r-', lw=1.5)
        ax.set_title(f'Component {i+1}: {pf:.1f} Hz')

    # Hide any unused subplots (if fewer than 3 components)
    for j in range(len(components), 3):
        component_axes[j].set_title('(no component)')

    fig.canvas.draw_idle()

# Connect callbacks
slider_f1.on_changed(update_mixed)
slider_f2.on_changed(update_mixed)
slider_f3.on_changed(update_mixed)
slider_A1.on_changed(update_mixed)
slider_A2.on_changed(update_mixed)
slider_A3.on_changed(update_mixed)
btn_sep.on_clicked(separate_callback)

# Initial draw
update_mixed()

plt.show()