import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# ------------------------------------------------------------
# 1. Data generation (linearly separable)
# ------------------------------------------------------------
def generate_data(n_samples=100, seed=42):
    np.random.seed(seed)
    # Class 1: y > 0.5*x + 0.2 + noise
    X1 = np.random.randn(n_samples//2, 2) + np.array([1.0, 1.5])
    y1 = np.ones(n_samples//2)
    # Class -1: y < 0.5*x + 0.2 - noise
    X2 = np.random.randn(n_samples//2, 2) + np.array([-1.0, -1.0])
    y2 = -np.ones(n_samples//2)
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

# ------------------------------------------------------------
# 2. Perceptron training (generator that yields state after each update)
# ------------------------------------------------------------
def perceptron_learning(X, y, lr=0.1, max_epochs=100):
    """
    Generator yields (weights, bias, epoch, misclassified_count, total_updates)
    after each misclassification update (i.e., each weight update).
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    epoch = 0
    total_updates = 0
    misclassified = []
    while epoch < max_epochs:
        errors = 0
        for i in range(n_samples):
            xi = X[i]
            yi = y[i]
            if yi * (np.dot(w, xi) + b) <= 0:
                # Update
                w += lr * yi * xi
                b += lr * yi
                errors += 1
                total_updates += 1
                # Yield the current state
                yield w.copy(), b, epoch, errors, total_updates
        # End of epoch
        misclassified.append(errors)
        if errors == 0:
            # Converged
            break
        epoch += 1
    # Final yield (converged or max epochs)
    yield w.copy(), b, epoch, 0, total_updates

# ------------------------------------------------------------
# 3. Set up the figure
# ------------------------------------------------------------
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=0.08, bottom=0.25, top=0.92, right=0.95)

# Left: data + decision boundary
ax_data = plt.subplot2grid((1, 2), (0, 0))
# Right: error curve
ax_err = plt.subplot2grid((1, 2), (0, 1))

# Generate initial data
X, y = generate_data(n_samples=80, seed=42)
n_samples = len(X)

# Plot data (static)
scatter = ax_data.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k', s=50)
ax_data.set_xlim(-3, 4)
ax_data.set_ylim(-3, 4)
ax_data.set_xlabel('x1')
ax_data.set_ylabel('x2')
ax_data.set_title('Perceptron Decision Boundary')
ax_data.grid(True, alpha=0.3)
# Decision boundary line (will be updated)
boundary_line, = ax_data.plot([], [], 'g-', lw=2, label='Decision boundary')
# Margin lines (optional)
margin_lines = ax_data.plot([], [], 'g--', lw=1, alpha=0.5)
ax_data.legend()

# Error plot
ax_err.set_xlabel('Epoch')
ax_err.set_ylabel('Misclassifications')
ax_err.set_title('Convergence')
ax_err.grid(True, alpha=0.3)
err_line, = ax_err.plot([], [], 'r-', lw=2)
err_errors = []   # store errors per epoch

# Current epoch and update counter (display)
ax_text = plt.axes([0.78, 0.10, 0.15, 0.06])
ax_text.axis('off')
text_display = ax_text.text(0, 0, '', fontsize=10, verticalalignment='top')

# ------------------------------------------------------------
# 4. Sliders and buttons
# ------------------------------------------------------------
ax_lr = plt.axes([0.15, 0.12, 0.20, 0.03])
slider_lr = Slider(ax_lr, 'Learning rate', 0.01, 1.0, valinit=0.1, valstep=0.01)
ax_speed = plt.axes([0.40, 0.12, 0.20, 0.03])
slider_speed = Slider(ax_speed, 'Speed (ms)', 50, 500, valinit=100, valstep=10)
ax_samples = plt.axes([0.65, 0.12, 0.20, 0.03])
slider_samples = Slider(ax_samples, 'Samples', 20, 200, valinit=80, valstep=10)

ax_btn_play = plt.axes([0.15, 0.05, 0.10, 0.04])
ax_btn_reset = plt.axes([0.30, 0.05, 0.10, 0.04])

btn_play = Button(ax_btn_play, 'Play')
btn_reset = Button(ax_btn_reset, 'Reset')

# ------------------------------------------------------------
# 5. Animation state
# ------------------------------------------------------------
anim = None
is_playing = False
perceptron_gen = None
current_w = np.zeros(2)
current_b = 0.0
current_epoch = 0
current_updates = 0
history_w = []   # store all (w,b,epoch,updates) for replay
history_err = [] # store errors per epoch

def init_animation():
    """Reset all data and restart training generator."""
    global perceptron_gen, current_w, current_b, current_epoch, current_updates
    global history_w, history_err, err_errors, X, y, n_samples
    n_samples = int(slider_samples.val)
    # Regenerate data with new number of samples
    X, y = generate_data(n_samples=n_samples, seed=np.random.randint(0, 1000))
    # Update scatter plot
    scatter.set_offsets(X)
    scatter.set_array(y)   # for colors
    ax_data.relim()
    ax_data.autoscale_view()
    # Reset error plot
    err_errors = []
    err_line.set_data([], [])
    ax_err.relim()
    # Reset history
    history_w = []
    history_err = []
    # Create new generator
    lr = slider_lr.val
    perceptron_gen = perceptron_learning(X, y, lr=lr, max_epochs=100)
    # Get first state (initial weights)
    try:
        w, b, epoch, err, updates = next(perceptron_gen)
        current_w = w
        current_b = b
        current_epoch = epoch
        current_updates = updates
        history_w.append((w.copy(), b, epoch, updates))
        history_err.append(err)
    except StopIteration:
        pass
    # Update boundary
    update_boundary()
    # Update text
    text_display.set_text(f'Epoch: {current_epoch}\nUpdates: {current_updates}')
    fig.canvas.draw_idle()

def update_boundary():
    """Redraw decision boundary line: w0*x1 + w1*x2 + b = 0 => x2 = -(w0*x1 + b)/w1 if w1 !=0."""
    w = current_w
    b = current_b
    xlim = ax_data.get_xlim()
    ylim = ax_data.get_ylim()
    if w[1] != 0:
        x_vals = np.array(xlim)
        y_vals = -(w[0] * x_vals + b) / w[1]
        # Clip to ylim
        y_vals = np.clip(y_vals, ylim[0], ylim[1])
        boundary_line.set_data(x_vals, y_vals)
    else:
        # vertical line
        if w[0] != 0:
            x_val = -b / w[0]
            boundary_line.set_data([x_val, x_val], ylim)
        else:
            boundary_line.set_data([], [])

def update_frame(frame):
    """Called by FuncAnimation for each frame."""
    global current_w, current_b, current_epoch, current_updates, perceptron_gen
    global history_w, history_err, err_errors
    try:
        w, b, epoch, err, updates = next(perceptron_gen)
        current_w = w
        current_b = b
        current_epoch = epoch
        current_updates = updates
        history_w.append((w.copy(), b, epoch, updates))
        history_err.append(err)
        # Update boundary
        update_boundary()
        # Update error plot
        err_errors.append(err)
        err_line.set_data(np.arange(len(err_errors)), err_errors)
        ax_err.relim()
        ax_err.autoscale_view()
        # Update text
        text_display.set_text(f'Epoch: {epoch}\nUpdates: {updates}')
    except StopIteration:
        # Training finished, stop animation
        global is_playing, anim
        if is_playing:
            toggle_animation(None)
    return boundary_line, err_line

# ------------------------------------------------------------
# 6. Callbacks for buttons
# ------------------------------------------------------------
def toggle_animation(event):
    global is_playing, anim
    if is_playing:
        # Pause
        if anim is not None:
            anim.event_source.stop()
        btn_play.label.set_text('Play')
        is_playing = False
    else:
        # Play
        btn_play.label.set_text('Pause')
        is_playing = True
        # If animation not created, create it
        if anim is None:
            interval = slider_speed.val
            anim = FuncAnimation(fig, update_frame, frames=None,
                                 interval=interval, blit=False, cache_frame_data=False)
        else:
            # Resume
            anim.event_source.start()

def reset_animation(event):
    global anim, is_playing
    # Stop any playing animation
    if is_playing:
        if anim is not None:
            anim.event_source.stop()
        btn_play.label.set_text('Play')
        is_playing = False
    # Reset generator and data
    init_animation()
    # Reset anim object to None so that play creates a fresh one
    if anim is not None:
        anim = None

# Connect buttons
btn_play.on_clicked(toggle_animation)
btn_reset.on_clicked(reset_animation)

# Slider callbacks: when sliders change, reset animation
def slider_change(val):
    reset_animation(None)

slider_lr.on_changed(slider_change)
slider_speed.on_changed(lambda val: setattr(anim, 'interval', slider_speed.val) if anim else None)
slider_samples.on_changed(slider_change)

# ------------------------------------------------------------
# 7. Initial setup
# ------------------------------------------------------------
init_animation()

plt.show()