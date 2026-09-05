import numpy as np
import matplotlib.pyplot as plt

def entropy(prob):
    """
    Compute Shannon entropy (in bits) for a discrete probability distribution.
    prob : array-like, probabilities summing to 1.
    """
    prob = np.array(prob)
    # Avoid log(0): set 0 probabilities to 1 so that p*log(p) = 0
    prob_safe = np.where(prob == 0, 1, prob)
    return -np.sum(prob * np.log2(prob_safe))

# ---------------------------
# 1. Binary entropy plot
# ---------------------------
p_vals = np.linspace(0.001, 0.999, 1000)  # avoid exact 0 and 1
h_vals = [entropy([p, 1-p]) for p in p_vals]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(p_vals, h_vals, 'b-', linewidth=2)
plt.xlabel('p')
plt.ylabel('H(p) [bits]')
plt.title('Binary Entropy')
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)

# Mark maximum entropy at p=0.5
plt.plot(0.5, 1, 'ro', markersize=8)
plt.annotate('Max entropy = 1 bit', xy=(0.5, 1), xytext=(0.6, 0.9),
             arrowprops=dict(facecolor='black', shrink=0.05))

# ---------------------------
# 2. Discrete distribution example
# ---------------------------
# Example distribution (must sum to 1)
probs = [0.5, 0.25, 0.15, 0.1]
labels = ['A', 'B', 'C', 'D']
H = entropy(probs)

plt.subplot(1, 2, 2)
bars = plt.bar(labels, probs, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
plt.ylim(0, max(probs)*1.2)
plt.ylabel('Probability')
plt.title(f'Discrete Distribution\nEntropy = {H:.3f} bits')

# Add probability values on top of bars
for bar, prob in zip(bars, probs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{prob:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Optionally, print entropy value
print(f"Entropy of distribution {dict(zip(labels, probs))} = {H:.3f} bits")
