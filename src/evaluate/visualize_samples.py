import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def plot_samples(
    mu, sigma_sq, latent1=0, latent2=1, limit=100, ax=None, color="b", label=None
):
    if ax is None:
        _, ax = plt.subplots()

    indices = np.random.choice(np.arange(mu.shape[0]), size=limit)

    ax.scatter(
        mu[indices, latent1],
        mu[indices, latent2],
        s=0.5,
        c=color,
        label=label,
    )
    for index in indices:
        elp = Ellipse(
            (mu[index, latent1], mu[index, latent2]),
            sigma_sq[index, latent1] + 1e-6,
            sigma_sq[index, latent2] + 1e-6,
            fc="None",
            edgecolor=color,
            lw=0.5,
        )
        ax.add_patch(elp)
    ax.set(
        xlabel=f"Latent dim {latent1}",
        ylabel=f"Latent dim {latent2}",
    )

