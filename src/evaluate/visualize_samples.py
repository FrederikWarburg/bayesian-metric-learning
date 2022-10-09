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
            sigma_sq[index, latent1],
            sigma_sq[index, latent2],
            fc="None",
            edgecolor=color,
            lw=0.5,
        )
        ax.add_patch(elp)
    ax.set(
        xlabel=f"Latent dim {latent1}",
        ylabel=f"Latent dim {latent2}",
    )


def visualize_top_5(id_sigma, id_images, ood_sigma, ood_images, vis_path, prefix, n=5):
    """Visualize the top 5 highest and lowest variance images"""
    id_sigma = id_sigma.numpy()
    id_images = id_images.numpy()
    ood_sigma = ood_sigma.numpy()
    ood_images = ood_images.numpy()

    model_name, dataset, run_name = get_names(vis_path)

    # Get colormap
    channels = id_images.shape[1]

    if channels == 1:
        cmap = "gray"
    elif channels == 3:
        cmap = None

    # Get l2 norm of ID variances
    id_var_mu = np.mean(id_sigma**2, axis=1)

    # get top 5 and bottom 5 of  l2 norm of ID variances
    top_5_id = (-id_var_mu).argsort()[:n]
    bot_5_id = (-id_var_mu).argsort()[-n:]

    # Get l2 norm of OOD variances
    ood_var_mu = np.mean(ood_sigma**2, axis=1)

    # get top 5 and bottom 5 of  l2 norm of OOD variances
    top_5_ood = (-ood_var_mu).argsort()[:n]
    bot_5_ood = (-ood_var_mu).argsort()[-n:]

    # plot top and bottom 5 images
    rows = 4
    columns = n
    fig = plt.figure(figsize=(10, 7))
    counter = 0
    for col in range(columns):
        fig.add_subplot(rows, columns, counter + 1)
        plt.xticks([])
        plt.yticks([])

        image = id_images[top_5_id[col]]
        image = image.transpose(1, 2, 0)
        # Min max scale image to 0, 1
        image = (image - image.min()) / (image.max() - image.min())

        plt.imshow(image, cmap=cmap)
        plt.title(f"ID V={id_var_mu[top_5_id[col]]:.2E}")
        if col == 0:
            plt.ylabel("Top 5 var ID")
        counter += 1

    for col in range(columns):
        fig.add_subplot(rows, columns, counter + 1)
        plt.xticks([])
        plt.yticks([])

        image = id_images[bot_5_id[col]]
        image = image.transpose(1, 2, 0)
        # Min max scale image to 0, 1
        image = (image - image.min()) / (image.max() - image.min())

        plt.imshow(image, cmap=cmap)
        plt.title(f"ID V={id_var_mu[bot_5_id[col]]:.2E}")
        if col == 0:
            plt.ylabel("Bot 5 var ID")
        counter += 1

    for col in range(columns):
        fig.add_subplot(rows, columns, counter + 1)
        plt.xticks([])
        plt.yticks([])

        image = ood_images[top_5_ood[col]]
        image = image.transpose(1, 2, 0)
        # Min max scale image to 0, 1
        image = (image - image.min()) / (image.max() - image.min())

        plt.imshow(image, cmap=cmap)
        plt.title(f"OOD V={ood_var_mu[top_5_ood[col]]:.2E}")
        if col == 0:
            plt.ylabel("Top 5 var OOD")
        counter += 1

    for col in range(columns):
        fig.add_subplot(rows, columns, counter + 1)
        plt.xticks([])
        plt.yticks([])

        image = ood_images[bot_5_ood[col]]
        image = image.transpose(1, 2, 0)
        # Min max scale image to 0, 1
        image = (image - image.min()) / (image.max() - image.min())

        plt.imshow(image, cmap=cmap)
        plt.title(f"OOD V={ood_var_mu[bot_5_ood[col]]:.2E}")
        if col == 0:
            plt.ylabel("Bot 5 var OOD")
        counter += 1

    plt.suptitle(
        f"Top and bottom 5 variance images for model {model_name} ({run_name}) on dataset {dataset}"
    )

    fig.savefig(vis_path / f"{prefix}top_bot_5_var.png")


def ood_visualisations(
    id_mu, id_sigma, id_images, ood_mu, ood_sigma, ood_images, vis_path, prefix
):

    if not prefix.endswith("_"):
        prefix += "_"

    # visualize top 5 and bottom 5 variance images
    visualize_top_5(id_sigma, id_images, ood_sigma, ood_images, vis_path, prefix)

    # Visualize
    metrics = plot_auc_curves(id_sigma, ood_sigma, vis_path, prefix)

    plot_ood(id_mu, id_sigma, ood_mu, ood_sigma, vis_path, prefix)

    return metrics
