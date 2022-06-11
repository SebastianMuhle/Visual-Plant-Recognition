#
# BSD 3-Clause License
#
#
# Copyright (c) 2022, Sebastian Muhle, Dominik Muhle.
# All rights reserved.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

_NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]


def image_visualization(image, gt_class, pred_class=None):
    fig, ax = plt.subplots(1, 1)

    ax.imshow(image)

    pred_class_string = pred_class if pred_class else "-"
    pred_class_name = _NAMES[pred_class] if pred_class else "-"

    title = f"Ground Truth: {gt_class}, {_NAMES[gt_class]}\nPrediction: {pred_class_string}, {pred_class_name}"
    ax.set_title(title)

    return fig, ax


def batch_visualization(dataset, size: tuple = (3, 2), fig_size: tuple = (10, 5)):
    fig, axs = plt.subplots(size[0], size[1], figsize=fig_size)

    for i, ax in enumerate(axs.flat):
        sample = dataset[i]
        ax.imshow(sample["image"])
        title = f"Ground Truth: {sample['plant_label']}, {_NAMES[sample['plant_label']]}"
        ax.set_title(title)


def per_class_accuracy(conf_matrix: np.ndarray) -> Figure:
    font = {"family": "Times", "size": 22}
    matplotlib.rc("font", **font)

    # TODO: remove +1 when evaluating the real dataset
    figsize = (15, 3)
    num_correct = np.diag(conf_matrix) + 1
    num_total = np.sum(conf_matrix, axis=1) + 1

    accuracy = num_correct / num_total

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(accuracy, bins=20, range=(0.0, 1.0))
    ax.set_ylim(0, conf_matrix.shape[0])
    ax.set_xlabel(f"class accuracy")
    ax.set_ylabel(f"number of classes")
    # plt.show()
    return fig


def worst_classes(conf_matrix: np.ndarray, k: int) -> Figure:
    font = {"family": "Times", "size": 22}
    matplotlib.rc("font", **font)

    # TODO: remove +1 when evaluating the real dataset
    figsize = (15, k * 3)
    num_correct = np.diag(conf_matrix) + 1
    num_total = np.sum(conf_matrix, axis=1) + 1

    accuracy = num_correct / num_total
    worst_k_idx = np.argsort(accuracy)[:k]

    worst_k = conf_matrix[worst_k_idx, :]

    fig, axs = plt.subplots(k, 1, figsize=figsize)
    plt.tight_layout()
    for predictions, idx, ax in zip(worst_k, worst_k_idx, axs):
        total_predictions = np.sum(predictions)
        ax.bar(np.arange(predictions.size) + 1, predictions / total_predictions, align="center")
        ax.set_title(f"{idx + 1}: {_NAMES[idx]}")
        ax.text(0.7, 0.85, f"{predictions[idx] / total_predictions * 100:.2f} % correct")
        ax.set_xlim(0.5, conf_matrix.shape[0] + 0.5)
        ax.set_ylim(0.0, 1.0)
    axs[-1].set_xlabel(f"labels")
    # plt.show()
    return fig
