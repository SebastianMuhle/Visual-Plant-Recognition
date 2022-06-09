#
# BSD 3-Clause License
#
#
# Copyright (c) 2022, Sebastian Muhle, Dominik Muhle.
# All rights reserved.

from matplotlib import pyplot as plt

_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
    "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia?", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
    "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow",
    "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily"
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
        ax.imshow(sample['image'])
        title = f"Ground Truth: {sample['plant_label']}, {_NAMES[sample['plant_label']]}"
        ax.set_title(title)

def class_balance_visualization(class_sample_counts: dict):
    plt.figure(figsize=(22, 5))    
    plt.bar(range(len(class_sample_counts)), list(class_sample_counts.values()), align='center')
    plt.xticks(range(len(class_sample_counts)), list(class_sample_counts.keys()))
    plt.xticks(rotation=80)
    plt.title('Samples per Class')
    plt.show()