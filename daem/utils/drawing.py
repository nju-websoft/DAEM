import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def draw_heatmap(seq_x, seq_y, heat, title=None, show_label=True):
    fig, ax = plt.subplots()
    im = ax.imshow(heat.T)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(seq_x)))
    ax.set_yticks(np.arange(len(seq_y)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(seq_x)
    ax.set_yticklabels(seq_y)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if show_label:
        for i in range(len(seq_x)):
            for j in range(len(seq_y)):
                text = ax.text(i, j, heat[i, j], ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def draw_series():
    pass