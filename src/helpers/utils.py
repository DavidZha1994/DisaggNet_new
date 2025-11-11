#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : NILMFormer - Utils
#
#################################################################################################################

import os
import matplotlib.pyplot as plt


def create_dir(path):
    os.makedirs(path, exist_ok=True)

    return path


def check_file_exist(path):
    return os.path.isfile(path)


def apply_graphics_setting(ax=None, legend_font_size=20, label_fontsize=20):
    if ax is None:
        ax = plt.gca()
        for pos in ["right", "top", "bottom", "left"]:
            ax.spines[pos].set_visible(False)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(label_fontsize)

        plt.grid(linestyle="-.")
        plt.legend(fontsize=legend_font_size)
        plt.tight_layout()
    else:
        for pos in ["right", "top", "bottom", "left"]:
            ax.spines[pos].set_visible(False)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(label_fontsize)

        ax.grid(linestyle="-.")
        ax.legend(fontsize=legend_font_size)
        ax.figure.tight_layout()
