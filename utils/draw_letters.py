import matplotlib.pyplot as plt
import seaborn as sns


def create_letter_plot(letter, ax, cmap='Blues'):
    p = sns.heatmap(letter, ax=ax, annot=False, cbar=False, cmap=cmap, square=True, linewidth=2, linecolor='black')
    p.xaxis.set_visible(False)
    p.yaxis.set_visible(False)
    return p


def print_letters_line(letters, cmap='Blues', cmaps=[]):
    fig, ax = plt.subplots(1,len(letters))
    fig.set_dpi(360)
    if not cmaps:
        cmaps = [cmap] * len(letters)
    if len(cmaps) != len(letters):
        raise Exception('cmaps list should be the same length as letters')
    for i, subplot in enumerate(ax):
        create_letter_plot(letters[i].reshape(5, 5), ax=subplot, cmap=cmaps[i])
    plt.show()
