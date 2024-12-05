import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.figure_factory as ff
from matplotlib import colors
import utils
from utils import training_path, evaluation_path, concept_path, training_tasks_files, evaluation_tasks_files, concept_tasks_files

# Visualize one item
def plot_one(task, ax, i, train_or_test, input_or_output='!'):
    cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    if input_or_output == '!':
        input_matrix = task
    else:
        input_matrix = task[train_or_test][i][input_or_output]

    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' ' + input_or_output, size='small')

# Visualize entire item
def plot_task(task):
    num_train = len(task['train'])
    fig, axs = plt.subplots(2, num_train, figsize=(2.5*num_train,2.5*2))
    for i in range(num_train):
        plot_one(task, axs[0,i], i, 'train', 'input')
        plot_one(task, axs[1,i], i, 'train', 'output')
    plt.tight_layout()
    plt.show()

    num_test = len(task['test'])
    num_output = len(task['test'][0])
    fig, axs = plt.subplots(2, num_test, figsize=(2.5*num_test,2.5*2))
    if num_test == 1:
        plot_one(task, axs[0], 0, 'test', 'input')
        if num_output > 1:
            plot_one(task, axs[1], 0, 'test', 'output')
    else:
        for i in range(num_test):
            plot_one(task, axs[0,i], i, 'test', 'input')
            plot_one(task, axs[1,i], i, 'test', 'output')
    plt.tight_layout()
    plt.show()

# Plot random ARC item
def plot_arc(example_num = None, path ='training'):
    idx = random.randint(0, 99) if example_num is None else example_num

    task_file = f'{eval(path + "_path")}{(eval(path + "_tasks_files"))[idx]}'
    with open(task_file, 'r') as f:
        example = json.load(f)

    plot_task(example)

# Visualize accuracy across processed items in scatterplot with average line
def plot_pix_acc(X_inp, X_out, exclude_zero=False):
    per_diff = utils.accuracy(X_inp, X_out, exclude_zero)
    m = np.mean(per_diff)

    _, _ = plt.subplots(figsize=(11,5))
    plt.plot(per_diff, color='steelblue', marker='.', linewidth=0)
    plt.axhline(m, xmax = len(per_diff), color='firebrick')
    plt.title(f'Accuracy ofn (Tasks: {len(per_diff)})', size='medium')
    plt.xlabel('Item')
    plt.ylabel('Correct Pixel (%)')
    plt.text(len(per_diff)/2, m+0.01, f'{(m*100).round(2)}%', size='medium', weight='bold')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=max([1])))
    plt.ylim(-0.01, 1.01)
    plt.margins(x=0.01)
    plt.show()
    print(f'Number of 100% Correct: {per_diff.count(1)}')
    print(f'Number of 90%+ Correct: {sum(i >= 0.9 for i in per_diff)}')
    print(f'Number of 80%+ Correct: {sum(i >= 0.8 for i in per_diff)}')
    print(f'Number of 70%+ Correct: {sum(i >= 0.7 for i in per_diff)}')

# Visualize heatmap of individually solved pixels
def plot_pix_heatmap(X_inp, X_out):
    pix_diff = []
    for i in range(len(X_inp)):
        pix_diff.append(X_inp[i] == X_out[i])

    pix_sum = np.sum(pix_diff, axis=0)

    _, ax = plt.subplots(figsize=(5,5))
    _ = sns.heatmap(pix_sum, cmap='inferno_r', square=True) # oder viridis_r
    plt.title('Accuracy of Individual Pixel Reconstruction', size='medium', y=1.04)
    plt.text(0.2, 32, f'Number of 100% Correct Pixels: {np.count_nonzero((pix_sum == len(X_inp)))}', size='medium')
    plt.axis('off')
    plt.show()


def summary_plot(X_test):

    # Plot chosen items
    matrices = utils.flatten_dataset(X_test)

    plot_arc(29, path ="concept")


    # Plot distribution of mean colors
    means = [np.mean(matrix) for matrix in matrices]
    fig = ff.create_distplot([means], group_labels=["Means"], colors=["green"])
    fig.update_layout(title_text="Distribution of matrix mean values")



    # Plot joint plot of item dimensions: scatter plot, density plot, distribution plot
    heights = [np.shape(matrix)[0] for matrix in matrices]
    widths = [np.shape(matrix)[1] for matrix in matrices]

    plot = sns.jointplot(x=widths, y=heights, kind="kde", fill=True, thresh = 0.09, color="blueviolet")
    plot.set_axis_labels(xlabel="Width", ylabel="Height", fontsize=14)
    plt.show()

    plot = sns.jointplot(x=widths, y=heights, kind="reg", color="blueviolet")
    plot.set_axis_labels(xlabel="Width", ylabel="Height", fontsize=14)
    plt.show()

# Showcase Padding, One-Hot Encoding, Dimensionality, and Augmentations
def showcase_padding(data_load, X_train, y_train):
    train_loader = data_load(X_train, y_train, aug=[True, True, True], shuffle=True)
    # e.g., all augmentations are applied

    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        idx = random.randrange(len(train_loader.dataset.X))  # a random grid is chosen
        plot_one(utils.reverse_one_hot_encoder(train_loader.dataset.X[idx]), axs[0, i], i, 'input')
        plot_one(utils.reverse_one_hot_encoder(train_loader.dataset.y[idx]), axs[1, i], i, 'output')
    plt.show()

def do_scatter_plot(data):
    sns.scatterplot(
        data=data,
        x="Average_Zeros_X",
        y="Average_Zeros_Y",
        hue="Average_Zeros_X",
        legend=None)
    sns.despine()
    plt.xlabel('Average % of Zero in Example Inputs')
    plt.ylabel('Average % of Zero in Example Outputs')
    plt.show()

    X = data.iloc[:, 0:-1]
    corr = X.corr(method='spearman')

    # Generate mask for upper triangle (just symmetric)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Create custom colormap to highlight pos. & neg. correlations
    cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)

    # Draw heatmap (/w mask & colormap)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)
    fig.suptitle('Correlation Matrix: Features', fontsize=15, weight='bold')
    fig.tight_layout()
    plt.show()

