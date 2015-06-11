import math
from matplotlib.pyplot import hist, title, subplot, scatter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn
import sys

'''
Optimization Methods Visualalization

Graph tools to help visualize how optimization is performing
'''


def load_file(path):
    return np.loadtxt(path, delimiter=',')


def sigmoid(hidden_mean):
    # return .5 * (1 + tanh(xx / 2.))
    return 1 / (1 + np.exp(-hidden_mean))


def render_plot(values, plot_type='histogram', chart_title=''):
    if np.product(values.shape) < 2:
        values = np.zeros((3, 3))
        chart_title += '-fake'

    if plot_type == 'histogram':
        hist(values)
    elif "scatter":
        scatter(values)
    else:
        print "The " + plot_type + " format is not supported. Please choose histogram or scatter."
    magnitude = ' mm %g ' % np.mean(np.fabs(values))
    chart_title += ' ' + magnitude
    title(chart_title)


def render_activation_probability(dataPath):
    hidden_mean = load_file(dataPath)
    # Should sigmoid be moved earlier?
    Image.fromarray(sigmoid(hidden_mean) * 256).show()


def plot_matrices(orig_path, plot_type=''):
    paths = orig_path.split(',')

    for idx, path in enumerate(paths):
        if idx % 2 == 0:
            title = paths[idx + 1]
            print 'Loading matrix ' + title + '\n'
            matrix = load_file(path)
            subplot(2, len(paths)/4, idx/2+1)
            render_plot(matrix, plot_type, chart_title=title)

    plt.tight_layout()
    plt.show()

# TODO Finish adapting. Code still does not fully run through.
# FilterRender produces a result for now
# Goal is to consolidate these into a common script
def render_filter(data_path, n_rows, n_cols):
    weight_data = load_file(data_path).reshape((n_rows, n_cols))
    patch_width = weight_data.shape[1]
    patch_height = 1

    # Initialize background to dark gray
    filter_frame = np.ones((n_rows*patch_width, n_cols * patch_height), dtype='uint8')

    for row in xrange(int(n_rows/n_cols)):
        for col in xrange(n_cols):
            patch = weight_data[row * n_cols + col].reshape((patch_width, patch_height))
            norm_patch = ((patch - patch.min()) / (patch.max() - patch.min() + 1e-6))
            filter_frame[row * patch_width: row * patch_width + patch_width,
                  col * patch_height:col * patch_height + patch_height] = norm_patch * 255
    Image.fromarray(filter_frame).show()


if __name__ == '__main__':
    if len(sys.argv) < 3:
       print 'Please specify a command: One of hbias,weights,plot and a file path'
       sys.exit(1)
    plot_type = sys.argv[1]
    path = sys.argv[2]

    if plot_type == 'activations':
        render_activation_probability(path)
    elif plot_type == 'single_matrix':
        render_plot(path)
    elif plot_type == 'histogram':
        plot_matrices(path, plot_type)
    elif plot_type == 'scatter':
        plot_matrices(path, plot_type)
    elif sys.argv[1] == 'filter':
        n_rows = int(sys.argv[3])
        n_cols = int(sys.argv[4])
        length = int(sys.argv[5])
        print 'Rendering ' + sys.argv[3] + ' x ' + sys.argv[4] + ' matrix'
        render_filter(path, n_rows, n_cols, length)