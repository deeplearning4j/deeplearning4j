import math
from matplotlib.pyplot import hist, title, subplot, scatter
import matplotlib.pyplot as plt
from numpy import tanh, fabs, mean, ones, loadtxt, fromfile, zeros, product
from PIL import Image
import seaborn
import sys

'''
Optimization Methods Visualalization

Graph tools to help visualize how optimization is performing
'''


def load_file(path):
    return loadtxt(path, delimiter=',')


def sigmoid(xx):
    return .5 * (1 + tanh(xx / 2.))


def render_hbias(path):
    hMean = load_file(path)
    Image.fromarray(hMean * 256).show()


def render_plot(values, plot_type="hist", chart_title=''):
    if product(values.shape) < 2:
        values = zeros((3, 3))
        chart_title += '-fake'

    if plot_type == "hist" or "multi":
        hist(values)
    else:
        scatter(values)

    magnitude = ' mm %g ' % mean(fabs(values))
    chart_title += ' ' + magnitude
    title(chart_title)


def plot_multiple_matrices(orig_path, plot_type):
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


def render_filter(path, n_rows, n_cols, data_length):
    X = load_file(path)
    data = math.sqrt(n_rows)
    print 'data ' + str(data)
    # Initialize background to dark gray
    tiled = ones((data * n_rows, data * n_cols), dtype='unit8') * 51

    for row in xrange(n_rows):
         for col in xrange(n_cols):
            curr_neuron = col
            patch = X[:, curr_neuron].reshape((data, data))
            normPatch = ((patch - patch.min()) / (patch.max() - patch.min() + 1e-6))
            tiled[row * data:row * data + data, col * data:col * data + data] = normPatch * 255
    Image.fromarray(tiled).show()


if __name__ == '__main__':
    if len(sys.argv) < 3:
       print 'Please specify a command: One of hbias,weights,plot and a file path'
       sys.exit(1)
    plot_type = sys.argv[1]
    path = sys.argv[2]

    if plot_type == 'hbias':
        render_hbias(path)
    elif plot_type == 'hist':
        render_plot(path, plot_type)
    elif plot_type == 'multi':
        plot_multiple_matrices(path, plot_type)
    elif plot_type == 'scatter':
        plot_multiple_matrices(path, plot_type)
    elif sys.argv[1] == 'filter':
        n_rows = int(sys.argv[3])
        n_cols = int(sys.argv[4])
        length = int(sys.argv[5])
        print 'Rendering ' + sys.argv[3] + ' x ' + sys.argv[4] + ' matrix'
        render_filter(path, n_rows, n_cols, length)