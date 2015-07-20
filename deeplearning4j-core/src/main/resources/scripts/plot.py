import math
from matplotlib.pyplot import hist, title, subplot, scatter, plot
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn # improves matplotlib look and feel
import sys
import time

'''
Optimization Methods Visualalization

Graph tools to help visualize how optimization is performing
'''

GLOBAL_TIME = 1.5

def load_file(path):
    return np.loadtxt(path, delimiter=',')


def sigmoid(hidden_mean):
    return 1 / (1 + np.exp(-hidden_mean))

def render_plot(values, plot_type='histogram', chart_title=''):
    if np.product(values.shape) < 2:
        values = np.zeros((3, 3))
        chart_title += '-fake'

    if plot_type == 'histogram':
        hist(values)
    elif plot_type == "scatter":
        scatter(values)
    else:
        print "The " + plot_type + " format is not supported. Please choose histogram or scatter."
    magnitude = ' mm %g ' % np.mean(np.fabs(values))
    chart_title += ' ' + magnitude
    title(chart_title)

def render_activation_probability(dataPath, filename):
    hidden_mean = load_file(dataPath)
    img = Image.fromarray(sigmoid(hidden_mean) * 256)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(filename, 'PNG')

def plot_single_graph(path, chart_title, filename):
    print 'Graphing ' + chart_title + '\n'
    values = load_file(path)
    plt.plot(values, 'b')
    plt.title(chart_title)
    plt.savefig(filename, format='png')
    plt.show(block=False)
    time.sleep(GLOBAL_TIME)
    plt.close()

def plot_matrices(orig_path, plot_type, filename):
    paths = orig_path.split(',')

    for idx, path in enumerate(paths):
        if idx % 2 == 0:
            title = paths[idx + 1]
            print 'Loading matrix ' + title + '\n'
            matrix = load_file(path)
            subplot(2, len(paths)/4, idx/2+1)
            render_plot(matrix, plot_type, chart_title=title)

    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.show(block=False)
    time.sleep(GLOBAL_TIME)
    plt.close()


# TODO Finish adapting. Code still does not fully run through.
# def render_filter(data_path, n_rows, n_cols, filename):
#     weight_data = load_file(data_path).reshape((n_rows, n_cols))
#     patch_width = weight_data.shape[1]
#     patch_height = 1
#
#     # Initialize background to dark gray
#     filter_frame = np.ones((n_rows*patch_width, n_cols * patch_height), dtype='uint8')
#
#     for row in xrange(int(n_rows/n_cols)):
#         for col in xrange(n_cols):
#             patch = weight_data[row * n_cols + col].reshape((patch_width, patch_height))
#             norm_patch = ((patch - patch.min()) / (patch.max() - patch.min() + 1e-6))
#             filter_frame[row * patch_width: row * patch_width + patch_width,
#             col * patch_height:col * patch_height + patch_height] = norm_patch * 255
#     img = Image.fromarray(filter_frame)
#     img.savefig(filename)
#     img.show()
#
# def render_filter(data_path, filename, filter_width=10, filter_height=10):
#     print 'Rendering filter image...'
#     weight_data = load_file(data_path)
#     n_rows = weight_data.shape[0]
#     n_cols = weight_data.shape[1]
#     padding = 1
#
#     # Initialize background to dark gray
#     filter_frame = np.ones(((filter_width+padding) * filter_width, (filter_height+padding) * filter_height), dtype='uint8') * 51
#
#     for row in xrange(n_rows):
#         for col in xrange(n_cols):
#             patch = weight_data[row * n_cols + col].reshape((filter_width, filter_height))
#             norm_patch = ((patch - patch.min()) / (patch.max() - patch.min() + 1e-6))
#             filter_frame[row * (filter_height+padding): row * (filter_height+padding)+filter_height, col * (filter_width+padding): col * (filter_width+padding)+filter_width] = norm_patch * 255
#             filter_frame[row * (filter_height+padding): row * (filter_height+padding) + filter_height, col * (filter_width+padding): col *(filter_width+padding) + filter_width]
#         img = Image.fromarray(filter_frame)
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     img.save(filename)

# def vis_square(data_path, filename, n_rows=28, n_cols=28, padsize=1, padval=0):
#     data = load_file(data_path)
#     data = data.reshape(n_rows, n_cols)
#
#     data -= data.min()
#     data /= data.max()
#
#     # force the number of filters to be square
#     n = int(np.ceil(np.sqrt(data.shape[0])))
#     padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
#     data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
#
#     # tile the filters into an image
#     data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
#     data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
#
#     plt.imshow(data)
#     time.sleep(GLOBAL_TIME)
#     plt.savefig(data, filename)



if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Please specify a command: One of hbias,weights,plot and a file path'
        sys.exit(1)
    plot_type = sys.argv[1]
    path = sys.argv[2]
    filename = sys.argv[3]

    if plot_type == 'activations':
        render_activation_probability(path, filename)
    elif plot_type == 'single_matrix':
        render_plot(path)
    elif plot_type == 'histogram':
        plot_matrices(path, plot_type, filename)
    elif plot_type == 'scatter':
        plot_matrices(path, plot_type, filename)
    elif plot_type == 'loss':
        plot_single_graph(path, plot_type, filename)
    elif plot_type == 'accuracy':
        plot_single_graph(path, plot_type, filename)
    # elif sys.argv[1] == 'filter':
    #     if sys.argv[7]:
    #         n_rows = int(sys.argv[4])
    #         n_cols = int(sys.argv[5])
    #         filter_width = int(sys.argv[6])
    #         filter_height = int(sys.argv[7])
    #         render_filter(path, filename, n_rows, n_cols, filter_height, filter_width)
    #     elif sys.argv[5]:
    #         n_rows = int(sys.argv[4])
    #         n_cols = int(sys.argv[5])
    #         render_filter(path,  filename, n_rows, n_cols)
    #     else:
    #         render_filter(path, filename)
