from numpy import tanh, fabs, mean, ones,loadtxt,fromfile,zeros,product
import sys
import math
from PIL import Image
from matplotlib.pyplot import hist, title, subplot
import matplotlib.pyplot as plot
def sigmoid(xx):
    return .5 * (1 + tanh(xx / 2.))



def from_file(path):
    return loadtxt(path,delimiter=',')
    
def hist_matrix(values,show = True,chart_title = ''):
    if product(values.shape) < 2:
      values = zeros((3,3))
      chart_title += '-fake'

    hist(values)
    magnitude = ' mm %g ' % mean(fabs(values))
    chart_title += ' ' + magnitude
    title(chart_title)
    


def render_hbias(path):
    hMean = from_file(path)
    image = Image.fromarray(hMean * 256).show()


def plot_multiple_matrices(paths):
    count = 231
    graph_count = 1
    print paths
    for i  in xrange(len(paths) - 1):
        if i % 2 == 0:
            path = paths[i]
            title = paths[i + 1]
            print 'Loading matrix ' + path + '\n'
            matrix = from_file(path)
            subplot(2,3,graph_count)
            plot.tight_layout()
            hist_matrix(matrix,False,chart_title=title)
            graph_count+= 1


    plot.show()    

def render_filter(path,nRows,nCols,data_length):
    X = from_file(path)
    data = math.sqrt(nRows)
    print 'data ' + str(data)
    # Initialize background to dark gray
    tiled = ones((data*nRows, data*nCols), dtype='uint8') * 51
    for row in xrange(nRows):
         for col in xrange(nCols):
            curr_neuron = col
            patch = X[:,curr_neuron].reshape((data,data))
            normPatch = ((patch - patch.min()) /
            (patch.max()-patch.min()+1e-6))
            tiled[row*data:row*data+data, col*data:col*data+data] = normPatch * 255
    Image.fromarray(tiled).show()

if __name__ == '__main__':
    if len(sys.argv) < 3:
       print 'Please specify a command: One of hbias,weights,plot and a file path'
       sys.exit(1)
    
    if sys.argv[1] == 'hbias':
        render_hbias(sys.argv[2])
    elif sys.argv[1] == 'weights':
        hist_matrix(sys.argv[2])
    elif sys.argv[1] == 'filter':
        nRows = int(sys.argv[3])
        nCols = int(sys.argv[4])
        length = int(sys.argv[5])
        print 'Rendering ' + sys.argv[3] + ' x ' + sys.argv[4] + ' matrix'
        render_filter(sys.argv[2],nRows,nCols,length)
    elif sys.argv[1] == 'multi':
        paths = sys.argv[2].split(',')
        plot_multiple_matrices(paths)
        
        
