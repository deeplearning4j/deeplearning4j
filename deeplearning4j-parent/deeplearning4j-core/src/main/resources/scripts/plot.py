from numpy import tanh, fabs, mean, ones,loadtxt
import sys
from PIL import Image
from matplotlib.pyplot import hist, title, subplot
import matplotlib.pyplot as plot
def sigmoid(xx):
    return .5 * (1 + tanh(xx / 2.))



def from_file(path):
    return loadtxt(path,delimiter=',')

def hist_matrix(values):
    hist(from_file(values))
    plot.show()
    #title('mm = %g' % mean(fabs(values)))


def render_hbias(path):
    hMean = from_file(path)
    print hMean
    image = Image.fromarray(hMean * 256).show()
    
def render_filter(path):
    matrix = from_file(path)
    # Initialize background to dark gray
    tiled = ones((11*10, 11*10), dtype='ieruint8') * 51
    for row in xrange(nRows):
         for col in xrange(nCols):
            patch = X[row*nCols + col].reshape((10,10))
            normPatch = ((patch - patch.min()) /
            (patch.max()-patch.min()+1e-6))
            tiled[row*11:row*11+10, col*11:col*11+10] = \
            normPatch * 255
    Image.fromarray(tiled).show()

if __name__ == '__main__':
    if len(sys.argv) < 3:
       print 'Please specify a command: One of hbias,weights,plot and a file path'
       sys.exit(1)
    
    if sys.argv[1] == 'hbias':
        render_hbias(sys.argv[2])
    elif sys.argv[1] == 'weights':
        hist_matrix(sys.argv[2])
        
        