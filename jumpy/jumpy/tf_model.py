from .java_classes import TFGraphMapper, Nd4j, NDArrayIndex
from .ndarray import array

class TFModel(object):
    def __init__(self, filepath):
        self.sd = TFGraphMapper.getInstance().importGraph(filepath)

    def __call__(self, input):
        input = array(input).array  # INDArray
        shape = input.shape()
        dummy_batched = False
        if shape[0] == 1:
            dummy_batched = True
            input = Nd4j.pile(input, input)
        self.sd.associateArrayWithVariable(input)
        out = self.sd.execAndEndResult()
        if dummy_batched:
            out = out.get(NDArrayIndex.point(0))
            out = Nd4j.expandDims(out, 0)
        return array(out)
