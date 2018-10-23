from .java_classes import  KerasModelImport
from .ndarray import array

class KerasModel(object):
    def __init__(self, filepath):
        if KerasModelImport is None:
            raise ImportError('DL4J Core not installed.')
        try:
            self.model = KerasModelImport.importKerasModelAndWeights(filepath)
            self.is_sequential = False
        except Exception:
            self.model = KerasModelImport.importKerasSequentialModelAndWeights(filepath)
            self.is_sequential = True


    def __call__(self, input):
        if self.is_sequential:
            if type(input) in [list, tuple]:
                n = len(input)
                if n != 1:
                    err = 'Expected 1 input to sequential model. Received {}.'.format(n)
                    raise ValueError(err)
                input = input[0]
            input = array(input).array
            out = self.model.output(input,  False)
            out = array(out)
            return out
        else:
            if not isinstance(input, list):
                input = [input]
            input = [array(x).array for x in input]
            out = self.model.output(False, *input)
            out = [array(x) for x in out]
            if len(out) == 1:
                return out[0]
            return out
