from .op import op
from ..java_classes import Nd4j



template = """
@op
def {}(arr, axis=None):
    if axis is None:
        return Nd4j.{}(arr)
    return Nd4j.{}(arr)
"""

reduction_ops = [['max'], ['min'], ['sum'], ['prod'], ['mean'], ['std'], ['var'], ['argmax', 'argMax'], ['argmin', 'argMin']]

for rop in reduction_ops:
    code = template.format(rop[0], rop[-1], rop[-1])
    exec(code)
