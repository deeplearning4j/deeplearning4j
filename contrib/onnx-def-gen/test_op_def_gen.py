from onnx_tf.common import attr_converter,attr_translator
from onnx_tf.handlers.backend import *
import onnx_tf
import onnx_tf.handlers.handler
import sys,inspect
import tensorflow as tf
from onnx_tf.backend import TensorflowBackend


current_module = sys.modules['onnx_tf.handlers.backend']
modules = inspect.getmembers(current_module)
for name, obj in modules:
 obj_modules = inspect.getmembers(obj)
 for name2,module2 in obj_modules:
    if inspect.isclass(module2):
        result = module2
        print(module2)