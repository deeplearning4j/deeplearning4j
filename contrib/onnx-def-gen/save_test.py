import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def graph_as_func():
    S = tf.Variable(tf.constant([1, 2, 3, 4]))
    result = tf.scatter_add(S, [0], [10])
    return result
a_function_that_uses_a_graph = tf.function(graph_as_func)
print(a_function_that_uses_a_graph.__attr__)
converted = convert_variables_to_constants_v2(a_function_that_uses_a_graph)
print(type(converted))