import tensorflow as tf

dtypes = [tf.float16,tf.float32,tf.float64,tf.int8,tf.int16,tf.int32,tf.int64,tf.uint8,tf.uint16,tf.uint32,tf.uint64]
# Quick solution from https://stackoverflow.com/questions/5360220/how-to-split-a-list-into-pairs-in-all-possible-ways :)
import itertools
def all_pairs(lst):
    return [(x,y) for x in dtypes for y in dtypes]


for item in all_pairs(dtypes):
        from_dtype, out_dtype = item
        tf.reset_default_graph()
        input = tf.placeholder(name='input',dtype=from_dtype)
        result = tf.cast(input,name='cast_output',dtype=out_dtype)

        with tf.Session() as session:
            tf.train.write_graph(tf.get_default_graph(),logdir='.',name='cast_' + from_dtype.name + '_'  + out_dtype.name + '.pb',as_text=True)