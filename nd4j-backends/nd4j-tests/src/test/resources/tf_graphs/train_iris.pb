node {
  name: "Placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 4
        }
      }
    }
  }
}
node {
  name: "Placeholder_1"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 3
        }
      }
    }
  }
}
node {
  name: "dense/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\004\000\000\000\005\000\000\000"
      }
    }
  }
}
node {
  name: "dense/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.816496610641
      }
    }
  }
}
node {
  name: "dense/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.816496610641
      }
    }
  }
}
node {
  name: "dense/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "dense/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "dense/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "dense/kernel/Initializer/random_uniform/max"
  input: "dense/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/kernel"
      }
    }
  }
}
node {
  name: "dense/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "dense/kernel/Initializer/random_uniform/RandomUniform"
  input: "dense/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/kernel"
      }
    }
  }
}
node {
  name: "dense/kernel/Initializer/random_uniform"
  op: "Add"
  input: "dense/kernel/Initializer/random_uniform/mul"
  input: "dense/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/kernel"
      }
    }
  }
}
node {
  name: "dense/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
        dim {
          size: 5
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "dense/kernel/Assign"
  op: "Assign"
  input: "dense/kernel"
  input: "dense/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "dense/kernel/read"
  op: "Identity"
  input: "dense/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/kernel"
      }
    }
  }
}
node {
  name: "dense/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 5
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "dense/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "dense/bias/Assign"
  op: "Assign"
  input: "dense/bias"
  input: "dense/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "dense/bias/read"
  op: "Identity"
  input: "dense/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/bias"
      }
    }
  }
}
node {
  name: "dense/MatMul"
  op: "MatMul"
  input: "Placeholder"
  input: "dense/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dense/BiasAdd"
  op: "BiasAdd"
  input: "dense/MatMul"
  input: "dense/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "dense/Sigmoid"
  op: "Sigmoid"
  input: "dense/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dense_1/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\005\000\000\000\003\000\000\000"
      }
    }
  }
}
node {
  name: "dense_1/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.866025388241
      }
    }
  }
}
node {
  name: "dense_1/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.866025388241
      }
    }
  }
}
node {
  name: "dense_1/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "dense_1/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "dense_1/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "dense_1/kernel/Initializer/random_uniform/max"
  input: "dense_1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/kernel"
      }
    }
  }
}
node {
  name: "dense_1/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "dense_1/kernel/Initializer/random_uniform/RandomUniform"
  input: "dense_1/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/kernel"
      }
    }
  }
}
node {
  name: "dense_1/kernel/Initializer/random_uniform"
  op: "Add"
  input: "dense_1/kernel/Initializer/random_uniform/mul"
  input: "dense_1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/kernel"
      }
    }
  }
}
node {
  name: "dense_1/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 3
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "dense_1/kernel/Assign"
  op: "Assign"
  input: "dense_1/kernel"
  input: "dense_1/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "dense_1/kernel/read"
  op: "Identity"
  input: "dense_1/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/kernel"
      }
    }
  }
}
node {
  name: "dense_1/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 3
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "dense_1/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "dense_1/bias/Assign"
  op: "Assign"
  input: "dense_1/bias"
  input: "dense_1/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "dense_1/bias/read"
  op: "Identity"
  input: "dense_1/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/bias"
      }
    }
  }
}
node {
  name: "dense_2/MatMul"
  op: "MatMul"
  input: "dense/Sigmoid"
  input: "dense_1/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dense_2/BiasAdd"
  op: "BiasAdd"
  input: "dense_2/MatMul"
  input: "dense_1/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "Softmax"
  op: "Softmax"
  input: "dense_2/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Shape"
  op: "Shape"
  input: "dense_2/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Rank_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Shape_1"
  op: "Shape"
  input: "dense_2/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Sub/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Sub"
  op: "Sub"
  input: "softmax_cross_entropy_loss/Rank_1"
  input: "softmax_cross_entropy_loss/Sub/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Slice/begin"
  op: "Pack"
  input: "softmax_cross_entropy_loss/Sub"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Slice/size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Slice"
  op: "Slice"
  input: "softmax_cross_entropy_loss/Shape_1"
  input: "softmax_cross_entropy_loss/Slice/begin"
  input: "softmax_cross_entropy_loss/Slice/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/concat/values_0"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/concat/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/concat"
  op: "ConcatV2"
  input: "softmax_cross_entropy_loss/concat/values_0"
  input: "softmax_cross_entropy_loss/Slice"
  input: "softmax_cross_entropy_loss/concat/axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Reshape"
  op: "Reshape"
  input: "dense_2/BiasAdd"
  input: "softmax_cross_entropy_loss/concat"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Rank_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Shape_2"
  op: "Shape"
  input: "Placeholder_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Sub_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Sub_1"
  op: "Sub"
  input: "softmax_cross_entropy_loss/Rank_2"
  input: "softmax_cross_entropy_loss/Sub_1/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Slice_1/begin"
  op: "Pack"
  input: "softmax_cross_entropy_loss/Sub_1"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Slice_1/size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Slice_1"
  op: "Slice"
  input: "softmax_cross_entropy_loss/Shape_2"
  input: "softmax_cross_entropy_loss/Slice_1/begin"
  input: "softmax_cross_entropy_loss/Slice_1/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/concat_1/values_0"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/concat_1/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/concat_1"
  op: "ConcatV2"
  input: "softmax_cross_entropy_loss/concat_1/values_0"
  input: "softmax_cross_entropy_loss/Slice_1"
  input: "softmax_cross_entropy_loss/concat_1/axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Reshape_1"
  op: "Reshape"
  input: "Placeholder_1"
  input: "softmax_cross_entropy_loss/concat_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/xentropy"
  op: "SoftmaxCrossEntropyWithLogits"
  input: "softmax_cross_entropy_loss/Reshape"
  input: "softmax_cross_entropy_loss/Reshape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Sub_2/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Sub_2"
  op: "Sub"
  input: "softmax_cross_entropy_loss/Rank"
  input: "softmax_cross_entropy_loss/Sub_2/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Slice_2/begin"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Slice_2/size"
  op: "Pack"
  input: "softmax_cross_entropy_loss/Sub_2"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Slice_2"
  op: "Slice"
  input: "softmax_cross_entropy_loss/Shape"
  input: "softmax_cross_entropy_loss/Slice_2/begin"
  input: "softmax_cross_entropy_loss/Slice_2/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Reshape_2"
  op: "Reshape"
  input: "softmax_cross_entropy_loss/xentropy"
  input: "softmax_cross_entropy_loss/Slice_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/assert_broadcastable/weights"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/assert_broadcastable/weights/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/assert_broadcastable/weights/rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/assert_broadcastable/values/shape"
  op: "Shape"
  input: "softmax_cross_entropy_loss/Reshape_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/assert_broadcastable/values/rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  op: "NoOp"
}
node {
  name: "softmax_cross_entropy_loss/ToFloat_1/x"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Mul"
  op: "Mul"
  input: "softmax_cross_entropy_loss/Reshape_2"
  input: "softmax_cross_entropy_loss/ToFloat_1/x"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Const"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Sum"
  op: "Sum"
  input: "softmax_cross_entropy_loss/Mul"
  input: "softmax_cross_entropy_loss/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/Equal/y"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/Equal"
  op: "Equal"
  input: "softmax_cross_entropy_loss/ToFloat_1/x"
  input: "softmax_cross_entropy_loss/num_present/Equal/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/zeros_like"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/ones_like/Shape"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/ones_like/Const"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/ones_like"
  op: "Fill"
  input: "softmax_cross_entropy_loss/num_present/ones_like/Shape"
  input: "softmax_cross_entropy_loss/num_present/ones_like/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/Select"
  op: "Select"
  input: "softmax_cross_entropy_loss/num_present/Equal"
  input: "softmax_cross_entropy_loss/num_present/zeros_like"
  input: "softmax_cross_entropy_loss/num_present/ones_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shape"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rank"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shape"
  op: "Shape"
  input: "softmax_cross_entropy_loss/Reshape_2"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rank"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success"
  op: "NoOp"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
}
node {
  name: "softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Shape"
  op: "Shape"
  input: "softmax_cross_entropy_loss/Reshape_2"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  input: "^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  input: "^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like"
  op: "Fill"
  input: "softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Shape"
  input: "softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/broadcast_weights"
  op: "Mul"
  input: "softmax_cross_entropy_loss/num_present/Select"
  input: "softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present/Const"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/num_present"
  op: "Sum"
  input: "softmax_cross_entropy_loss/num_present/broadcast_weights"
  input: "softmax_cross_entropy_loss/num_present/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Const_1"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Sum_1"
  op: "Sum"
  input: "softmax_cross_entropy_loss/Sum"
  input: "softmax_cross_entropy_loss/Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Greater/y"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Greater"
  op: "Greater"
  input: "softmax_cross_entropy_loss/num_present"
  input: "softmax_cross_entropy_loss/Greater/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Equal/y"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Equal"
  op: "Equal"
  input: "softmax_cross_entropy_loss/num_present"
  input: "softmax_cross_entropy_loss/Equal/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/ones_like/Shape"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/ones_like/Const"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/ones_like"
  op: "Fill"
  input: "softmax_cross_entropy_loss/ones_like/Shape"
  input: "softmax_cross_entropy_loss/ones_like/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/Select"
  op: "Select"
  input: "softmax_cross_entropy_loss/Equal"
  input: "softmax_cross_entropy_loss/ones_like"
  input: "softmax_cross_entropy_loss/num_present"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/div"
  op: "RealDiv"
  input: "softmax_cross_entropy_loss/Sum_1"
  input: "softmax_cross_entropy_loss/Select"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/zeros_like"
  op: "Const"
  input: "^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss/value"
  op: "Select"
  input: "softmax_cross_entropy_loss/Greater"
  input: "softmax_cross_entropy_loss/div"
  input: "softmax_cross_entropy_loss/zeros_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "model"
      }
    }
  }
}
node {
  name: "save/SaveV2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 4
          }
        }
        string_val: "dense/bias"
        string_val: "dense/kernel"
        string_val: "dense_1/bias"
        string_val: "dense_1/kernel"
      }
    }
  }
}
node {
  name: "save/SaveV2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 4
          }
        }
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
      }
    }
  }
}
node {
  name: "save/SaveV2"
  op: "SaveV2"
  input: "save/Const"
  input: "save/SaveV2/tensor_names"
  input: "save/SaveV2/shape_and_slices"
  input: "dense/bias"
  input: "dense/kernel"
  input: "dense_1/bias"
  input: "dense_1/kernel"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/control_dependency"
  op: "Identity"
  input: "save/Const"
  input: "^save/SaveV2"
  attr {
    key: "T"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@save/Const"
      }
    }
  }
}
node {
  name: "save/RestoreV2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "dense/bias"
      }
    }
  }
}
node {
  name: "save/RestoreV2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2/tensor_names"
  input: "save/RestoreV2/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign"
  op: "Assign"
  input: "dense/bias"
  input: "save/RestoreV2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_1/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "dense/kernel"
      }
    }
  }
}
node {
  name: "save/RestoreV2_1/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_1"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_1/tensor_names"
  input: "save/RestoreV2_1/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_1"
  op: "Assign"
  input: "dense/kernel"
  input: "save/RestoreV2_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "dense_1/bias"
      }
    }
  }
}
node {
  name: "save/RestoreV2_2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_2"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_2/tensor_names"
  input: "save/RestoreV2_2/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_2"
  op: "Assign"
  input: "dense_1/bias"
  input: "save/RestoreV2_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_3/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "dense_1/kernel"
      }
    }
  }
}
node {
  name: "save/RestoreV2_3/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_3"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_3/tensor_names"
  input: "save/RestoreV2_3/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_3"
  op: "Assign"
  input: "dense_1/kernel"
  input: "save/RestoreV2_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_all"
  op: "NoOp"
  input: "^save/Assign"
  input: "^save/Assign_1"
  input: "^save/Assign_2"
  input: "^save/Assign_3"
}
node {
  name: "gradients/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "gradients/Fill"
  op: "Fill"
  input: "gradients/Shape"
  input: "gradients/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/value_grad/zeros_like"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/value_grad/Select"
  op: "Select"
  input: "softmax_cross_entropy_loss/Greater"
  input: "gradients/Fill"
  input: "gradients/softmax_cross_entropy_loss/value_grad/zeros_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/value_grad/Select_1"
  op: "Select"
  input: "softmax_cross_entropy_loss/Greater"
  input: "gradients/softmax_cross_entropy_loss/value_grad/zeros_like"
  input: "gradients/Fill"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/softmax_cross_entropy_loss/value_grad/Select"
  input: "^gradients/softmax_cross_entropy_loss/value_grad/Select_1"
}
node {
  name: "gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/softmax_cross_entropy_loss/value_grad/Select"
  input: "^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/softmax_cross_entropy_loss/value_grad/Select"
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/softmax_cross_entropy_loss/value_grad/Select_1"
  input: "^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/softmax_cross_entropy_loss/value_grad/Select_1"
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/softmax_cross_entropy_loss/div_grad/Shape"
  input: "gradients/softmax_cross_entropy_loss/div_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/RealDiv"
  op: "RealDiv"
  input: "gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency"
  input: "softmax_cross_entropy_loss/Select"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/Sum"
  op: "Sum"
  input: "gradients/softmax_cross_entropy_loss/div_grad/RealDiv"
  input: "gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/Reshape"
  op: "Reshape"
  input: "gradients/softmax_cross_entropy_loss/div_grad/Sum"
  input: "gradients/softmax_cross_entropy_loss/div_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/Neg"
  op: "Neg"
  input: "softmax_cross_entropy_loss/Sum_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1"
  op: "RealDiv"
  input: "gradients/softmax_cross_entropy_loss/div_grad/Neg"
  input: "softmax_cross_entropy_loss/Select"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2"
  op: "RealDiv"
  input: "gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1"
  input: "softmax_cross_entropy_loss/Select"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/mul"
  op: "Mul"
  input: "gradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency"
  input: "gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/Sum_1"
  op: "Sum"
  input: "gradients/softmax_cross_entropy_loss/div_grad/mul"
  input: "gradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/softmax_cross_entropy_loss/div_grad/Sum_1"
  input: "gradients/softmax_cross_entropy_loss/div_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/softmax_cross_entropy_loss/div_grad/Reshape"
  input: "^gradients/softmax_cross_entropy_loss/div_grad/Reshape_1"
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/softmax_cross_entropy_loss/div_grad/Reshape"
  input: "^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/softmax_cross_entropy_loss/div_grad/Reshape_1"
  input: "^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape"
  op: "Reshape"
  input: "gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency"
  input: "gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile"
  op: "Tile"
  input: "gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape"
  input: "gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Select_grad/zeros_like"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Select_grad/Select"
  op: "Select"
  input: "softmax_cross_entropy_loss/Equal"
  input: "gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1"
  input: "gradients/softmax_cross_entropy_loss/Select_grad/zeros_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Select_grad/Select_1"
  op: "Select"
  input: "softmax_cross_entropy_loss/Equal"
  input: "gradients/softmax_cross_entropy_loss/Select_grad/zeros_like"
  input: "gradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/softmax_cross_entropy_loss/Select_grad/Select"
  input: "^gradients/softmax_cross_entropy_loss/Select_grad/Select_1"
}
node {
  name: "gradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/softmax_cross_entropy_loss/Select_grad/Select"
  input: "^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select"
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/softmax_cross_entropy_loss/Select_grad/Select_1"
  input: "^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1"
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Sum_grad/Reshape"
  op: "Reshape"
  input: "gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile"
  input: "gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Sum_grad/Shape"
  op: "Shape"
  input: "softmax_cross_entropy_loss/Mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Sum_grad/Tile"
  op: "Tile"
  input: "gradients/softmax_cross_entropy_loss/Sum_grad/Reshape"
  input: "gradients/softmax_cross_entropy_loss/Sum_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Mul_grad/Shape"
  op: "Shape"
  input: "softmax_cross_entropy_loss/Reshape_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/softmax_cross_entropy_loss/Mul_grad/Shape"
  input: "gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Mul_grad/mul"
  op: "Mul"
  input: "gradients/softmax_cross_entropy_loss/Sum_grad/Tile"
  input: "softmax_cross_entropy_loss/ToFloat_1/x"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Mul_grad/Sum"
  op: "Sum"
  input: "gradients/softmax_cross_entropy_loss/Mul_grad/mul"
  input: "gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Mul_grad/Reshape"
  op: "Reshape"
  input: "gradients/softmax_cross_entropy_loss/Mul_grad/Sum"
  input: "gradients/softmax_cross_entropy_loss/Mul_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Mul_grad/mul_1"
  op: "Mul"
  input: "softmax_cross_entropy_loss/Reshape_2"
  input: "gradients/softmax_cross_entropy_loss/Sum_grad/Tile"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1"
  op: "Sum"
  input: "gradients/softmax_cross_entropy_loss/Mul_grad/mul_1"
  input: "gradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1"
  input: "gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape"
  input: "^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1"
}
node {
  name: "gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/softmax_cross_entropy_loss/Mul_grad/Reshape"
  input: "^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1"
  input: "^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present_grad/Reshape"
  op: "Reshape"
  input: "gradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1"
  input: "gradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present_grad/Shape"
  op: "Shape"
  input: "softmax_cross_entropy_loss/num_present/broadcast_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present_grad/Tile"
  op: "Tile"
  input: "gradients/softmax_cross_entropy_loss/num_present_grad/Reshape"
  input: "gradients/softmax_cross_entropy_loss/num_present_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1"
  op: "Shape"
  input: "softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape"
  input: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul"
  op: "Mul"
  input: "gradients/softmax_cross_entropy_loss/num_present_grad/Tile"
  input: "softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum"
  op: "Sum"
  input: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul"
  input: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape"
  op: "Reshape"
  input: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum"
  input: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1"
  op: "Mul"
  input: "softmax_cross_entropy_loss/num_present/Select"
  input: "gradients/softmax_cross_entropy_loss/num_present_grad/Tile"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1"
  op: "Sum"
  input: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1"
  input: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1"
  input: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape"
  input: "^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1"
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape"
  input: "^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1"
  input: "^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Sum"
  op: "Sum"
  input: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1"
  input: "gradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape"
  op: "Shape"
  input: "softmax_cross_entropy_loss/xentropy"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Reshape_2_grad/Reshape"
  op: "Reshape"
  input: "gradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency"
  input: "gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/zeros_like"
  op: "ZerosLike"
  input: "softmax_cross_entropy_loss/xentropy:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims"
  op: "ExpandDims"
  input: "gradients/softmax_cross_entropy_loss/Reshape_2_grad/Reshape"
  input: "gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/xentropy_grad/mul"
  op: "Mul"
  input: "gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims"
  input: "softmax_cross_entropy_loss/xentropy:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Reshape_grad/Shape"
  op: "Shape"
  input: "dense_2/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape"
  op: "Reshape"
  input: "gradients/softmax_cross_entropy_loss/xentropy_grad/mul"
  input: "gradients/softmax_cross_entropy_loss/Reshape_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dense_2/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "gradients/dense_2/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape"
  input: "^gradients/dense_2/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "gradients/dense_2/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape"
  input: "^gradients/dense_2/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/dense_2/BiasAdd_grad/BiasAddGrad"
  input: "^gradients/dense_2/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dense_2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "gradients/dense_2/MatMul_grad/MatMul"
  op: "MatMul"
  input: "gradients/dense_2/BiasAdd_grad/tuple/control_dependency"
  input: "dense_1/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/dense_2/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "dense/Sigmoid"
  input: "gradients/dense_2/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/dense_2/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/dense_2/MatMul_grad/MatMul"
  input: "^gradients/dense_2/MatMul_grad/MatMul_1"
}
node {
  name: "gradients/dense_2/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/dense_2/MatMul_grad/MatMul"
  input: "^gradients/dense_2/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dense_2/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "gradients/dense_2/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/dense_2/MatMul_grad/MatMul_1"
  input: "^gradients/dense_2/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dense_2/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "gradients/dense/Sigmoid_grad/SigmoidGrad"
  op: "SigmoidGrad"
  input: "dense/Sigmoid"
  input: "gradients/dense_2/MatMul_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dense/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "gradients/dense/Sigmoid_grad/SigmoidGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "gradients/dense/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/dense/Sigmoid_grad/SigmoidGrad"
  input: "^gradients/dense/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "gradients/dense/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/dense/Sigmoid_grad/SigmoidGrad"
  input: "^gradients/dense/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dense/Sigmoid_grad/SigmoidGrad"
      }
    }
  }
}
node {
  name: "gradients/dense/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/dense/BiasAdd_grad/BiasAddGrad"
  input: "^gradients/dense/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dense/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "gradients/dense/MatMul_grad/MatMul"
  op: "MatMul"
  input: "gradients/dense/BiasAdd_grad/tuple/control_dependency"
  input: "dense/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/dense/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "Placeholder"
  input: "gradients/dense/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/dense/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/dense/MatMul_grad/MatMul"
  input: "^gradients/dense/MatMul_grad/MatMul_1"
}
node {
  name: "gradients/dense/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/dense/MatMul_grad/MatMul"
  input: "^gradients/dense/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dense/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "gradients/dense/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/dense/MatMul_grad/MatMul_1"
  input: "^gradients/dense/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dense/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "GradientDescent/learning_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000475
      }
    }
  }
}
node {
  name: "GradientDescent/update_dense/kernel/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "dense/kernel"
  input: "GradientDescent/learning_rate"
  input: "gradients/dense/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent/update_dense/bias/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "dense/bias"
  input: "GradientDescent/learning_rate"
  input: "gradients/dense/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent/update_dense_1/kernel/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "dense_1/kernel"
  input: "GradientDescent/learning_rate"
  input: "gradients/dense_2/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent/update_dense_1/bias/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "dense_1/bias"
  input: "GradientDescent/learning_rate"
  input: "gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent"
  op: "NoOp"
  input: "^GradientDescent/update_dense/kernel/ApplyGradientDescent"
  input: "^GradientDescent/update_dense/bias/ApplyGradientDescent"
  input: "^GradientDescent/update_dense_1/kernel/ApplyGradientDescent"
  input: "^GradientDescent/update_dense_1/bias/ApplyGradientDescent"
}
node {
  name: "init"
  op: "NoOp"
  input: "^dense/kernel/Assign"
  input: "^dense/bias/Assign"
  input: "^dense_1/kernel/Assign"
  input: "^dense_1/bias/Assign"
}
node {
  name: "Placeholder_2"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 4
        }
      }
    }
  }
}
node {
  name: "Placeholder_3"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 3
        }
      }
    }
  }
}
node {
  name: "dense_2/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\004\000\000\000\005\000\000\000"
      }
    }
  }
}
node {
  name: "dense_2/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.816496610641
      }
    }
  }
}
node {
  name: "dense_2/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.816496610641
      }
    }
  }
}
node {
  name: "dense_2/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "dense_2/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "dense_2/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "dense_2/kernel/Initializer/random_uniform/max"
  input: "dense_2/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/kernel"
      }
    }
  }
}
node {
  name: "dense_2/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "dense_2/kernel/Initializer/random_uniform/RandomUniform"
  input: "dense_2/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/kernel"
      }
    }
  }
}
node {
  name: "dense_2/kernel/Initializer/random_uniform"
  op: "Add"
  input: "dense_2/kernel/Initializer/random_uniform/mul"
  input: "dense_2/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/kernel"
      }
    }
  }
}
node {
  name: "dense_2/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
        dim {
          size: 5
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "dense_2/kernel/Assign"
  op: "Assign"
  input: "dense_2/kernel"
  input: "dense_2/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "dense_2/kernel/read"
  op: "Identity"
  input: "dense_2/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/kernel"
      }
    }
  }
}
node {
  name: "dense_2/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 5
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "dense_2/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "dense_2/bias/Assign"
  op: "Assign"
  input: "dense_2/bias"
  input: "dense_2/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "dense_2/bias/read"
  op: "Identity"
  input: "dense_2/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/bias"
      }
    }
  }
}
node {
  name: "dense_3/MatMul"
  op: "MatMul"
  input: "Placeholder_2"
  input: "dense_2/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dense_3/BiasAdd"
  op: "BiasAdd"
  input: "dense_3/MatMul"
  input: "dense_2/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "dense_3/Sigmoid"
  op: "Sigmoid"
  input: "dense_3/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dense_3/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\005\000\000\000\003\000\000\000"
      }
    }
  }
}
node {
  name: "dense_3/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.866025388241
      }
    }
  }
}
node {
  name: "dense_3/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.866025388241
      }
    }
  }
}
node {
  name: "dense_3/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "dense_3/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "dense_3/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "dense_3/kernel/Initializer/random_uniform/max"
  input: "dense_3/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/kernel"
      }
    }
  }
}
node {
  name: "dense_3/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "dense_3/kernel/Initializer/random_uniform/RandomUniform"
  input: "dense_3/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/kernel"
      }
    }
  }
}
node {
  name: "dense_3/kernel/Initializer/random_uniform"
  op: "Add"
  input: "dense_3/kernel/Initializer/random_uniform/mul"
  input: "dense_3/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/kernel"
      }
    }
  }
}
node {
  name: "dense_3/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 3
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "dense_3/kernel/Assign"
  op: "Assign"
  input: "dense_3/kernel"
  input: "dense_3/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "dense_3/kernel/read"
  op: "Identity"
  input: "dense_3/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/kernel"
      }
    }
  }
}
node {
  name: "dense_3/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 3
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "dense_3/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "dense_3/bias/Assign"
  op: "Assign"
  input: "dense_3/bias"
  input: "dense_3/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "dense_3/bias/read"
  op: "Identity"
  input: "dense_3/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/bias"
      }
    }
  }
}
node {
  name: "dense_4/MatMul"
  op: "MatMul"
  input: "dense_3/Sigmoid"
  input: "dense_3/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dense_4/BiasAdd"
  op: "BiasAdd"
  input: "dense_4/MatMul"
  input: "dense_3/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "Softmax_1"
  op: "Softmax"
  input: "dense_4/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Shape"
  op: "Shape"
  input: "dense_4/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Rank_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Shape_1"
  op: "Shape"
  input: "dense_4/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Sub/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Sub"
  op: "Sub"
  input: "softmax_cross_entropy_loss_1/Rank_1"
  input: "softmax_cross_entropy_loss_1/Sub/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Slice/begin"
  op: "Pack"
  input: "softmax_cross_entropy_loss_1/Sub"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Slice/size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Slice"
  op: "Slice"
  input: "softmax_cross_entropy_loss_1/Shape_1"
  input: "softmax_cross_entropy_loss_1/Slice/begin"
  input: "softmax_cross_entropy_loss_1/Slice/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/concat/values_0"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/concat/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/concat"
  op: "ConcatV2"
  input: "softmax_cross_entropy_loss_1/concat/values_0"
  input: "softmax_cross_entropy_loss_1/Slice"
  input: "softmax_cross_entropy_loss_1/concat/axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Reshape"
  op: "Reshape"
  input: "dense_4/BiasAdd"
  input: "softmax_cross_entropy_loss_1/concat"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Rank_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Shape_2"
  op: "Shape"
  input: "Placeholder_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Sub_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Sub_1"
  op: "Sub"
  input: "softmax_cross_entropy_loss_1/Rank_2"
  input: "softmax_cross_entropy_loss_1/Sub_1/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Slice_1/begin"
  op: "Pack"
  input: "softmax_cross_entropy_loss_1/Sub_1"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Slice_1/size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Slice_1"
  op: "Slice"
  input: "softmax_cross_entropy_loss_1/Shape_2"
  input: "softmax_cross_entropy_loss_1/Slice_1/begin"
  input: "softmax_cross_entropy_loss_1/Slice_1/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/concat_1/values_0"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/concat_1/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/concat_1"
  op: "ConcatV2"
  input: "softmax_cross_entropy_loss_1/concat_1/values_0"
  input: "softmax_cross_entropy_loss_1/Slice_1"
  input: "softmax_cross_entropy_loss_1/concat_1/axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Reshape_1"
  op: "Reshape"
  input: "Placeholder_3"
  input: "softmax_cross_entropy_loss_1/concat_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/xentropy"
  op: "SoftmaxCrossEntropyWithLogits"
  input: "softmax_cross_entropy_loss_1/Reshape"
  input: "softmax_cross_entropy_loss_1/Reshape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Sub_2/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Sub_2"
  op: "Sub"
  input: "softmax_cross_entropy_loss_1/Rank"
  input: "softmax_cross_entropy_loss_1/Sub_2/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Slice_2/begin"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Slice_2/size"
  op: "Pack"
  input: "softmax_cross_entropy_loss_1/Sub_2"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Slice_2"
  op: "Slice"
  input: "softmax_cross_entropy_loss_1/Shape"
  input: "softmax_cross_entropy_loss_1/Slice_2/begin"
  input: "softmax_cross_entropy_loss_1/Slice_2/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Reshape_2"
  op: "Reshape"
  input: "softmax_cross_entropy_loss_1/xentropy"
  input: "softmax_cross_entropy_loss_1/Slice_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/assert_broadcastable/weights"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/assert_broadcastable/weights/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/assert_broadcastable/weights/rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/assert_broadcastable/values/shape"
  op: "Shape"
  input: "softmax_cross_entropy_loss_1/Reshape_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/assert_broadcastable/values/rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  op: "NoOp"
}
node {
  name: "softmax_cross_entropy_loss_1/ToFloat_1/x"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Mul"
  op: "Mul"
  input: "softmax_cross_entropy_loss_1/Reshape_2"
  input: "softmax_cross_entropy_loss_1/ToFloat_1/x"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Const"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Sum"
  op: "Sum"
  input: "softmax_cross_entropy_loss_1/Mul"
  input: "softmax_cross_entropy_loss_1/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/Equal/y"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/Equal"
  op: "Equal"
  input: "softmax_cross_entropy_loss_1/ToFloat_1/x"
  input: "softmax_cross_entropy_loss_1/num_present/Equal/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/zeros_like"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/ones_like/Shape"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/ones_like/Const"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/ones_like"
  op: "Fill"
  input: "softmax_cross_entropy_loss_1/num_present/ones_like/Shape"
  input: "softmax_cross_entropy_loss_1/num_present/ones_like/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/Select"
  op: "Select"
  input: "softmax_cross_entropy_loss_1/num_present/Equal"
  input: "softmax_cross_entropy_loss_1/num_present/zeros_like"
  input: "softmax_cross_entropy_loss_1/num_present/ones_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/shape"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/rank"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/shape"
  op: "Shape"
  input: "softmax_cross_entropy_loss_1/Reshape_2"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/rank"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success"
  op: "NoOp"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/Shape"
  op: "Shape"
  input: "softmax_cross_entropy_loss_1/Reshape_2"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  input: "^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/Const"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  input: "^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like"
  op: "Fill"
  input: "softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/Shape"
  input: "softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/broadcast_weights"
  op: "Mul"
  input: "softmax_cross_entropy_loss_1/num_present/Select"
  input: "softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present/Const"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/num_present"
  op: "Sum"
  input: "softmax_cross_entropy_loss_1/num_present/broadcast_weights"
  input: "softmax_cross_entropy_loss_1/num_present/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Const_1"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Sum_1"
  op: "Sum"
  input: "softmax_cross_entropy_loss_1/Sum"
  input: "softmax_cross_entropy_loss_1/Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Greater/y"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Greater"
  op: "Greater"
  input: "softmax_cross_entropy_loss_1/num_present"
  input: "softmax_cross_entropy_loss_1/Greater/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Equal/y"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Equal"
  op: "Equal"
  input: "softmax_cross_entropy_loss_1/num_present"
  input: "softmax_cross_entropy_loss_1/Equal/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/ones_like/Shape"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/ones_like/Const"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/ones_like"
  op: "Fill"
  input: "softmax_cross_entropy_loss_1/ones_like/Shape"
  input: "softmax_cross_entropy_loss_1/ones_like/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/Select"
  op: "Select"
  input: "softmax_cross_entropy_loss_1/Equal"
  input: "softmax_cross_entropy_loss_1/ones_like"
  input: "softmax_cross_entropy_loss_1/num_present"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/div"
  op: "RealDiv"
  input: "softmax_cross_entropy_loss_1/Sum_1"
  input: "softmax_cross_entropy_loss_1/Select"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/zeros_like"
  op: "Const"
  input: "^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "softmax_cross_entropy_loss_1/value"
  op: "Select"
  input: "softmax_cross_entropy_loss_1/Greater"
  input: "softmax_cross_entropy_loss_1/div"
  input: "softmax_cross_entropy_loss_1/zeros_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save_1/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "model"
      }
    }
  }
}
node {
  name: "save_1/SaveV2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 8
          }
        }
        string_val: "dense/bias"
        string_val: "dense/kernel"
        string_val: "dense_1/bias"
        string_val: "dense_1/kernel"
        string_val: "dense_2/bias"
        string_val: "dense_2/kernel"
        string_val: "dense_3/bias"
        string_val: "dense_3/kernel"
      }
    }
  }
}
node {
  name: "save_1/SaveV2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 8
          }
        }
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
      }
    }
  }
}
node {
  name: "save_1/SaveV2"
  op: "SaveV2"
  input: "save_1/Const"
  input: "save_1/SaveV2/tensor_names"
  input: "save_1/SaveV2/shape_and_slices"
  input: "dense/bias"
  input: "dense/kernel"
  input: "dense_1/bias"
  input: "dense_1/kernel"
  input: "dense_2/bias"
  input: "dense_2/kernel"
  input: "dense_3/bias"
  input: "dense_3/kernel"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save_1/control_dependency"
  op: "Identity"
  input: "save_1/Const"
  input: "^save_1/SaveV2"
  attr {
    key: "T"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@save_1/Const"
      }
    }
  }
}
node {
  name: "save_1/RestoreV2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "dense/bias"
      }
    }
  }
}
node {
  name: "save_1/RestoreV2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save_1/RestoreV2"
  op: "RestoreV2"
  input: "save_1/Const"
  input: "save_1/RestoreV2/tensor_names"
  input: "save_1/RestoreV2/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save_1/Assign"
  op: "Assign"
  input: "dense/bias"
  input: "save_1/RestoreV2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save_1/RestoreV2_1/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "dense/kernel"
      }
    }
  }
}
node {
  name: "save_1/RestoreV2_1/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save_1/RestoreV2_1"
  op: "RestoreV2"
  input: "save_1/Const"
  input: "save_1/RestoreV2_1/tensor_names"
  input: "save_1/RestoreV2_1/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save_1/Assign_1"
  op: "Assign"
  input: "dense/kernel"
  input: "save_1/RestoreV2_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save_1/RestoreV2_2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "dense_1/bias"
      }
    }
  }
}
node {
  name: "save_1/RestoreV2_2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save_1/RestoreV2_2"
  op: "RestoreV2"
  input: "save_1/Const"
  input: "save_1/RestoreV2_2/tensor_names"
  input: "save_1/RestoreV2_2/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save_1/Assign_2"
  op: "Assign"
  input: "dense_1/bias"
  input: "save_1/RestoreV2_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save_1/RestoreV2_3/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "dense_1/kernel"
      }
    }
  }
}
node {
  name: "save_1/RestoreV2_3/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save_1/RestoreV2_3"
  op: "RestoreV2"
  input: "save_1/Const"
  input: "save_1/RestoreV2_3/tensor_names"
  input: "save_1/RestoreV2_3/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save_1/Assign_3"
  op: "Assign"
  input: "dense_1/kernel"
  input: "save_1/RestoreV2_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save_1/RestoreV2_4/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "dense_2/bias"
      }
    }
  }
}
node {
  name: "save_1/RestoreV2_4/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save_1/RestoreV2_4"
  op: "RestoreV2"
  input: "save_1/Const"
  input: "save_1/RestoreV2_4/tensor_names"
  input: "save_1/RestoreV2_4/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save_1/Assign_4"
  op: "Assign"
  input: "dense_2/bias"
  input: "save_1/RestoreV2_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save_1/RestoreV2_5/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "dense_2/kernel"
      }
    }
  }
}
node {
  name: "save_1/RestoreV2_5/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save_1/RestoreV2_5"
  op: "RestoreV2"
  input: "save_1/Const"
  input: "save_1/RestoreV2_5/tensor_names"
  input: "save_1/RestoreV2_5/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save_1/Assign_5"
  op: "Assign"
  input: "dense_2/kernel"
  input: "save_1/RestoreV2_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save_1/RestoreV2_6/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "dense_3/bias"
      }
    }
  }
}
node {
  name: "save_1/RestoreV2_6/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save_1/RestoreV2_6"
  op: "RestoreV2"
  input: "save_1/Const"
  input: "save_1/RestoreV2_6/tensor_names"
  input: "save_1/RestoreV2_6/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save_1/Assign_6"
  op: "Assign"
  input: "dense_3/bias"
  input: "save_1/RestoreV2_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save_1/RestoreV2_7/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "dense_3/kernel"
      }
    }
  }
}
node {
  name: "save_1/RestoreV2_7/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save_1/RestoreV2_7"
  op: "RestoreV2"
  input: "save_1/Const"
  input: "save_1/RestoreV2_7/tensor_names"
  input: "save_1/RestoreV2_7/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save_1/Assign_7"
  op: "Assign"
  input: "dense_3/kernel"
  input: "save_1/RestoreV2_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save_1/restore_all"
  op: "NoOp"
  input: "^save_1/Assign"
  input: "^save_1/Assign_1"
  input: "^save_1/Assign_2"
  input: "^save_1/Assign_3"
  input: "^save_1/Assign_4"
  input: "^save_1/Assign_5"
  input: "^save_1/Assign_6"
  input: "^save_1/Assign_7"
}
node {
  name: "gradients_1/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients_1/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "gradients_1/Fill"
  op: "Fill"
  input: "gradients_1/Shape"
  input: "gradients_1/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/value_grad/zeros_like"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/value_grad/Select"
  op: "Select"
  input: "softmax_cross_entropy_loss_1/Greater"
  input: "gradients_1/Fill"
  input: "gradients_1/softmax_cross_entropy_loss_1/value_grad/zeros_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/value_grad/Select_1"
  op: "Select"
  input: "softmax_cross_entropy_loss_1/Greater"
  input: "gradients_1/softmax_cross_entropy_loss_1/value_grad/zeros_like"
  input: "gradients_1/Fill"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients_1/softmax_cross_entropy_loss_1/value_grad/Select"
  input: "^gradients_1/softmax_cross_entropy_loss_1/value_grad/Select_1"
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients_1/softmax_cross_entropy_loss_1/value_grad/Select"
  input: "^gradients_1/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/softmax_cross_entropy_loss_1/value_grad/Select"
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients_1/softmax_cross_entropy_loss_1/value_grad/Select_1"
  input: "^gradients_1/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/softmax_cross_entropy_loss_1/value_grad/Select_1"
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Shape"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/RealDiv"
  op: "RealDiv"
  input: "gradients_1/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency"
  input: "softmax_cross_entropy_loss_1/Select"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Sum"
  op: "Sum"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/RealDiv"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Reshape"
  op: "Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Sum"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Neg"
  op: "Neg"
  input: "softmax_cross_entropy_loss_1/Sum_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/RealDiv_1"
  op: "RealDiv"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Neg"
  input: "softmax_cross_entropy_loss_1/Select"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/RealDiv_2"
  op: "RealDiv"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/RealDiv_1"
  input: "softmax_cross_entropy_loss_1/Select"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/mul"
  op: "Mul"
  input: "gradients_1/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/RealDiv_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Sum_1"
  op: "Sum"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/mul"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Reshape_1"
  op: "Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Sum_1"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients_1/softmax_cross_entropy_loss_1/div_grad/Reshape"
  input: "^gradients_1/softmax_cross_entropy_loss_1/div_grad/Reshape_1"
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Reshape"
  input: "^gradients_1/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/softmax_cross_entropy_loss_1/div_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/Reshape_1"
  input: "^gradients_1/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/softmax_cross_entropy_loss_1/div_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape"
  op: "Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency"
  input: "gradients_1/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiples"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Sum_1_grad/Tile"
  op: "Tile"
  input: "gradients_1/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiples"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Select_grad/zeros_like"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Select_grad/Select"
  op: "Select"
  input: "softmax_cross_entropy_loss_1/Equal"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1"
  input: "gradients_1/softmax_cross_entropy_loss_1/Select_grad/zeros_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Select_grad/Select_1"
  op: "Select"
  input: "softmax_cross_entropy_loss_1/Equal"
  input: "gradients_1/softmax_cross_entropy_loss_1/Select_grad/zeros_like"
  input: "gradients_1/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients_1/softmax_cross_entropy_loss_1/Select_grad/Select"
  input: "^gradients_1/softmax_cross_entropy_loss_1/Select_grad/Select_1"
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients_1/softmax_cross_entropy_loss_1/Select_grad/Select"
  input: "^gradients_1/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/softmax_cross_entropy_loss_1/Select_grad/Select"
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients_1/softmax_cross_entropy_loss_1/Select_grad/Select_1"
  input: "^gradients_1/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/softmax_cross_entropy_loss_1/Select_grad/Select_1"
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Sum_grad/Reshape"
  op: "Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/Sum_1_grad/Tile"
  input: "gradients_1/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Sum_grad/Shape"
  op: "Shape"
  input: "softmax_cross_entropy_loss_1/Mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Sum_grad/Tile"
  op: "Tile"
  input: "gradients_1/softmax_cross_entropy_loss_1/Sum_grad/Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/Sum_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Shape"
  op: "Shape"
  input: "softmax_cross_entropy_loss_1/Reshape_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Shape"
  input: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/mul"
  op: "Mul"
  input: "gradients_1/softmax_cross_entropy_loss_1/Sum_grad/Tile"
  input: "softmax_cross_entropy_loss_1/ToFloat_1/x"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Sum"
  op: "Sum"
  input: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/mul"
  input: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Reshape"
  op: "Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Sum"
  input: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/mul_1"
  op: "Mul"
  input: "softmax_cross_entropy_loss_1/Reshape_2"
  input: "gradients_1/softmax_cross_entropy_loss_1/Sum_grad/Tile"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Sum_1"
  op: "Sum"
  input: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/mul_1"
  input: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1"
  op: "Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Sum_1"
  input: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Reshape"
  input: "^gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1"
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Reshape"
  input: "^gradients_1/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1"
  input: "^gradients_1/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present_grad/Reshape"
  op: "Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present_grad/Shape"
  op: "Shape"
  input: "softmax_cross_entropy_loss_1/num_present/broadcast_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present_grad/Tile"
  op: "Tile"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present_grad/Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1"
  op: "Shape"
  input: "softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul"
  op: "Mul"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present_grad/Tile"
  input: "softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum"
  op: "Sum"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape"
  op: "Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1"
  op: "Mul"
  input: "softmax_cross_entropy_loss_1/num_present/Select"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present_grad/Tile"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1"
  op: "Sum"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1"
  op: "Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape"
  input: "^gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1"
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape"
  input: "^gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1"
  input: "^gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/Sum"
  op: "Sum"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1"
  input: "gradients_1/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Reshape_2_grad/Shape"
  op: "Shape"
  input: "softmax_cross_entropy_loss_1/xentropy"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Reshape_2_grad/Reshape"
  op: "Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency"
  input: "gradients_1/softmax_cross_entropy_loss_1/Reshape_2_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/zeros_like"
  op: "ZerosLike"
  input: "softmax_cross_entropy_loss_1/xentropy:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims"
  op: "ExpandDims"
  input: "gradients_1/softmax_cross_entropy_loss_1/Reshape_2_grad/Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/xentropy_grad/mul"
  op: "Mul"
  input: "gradients_1/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims"
  input: "softmax_cross_entropy_loss_1/xentropy:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Reshape_grad/Shape"
  op: "Shape"
  input: "dense_4/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/softmax_cross_entropy_loss_1/Reshape_grad/Reshape"
  op: "Reshape"
  input: "gradients_1/softmax_cross_entropy_loss_1/xentropy_grad/mul"
  input: "gradients_1/softmax_cross_entropy_loss_1/Reshape_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients_1/dense_4/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "gradients_1/softmax_cross_entropy_loss_1/Reshape_grad/Reshape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "gradients_1/dense_4/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients_1/softmax_cross_entropy_loss_1/Reshape_grad/Reshape"
  input: "^gradients_1/dense_4/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "gradients_1/dense_4/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients_1/softmax_cross_entropy_loss_1/Reshape_grad/Reshape"
  input: "^gradients_1/dense_4/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/softmax_cross_entropy_loss_1/Reshape_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients_1/dense_4/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients_1/dense_4/BiasAdd_grad/BiasAddGrad"
  input: "^gradients_1/dense_4/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/dense_4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "gradients_1/dense_4/MatMul_grad/MatMul"
  op: "MatMul"
  input: "gradients_1/dense_4/BiasAdd_grad/tuple/control_dependency"
  input: "dense_3/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "gradients_1/dense_4/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "dense_3/Sigmoid"
  input: "gradients_1/dense_4/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "gradients_1/dense_4/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients_1/dense_4/MatMul_grad/MatMul"
  input: "^gradients_1/dense_4/MatMul_grad/MatMul_1"
}
node {
  name: "gradients_1/dense_4/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients_1/dense_4/MatMul_grad/MatMul"
  input: "^gradients_1/dense_4/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/dense_4/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "gradients_1/dense_4/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients_1/dense_4/MatMul_grad/MatMul_1"
  input: "^gradients_1/dense_4/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/dense_4/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "gradients_1/dense_3/Sigmoid_grad/SigmoidGrad"
  op: "SigmoidGrad"
  input: "dense_3/Sigmoid"
  input: "gradients_1/dense_4/MatMul_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients_1/dense_3/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "gradients_1/dense_3/Sigmoid_grad/SigmoidGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "gradients_1/dense_3/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients_1/dense_3/Sigmoid_grad/SigmoidGrad"
  input: "^gradients_1/dense_3/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "gradients_1/dense_3/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients_1/dense_3/Sigmoid_grad/SigmoidGrad"
  input: "^gradients_1/dense_3/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/dense_3/Sigmoid_grad/SigmoidGrad"
      }
    }
  }
}
node {
  name: "gradients_1/dense_3/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients_1/dense_3/BiasAdd_grad/BiasAddGrad"
  input: "^gradients_1/dense_3/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/dense_3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "gradients_1/dense_3/MatMul_grad/MatMul"
  op: "MatMul"
  input: "gradients_1/dense_3/BiasAdd_grad/tuple/control_dependency"
  input: "dense_2/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "gradients_1/dense_3/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "Placeholder_2"
  input: "gradients_1/dense_3/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "gradients_1/dense_3/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients_1/dense_3/MatMul_grad/MatMul"
  input: "^gradients_1/dense_3/MatMul_grad/MatMul_1"
}
node {
  name: "gradients_1/dense_3/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients_1/dense_3/MatMul_grad/MatMul"
  input: "^gradients_1/dense_3/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/dense_3/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "gradients_1/dense_3/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients_1/dense_3/MatMul_grad/MatMul_1"
  input: "^gradients_1/dense_3/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients_1/dense_3/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "GradientDescent_1/learning_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000475
      }
    }
  }
}
node {
  name: "GradientDescent_1/update_dense_2/kernel/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "dense_2/kernel"
  input: "GradientDescent_1/learning_rate"
  input: "gradients_1/dense_3/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent_1/update_dense_2/bias/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "dense_2/bias"
  input: "GradientDescent_1/learning_rate"
  input: "gradients_1/dense_3/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent_1/update_dense_3/kernel/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "dense_3/kernel"
  input: "GradientDescent_1/learning_rate"
  input: "gradients_1/dense_4/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent_1/update_dense_3/bias/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "dense_3/bias"
  input: "GradientDescent_1/learning_rate"
  input: "gradients_1/dense_4/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dense_3/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent_1"
  op: "NoOp"
  input: "^GradientDescent_1/update_dense_2/kernel/ApplyGradientDescent"
  input: "^GradientDescent_1/update_dense_2/bias/ApplyGradientDescent"
  input: "^GradientDescent_1/update_dense_3/kernel/ApplyGradientDescent"
  input: "^GradientDescent_1/update_dense_3/bias/ApplyGradientDescent"
}
node {
  name: "init_1"
  op: "NoOp"
  input: "^dense/kernel/Assign"
  input: "^dense/bias/Assign"
  input: "^dense_1/kernel/Assign"
  input: "^dense_1/bias/Assign"
  input: "^dense_2/kernel/Assign"
  input: "^dense_2/bias/Assign"
  input: "^dense_3/kernel/Assign"
  input: "^dense_3/bias/Assign"
}
versions {
  producer: 24
}
