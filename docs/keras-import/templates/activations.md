## Available activations

We support all [Keras activation functions](https://keras.io/activations), namely:

* <i class="fa fa-check-square-o"></i> softmax
* <i class="fa fa-check-square-o"></i> elu
* <i class="fa fa-check-square-o"></i> selu
* <i class="fa fa-check-square-o"></i> softplus
* <i class="fa fa-check-square-o"></i> softsign
* <i class="fa fa-check-square-o"></i> relu
* <i class="fa fa-check-square-o"></i> tanh
* <i class="fa fa-check-square-o"></i> sigmoid
* <i class="fa fa-check-square-o"></i> hard_sigmoid
* <i class="fa fa-check-square-o"></i> linear

The mapping of Keras to DL4J activation functions is defined in [KerasActivationUtils](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasActivationUtils.java)
