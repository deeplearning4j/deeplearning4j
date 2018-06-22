## Supported initializers

DL4J supports all available [Keras initializers](https://keras.io/initializers), namely:

* <i class="fa fa-check-square-o"></i> Zeros
* <i class="fa fa-check-square-o"></i> Ones
* <i class="fa fa-check-square-o"></i> Constant
* <i class="fa fa-check-square-o"></i> RandomNormal
* <i class="fa fa-check-square-o"></i> RandomUniform
* <i class="fa fa-check-square-o"></i> TruncatedNormal
* <i class="fa fa-check-square-o"></i> VarianceScaling
* <i class="fa fa-check-square-o"></i> Orthogonal
* <i class="fa fa-check-square-o"></i> Identity
* <i class="fa fa-check-square-o"></i> lecun_uniform
* <i class="fa fa-check-square-o"></i> lecun_normal
* <i class="fa fa-check-square-o"></i> glorot_normal
* <i class="fa fa-check-square-o"></i> glorot_uniform
* <i class="fa fa-check-square-o"></i> he_normal
* <i class="fa fa-check-square-o"></i> he_uniform

The mapping of Keras to DL4J initializers can be found in [KerasInitilizationUtils](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasInitilizationUtils.java).