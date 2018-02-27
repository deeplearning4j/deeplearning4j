package org.deeplearning4j.scalnet.layers

import org.deeplearning4j.nn.conf.layers.{OutputLayer => JOutputLayer}
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * Extension of base layer, used to construct a DL4J OutputLayer after compilation.
  * OutputLayer has an output object and the ability to return an OutputLayer version
  * of itself, by providing a loss function.
  *
  * @author Max Pumperla
  */
trait OutputLayer extends Layer {
  def output: Output
  def toOutputLayer(lossFunction: LossFunctions.LossFunction): OutputLayer
}
