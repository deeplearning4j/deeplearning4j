package org.deeplearning4j.scalnet.layers

import org.deeplearning4j.nn.conf.layers
import org.deeplearning4j.nn.conf.layers.{ OutputLayer => JOutputLayer }
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

class RnnOutputLayer(nIn: Int,
                     nOut: Int,
                     activation: Activation,
                     loss: Option[LossFunction],
                     weightInit: WeightInit,
                     dropOut: Double,
                     override val name: String = "")
    extends BaseOutputLayer
    with OutputLayer {

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    Option(output.lossFunction) match {
      case None =>
        new layers.RnnOutputLayer.Builder()
          .nIn(nIn)
          .nOut(nOut)
          .activation(activation)
          .weightInit(weightInit)
          .dropOut(dropOut)
          .name(name)
          .build()
      case _ =>
        new layers.RnnOutputLayer.Builder(output.lossFunction)
          .nIn(nIn)
          .nOut(nOut)
          .activation(activation)
          .weightInit(weightInit)
          .lossFunction(output.lossFunction)
          .dropOut(dropOut)
          .name(name)
          .build()
    }

  override val inputShape: List[Int] = List(nIn, nOut)

  override val outputShape: List[Int] = List(nOut, nIn)

  override val output: Output = Output(isOutput = loss.isDefined, lossFunction = loss.orNull)

  override def toOutputLayer(lossFunction: LossFunctions.LossFunction): OutputLayer =
    new RnnOutputLayer(
      nIn,
      nOut,
      activation,
      Some(lossFunction),
      weightInit,
      dropOut
    )
}

object RnnOutputLayer {
  def apply(nIn: Int,
            nOut: Int,
            activation: Activation,
            loss: Option[LossFunction] = None,
            weightInit: WeightInit = WeightInit.XAVIER,
            dropOut: Double = 0.0): RnnOutputLayer =
    new RnnOutputLayer(
      nIn,
      nOut,
      activation,
      loss,
      weightInit,
      dropOut
    )

}
