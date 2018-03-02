package org.deeplearning4j.scalnet.layers

import org.deeplearning4j.nn.conf.layers
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation

class GravesLSTM(nIn: Int,
                 nOut: Int,
                 activation: Activation,
                 forgetGateBiasInit: Double,
                 gateActivation: Activation,
                 weightInit: WeightInit,
                 dropOut: Double,
                 override val name: String = "")
    extends AbstractLSTM {

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new layers.GravesLSTM.Builder()
      .nIn(nIn)
      .nOut(nOut)
      .activation(activation)
      .forgetGateBiasInit(forgetGateBiasInit)
      .gateActivationFunction(gateActivation)
      .weightInit(weightInit)
      .dropOut(dropOut)
      .name(name)
      .build()

  override def inputShape: List[Int] = List(nIn, nOut)

  override def outputShape: List[Int] = List(nOut, nIn)

}

object GravesLSTM {
  def apply(nIn: Int,
            nOut: Int,
            activation: Activation = Activation.IDENTITY,
            forgetGateBiasInit: Double = 1.0,
            gateActivationFn: Activation = Activation.SIGMOID,
            weightInit: WeightInit = WeightInit.XAVIER,
            dropOut: Double = 0.0): GravesLSTM =
    new GravesLSTM(
      nIn,
      nOut,
      activation,
      forgetGateBiasInit,
      gateActivationFn,
      weightInit,
      dropOut
    )
}
