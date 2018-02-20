package org.deeplearning4j.scalnet.layers

import org.deeplearning4j.nn.conf.{ GradientNormalization, Updater, layers }
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.conf.weightnoise.WeightNoise
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.IUpdater

class GravesLSTM(nIn: Int,
                 nOut: Int,
                 forgetGateBiasInit: Double = 1.0,
                 gateActivationFn: Activation = Activation.SIGMOID,
                 weightInitRecurrent: WeightInit = WeightInit.XAVIER,
                 distRecurrent: Distribution,
                 activationFn: Activation = Activation.IDENTITY,
                 weightInit: WeightInit = WeightInit.XAVIER,
                 biasInit: Double,
                 dist: Distribution,
                 l1: Double,
                 l2: Double,
                 l1Bias: Double,
                 l2Bias: Double,
                 updater: IUpdater,
                 biasUpdater: IUpdater,
                 weightNoise: WeightNoise,
                 gradientNormalization: GradientNormalization = GradientNormalization.None,
                 gradientNormalizationThreshold: Double = 1.0,
                 override val name: String = "")
    extends AbstractLSTM(
      forgetGateBiasInit,
      gateActivationFn,
      weightInitRecurrent,
      distRecurrent,
      activationFn,
      weightInit,
      biasInit,
      dist,
      l1,
      l2,
      l1Bias,
      l2Bias,
      updater,
      biasUpdater,
      weightNoise,
      gradientNormalization,
      gradientNormalizationThreshold
    ) {

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new layers.EmbeddingLayer.Builder().build()

  override def inputShape: List[Int] = List(nIn, nOut)

  override def outputShape: List[Int] = List(nOut, nIn)

}

object GravesLSTM {
  def apply(nIn: Int,
            nOut: Int,
            forgetGateBiasInit: Double = 1.0,
            gateActivationFn: Activation = Activation.SIGMOID,
            weightInitRecurrent: WeightInit = WeightInit.XAVIER,
            distRecurrent: Distribution = null,
            activationFn: Activation = Activation.IDENTITY,
            weightInit: WeightInit = WeightInit.XAVIER,
            biasInit: Double = Double.NaN,
            dist: Distribution = null,
            l1: Double = Double.NaN,
            l2: Double = Double.NaN,
            l1Bias: Double = Double.NaN,
            l2Bias: Double = Double.NaN,
            updater: IUpdater = null,
            biasUpdater: IUpdater = null,
            weightNoise: WeightNoise = null,
            gradientNormalization: GradientNormalization = GradientNormalization.None,
            gradientNormalizationThreshold: Double = 1.0): GravesLSTM =
    new GravesLSTM(
      nIn,
      nOut,
      forgetGateBiasInit,
      gateActivationFn,
      weightInitRecurrent,
      distRecurrent,
      activationFn,
      weightInit,
      biasInit,
      dist,
      l1,
      l2,
      l1Bias,
      l2Bias,
      updater,
      biasUpdater,
      weightNoise,
      gradientNormalization,
      gradientNormalizationThreshold
    )
}
