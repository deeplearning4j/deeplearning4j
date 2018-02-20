package org.deeplearning4j.scalnet.layers
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.conf.weightnoise.WeightNoise
import org.deeplearning4j.nn.conf.{ GradientNormalization, Updater, layers }
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.IUpdater

class EmbeddingLayer(nIn: Int,
                     nOut: Int,
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
                     hasBias: Boolean = true,
                     override val name: String = "")
    extends FeedForwardLayer(activationFn,
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
                             gradientNormalizationThreshold) {

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new layers.EmbeddingLayer.Builder()
      .nIn(nIn)
      .nOut(nOut)
      .activation(activationFn)
      .weightInit(weightInit)
      .biasInit(biasInit)
      .dist(dist)
      .l1(l1)
      .l2(l2)
      .l1Bias(l1Bias)
      .l2Bias(l2Bias)
      .updater(updater)
      .biasUpdater(biasUpdater)
      .weightNoise(weightNoise)
      .gradientNormalization(gradientNormalization)
      .gradientNormalizationThreshold(gradientNormalizationThreshold)
      .hasBias(hasBias)
      .build()

  override def inputShape: List[Int] = List(nIn, nOut)

  override def outputShape: List[Int] = List(nOut, nIn)
}

object EmbeddingLayer {
  def apply(nIn: Int,
            nOut: Int,
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
            gradientNormalizationThreshold: Double = 1.0,
            hasBias: Boolean = true): EmbeddingLayer =
    new EmbeddingLayer(
      nIn,
      nOut,
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
      gradientNormalizationThreshold,
      hasBias
    )
}
