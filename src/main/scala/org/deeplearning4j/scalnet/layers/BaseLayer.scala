package org.deeplearning4j.scalnet.layers

import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.conf.weightnoise.WeightNoise
import org.deeplearning4j.nn.conf.{GradientNormalization, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.IUpdater

abstract class BaseLayer(activationFn: Activation = Activation.IDENTITY,
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
    extends Layer
