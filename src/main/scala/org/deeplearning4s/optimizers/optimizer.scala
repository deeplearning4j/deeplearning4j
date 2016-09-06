package org.deeplearning4s.optimizers

import org.deeplearning4j.nn.api.OptimizationAlgorithm

/**
  * Optimizers.
  */

sealed class Optimizer(val optimizationAlgorithm: OptimizationAlgorithm, lr: Double = 0.01)

case class SGD(lr: Double = 0.01)
  extends Optimizer(optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, lr = lr)
