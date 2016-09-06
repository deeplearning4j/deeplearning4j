package org.deeplearning4s.optimizers

import org.deeplearning4j.nn.api.OptimizationAlgorithm

/**
  * Optimizers.
  */

sealed class Optimizer(val optimizationAlgorithm: OptimizationAlgorithm, val lr: Double = 1e-1)
case class SGD(override val lr: Double = 1e-1, val momentum: Double = Double.NaN, val nesterov: Boolean = false)
  extends Optimizer(optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
