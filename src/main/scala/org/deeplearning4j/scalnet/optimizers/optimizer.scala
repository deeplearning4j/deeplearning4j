package org.deeplearning4j.scalnet.optimizers

import org.deeplearning4j.nn.api.OptimizationAlgorithm

/**
  * Optimizers for neural nets.
  *
  * @author David Kale
  */
sealed class Optimizer(val optimizationAlgorithm: OptimizationAlgorithm, val lr: Double = 1e-1)
case class SGD(override val lr: Double = 1e-1,
               momentum: Double = Double.NaN,
               nesterov: Boolean = false)
    extends Optimizer(optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
