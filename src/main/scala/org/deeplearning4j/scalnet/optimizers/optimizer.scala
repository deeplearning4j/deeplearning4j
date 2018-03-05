/*
 * Copyright 2016 Skymind
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.deeplearning4j.scalnet.optimizers

import org.deeplearning4j.nn.api.OptimizationAlgorithm

/**
  * Optimizers for neural nets.
  *
  * @author David Kale
  */
sealed class Optimizer(val optimizationAlgorithm: OptimizationAlgorithm, val lr: Double = 1e-1)
case class SGD(override val lr: Double = 1e-1, momentum: Double = Double.NaN, nesterov: Boolean = false)
    extends Optimizer(optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
