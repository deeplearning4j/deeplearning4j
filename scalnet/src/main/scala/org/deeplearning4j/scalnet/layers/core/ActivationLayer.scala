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

package org.deeplearning4j.scalnet.layers.core

import org.deeplearning4j.nn.conf.layers.{ActivationLayer => JActivationLayer}
import org.nd4j.linalg.activations.Activation


/**
  * Activation layer
  *
  * @author Max Pumperla
  */
class ActivationLayer(activation: Activation,
                      nOut: Option[List[Int]],
                      nIn: Option[List[Int]],
                 override val name: String)
  extends Layer {

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new JActivationLayer.Builder()
      .activation(activation)
      .name(name)
      .build()

  override val outputShape: List[Int] = nOut.getOrElse(List(0))
  override val inputShape: List[Int] =  nIn.getOrElse(List(0))

  override def reshapeInput(newIn: List[Int]): ActivationLayer =
    new ActivationLayer(activation, Some(newIn), Some(newIn),  name)
}

object ActivationLayer {
  def apply(activation: Activation,
            nOut: Int = 0,
            nIn: Int = 0,
            name: String = ""): ActivationLayer =
    new ActivationLayer(activation, Some(List(nOut)), Some(List(nIn)), name)
}



