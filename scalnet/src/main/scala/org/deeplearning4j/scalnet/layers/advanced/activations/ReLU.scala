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

package org.deeplearning4j.scalnet.layers.advanced.activation

import org.deeplearning4j.nn.conf.layers.{ ActivationLayer => JActivationLayer }
import org.deeplearning4j.scalnet.layers.core.Layer
import org.nd4j.linalg.activations.impl.ActivationReLU

/**
  * ReLU layer
  *
  * @author Max Pumperla
  */
class ReLU(nOut: Option[List[Int]], nIn: Option[List[Int]], override val name: String = "") extends Layer {

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new JActivationLayer.Builder()
      .activation(new ActivationReLU())
      .name(name)
      .build()

  override val outputShape: List[Int] = nOut.getOrElse(List(0))
  override val inputShape: List[Int] = nIn.getOrElse(List(0))

  override def reshapeInput(newIn: List[Int]): ReLU =
    new ReLU(Some(newIn), Some(newIn), name)
}

object ReLU {
  def apply(nOut: Int = 0, nIn: Int = 0, name: String = ""): ReLU =
    new ReLU(Some(List(nOut)), Some(List(nIn)), name)
}
