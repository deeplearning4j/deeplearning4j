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

package org.deeplearning4j.scalnet.layers.recurrent

import org.deeplearning4j.nn.conf.layers
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional.Mode
import org.deeplearning4j.scalnet.layers.core.{ Layer, WrapperLayer }

class Bidirectional(layer: Layer, mode: Mode, override val name: String = "") extends WrapperLayer {

  val underlying: Layer = layer

  override def compile: layers.Layer = new layers.recurrent.Bidirectional(mode, underlying.compile)

}

object Bidirectional {

  val CONCAT = Mode.CONCAT
  val ADD = Mode.ADD
  val MUL = Mode.MUL
  val AVERAGE = Mode.AVERAGE

  def apply(layer: Layer, mode: Mode = Mode.CONCAT): Bidirectional = new Bidirectional(layer, mode)
}
