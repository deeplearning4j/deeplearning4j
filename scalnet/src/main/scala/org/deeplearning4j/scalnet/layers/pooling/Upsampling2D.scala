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

package org.deeplearning4j.scalnet.layers.pooling

import org.deeplearning4j.nn.conf.layers.{Upsampling2D => JUpsampling2D}
import org.deeplearning4j.scalnet.layers.core.Layer
/**
  * 2D upsampling layer
  *
  * @author Max Pumperla
  */
class Upsampling2D(size: List[Int],
                   nChannels: Int = 0,
                   nIn: Option[List[Int]] = None,
                   override val name: String = "")
  extends Upsampling(dimension = 2, size, nChannels, nIn, name) with Layer {
  if (size.length != 2) {
    throw new IllegalArgumentException("Size must be length 2.")
  }

  override def reshapeInput(nIn: List[Int]): Upsampling2D =
    new Upsampling2D(size, nChannels, Some(nIn), name)


  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new JUpsampling2D.Builder()
      .size(size.toArray)
      .name(name)
      .build()
}

object Upsampling2D {
  def apply(size: List[Int],
            nChannels: Int = 0,
            nIn: Option[List[Int]] = None): Upsampling2D =
    new Upsampling2D(size, nChannels, nIn)
}
