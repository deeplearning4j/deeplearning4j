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

package org.deeplearning4j.scalnet.layers.convolutional

import org.deeplearning4j.nn.conf.layers.ZeroPadding1DLayer
import org.deeplearning4j.scalnet.layers.core.{ Layer, Node }

/**
  * 1D zero padding layer
  *
  * @author Max Pumperla
  */
class ZeroPadding1D(padLeftH: Int, padRightH: Int, nIn: List[Int], override val name: String = "")
    extends Node
    with Layer {

  override def inputShape: List[Int] = nIn

  override def outputShape: List[Int] = {
    val nOutChannels: Int =
      if (inputShape.nonEmpty) inputShape.last
      else 0
    if (inputShape.lengthCompare(2) == 0) {
      List[Int](inputShape.head + padLeftH + padRightH, nOutChannels)
    } else if (nOutChannels > 0) List(nOutChannels)
    else List()
  }

  override def reshapeInput(nIn: List[Int]): ZeroPadding1D =
    new ZeroPadding1D(padLeftH, padRightH, nIn, name)

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new ZeroPadding1DLayer.Builder(padLeftH, padRightH)
      .name(name)
      .build()
}

object ZeroPadding1D {
  def apply(padLeftH: Int, padRightH: Int, nIn: List[Int], name: String): ZeroPadding1D =
    new ZeroPadding1D(padLeftH, padRightH, nIn, name)
}
