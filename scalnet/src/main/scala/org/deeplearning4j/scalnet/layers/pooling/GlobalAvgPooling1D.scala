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

import org.deeplearning4j.nn.conf.layers.{GlobalPoolingLayer, PoolingType}
import org.deeplearning4j.scalnet.layers.core.{Layer, Node}

/**
  * 1D global avg pooling layer.
  *
  * @author Max Pumperla
  */
class GlobalAvgPooling1D(nIn: Option[List[Int]] = None,
                         override val name: String = null)
  extends Node with Layer {

  override def inputShape: List[Int] = nIn.getOrElse(List(0))

  override def outputShape: List[Int] = {
    val nOutChannels: Int =
      if (inputShape.nonEmpty) inputShape.last
      else 0
    if (inputShape.lengthCompare(2) == 0) {
      List[Int](inputShape.head, nOutChannels)
    } else if (nOutChannels > 0) List(nOutChannels)
    else List()
  }

  override def reshapeInput(nIn: List[Int]): GlobalAvgPooling1D =
    new GlobalAvgPooling1D(Some(nIn), name)

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new GlobalPoolingLayer.Builder()
      .poolingType(PoolingType.AVG)
      .name(name)
      .build()
}

object GlobalAvgPooling1D {
  def apply(nIn: Option[List[Int]] = None,
            name: String = null): GlobalAvgPooling1D =
    new GlobalAvgPooling1D(nIn, name)
}


