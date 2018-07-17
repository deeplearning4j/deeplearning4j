/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.scalnet.layers.pooling

import org.deeplearning4j.nn.conf.layers.{ GlobalPoolingLayer, PoolingType }
import org.deeplearning4j.scalnet.layers.core.{ Layer, Node }

/**
  * 2D global avg pooling layer.
  *
  * @author Max Pumperla
  */
class GlobalAvgPooling2D(nIn: Option[List[Int]] = None, override val name: String = null) extends Node with Layer {

  override def inputShape: List[Int] = nIn.getOrElse(List(0))

  override def outputShape: List[Int] = {
    val nOutChannels: Int =
      if (inputShape.nonEmpty) inputShape.last
      else 0
    if (inputShape.lengthCompare(3) == 0) {
      List[Int](inputShape.head, nOutChannels)
    } else if (nOutChannels > 0) List(nOutChannels)
    else List()
  }

  override def reshapeInput(nIn: List[Int]): GlobalAvgPooling2D =
    new GlobalAvgPooling2D(Some(nIn), name)

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new GlobalPoolingLayer.Builder()
      .poolingType(PoolingType.AVG)
      .name(name)
      .build()
}

object GlobalAvgPooling2D {
  def apply(nIn: Option[List[Int]] = None, name: String = null): GlobalAvgPooling2D =
    new GlobalAvgPooling2D(nIn, name)
}
