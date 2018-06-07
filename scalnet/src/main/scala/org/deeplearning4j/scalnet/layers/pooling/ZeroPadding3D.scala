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

import org.deeplearning4j.nn.conf.layers.{ZeroPadding3DLayer, ZeroPaddingLayer}
import org.deeplearning4j.scalnet.layers.core.{Layer, Node}

/**
  * 3D zero padding layer
  *
  * @author Max Pumperla
  */
class ZeroPadding3D(padLeftD: Int,
                    padRightD: Int,
                    padLeftH: Int,
                    padRightH: Int,
                    padLeftW: Int,
                    padRightW: Int,
                    nIn: List[Int],
                    override val name: String = "")
  extends Node with Layer {

  override def inputShape: List[Int] = nIn

  override def outputShape: List[Int] = {
    val nOutChannels: Int =
      if (inputShape.nonEmpty) inputShape.last
      else 0
    if (inputShape.lengthCompare(4) == 0) {
      List[Int](inputShape.head + padLeftD + padRightD,
        inputShape(1) + padLeftH + padRightH,
        inputShape(2) + padLeftW + padRightW,
        nOutChannels)
    } else if (nOutChannels > 0) List(nOutChannels)
    else List()
  }

  override def reshapeInput(nIn: List[Int]): ZeroPadding3D =
    new ZeroPadding3D(padLeftD, padRightD, padLeftH, padRightH, padLeftW, padRightW, nIn, name)


  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new ZeroPadding3DLayer.Builder(padLeftD, padRightD, padLeftH, padRightH, padLeftW, padRightW)
      .name(name)
      .build()
}

object ZeroPadding3D {
  def apply(padLeftD: Int,
            padRightD: Int,
            padLeftH: Int,
            padRightH: Int,
            padLeftW: Int,
            padRightW: Int,
            nIn: List[Int],
            name: String): ZeroPadding3D =
    new ZeroPadding3D(padLeftD, padRightD, padLeftH, padRightH, padLeftW, padRightW, nIn, name)
}


