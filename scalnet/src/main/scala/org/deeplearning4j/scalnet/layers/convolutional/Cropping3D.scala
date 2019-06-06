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
package org.deeplearning4j.scalnet.layers.convolutional

import org.deeplearning4j.nn.conf.layers.convolutional.{ Cropping3D => JCropping3D }
import org.deeplearning4j.scalnet.layers.core.{ Layer, Node }

/**
  * 3D cropping layer
  *
  * @author Max Pumperla
  */
class Cropping3D(cropLeftD: Int,
                 cropRightD: Int,
                 cropLeftH: Int,
                 cropRightH: Int,
                 cropLeftW: Int,
                 cropRightW: Int,
                 nIn: List[Int],
                 override val name: String = "")
    extends Node
    with Layer {

  override def inputShape: List[Int] = nIn

  override def outputShape: List[Int] = {
    val nOutChannels: Int =
      if (inputShape.nonEmpty) inputShape.last
      else 0
    if (inputShape.lengthCompare(4) == 0) {
      List[Int](inputShape.head - cropLeftD - cropRightD,
                inputShape(1) - cropLeftH - cropRightH,
                inputShape(2) - cropLeftW - cropRightW,
                nOutChannels)
    } else if (nOutChannels > 0) List(nOutChannels)
    else List()
  }

  override def reshapeInput(nIn: List[Int]): Cropping3D =
    new Cropping3D(cropLeftD, cropRightD, cropLeftH, cropRightH, cropLeftW, cropRightW, nIn, name)

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new JCropping3D.Builder(cropLeftD, cropRightD, cropLeftH, cropRightH, cropLeftW, cropRightW)
      .name(name)
      .build()
}

object Cropping3D {
  def apply(cropLeftD: Int,
            cropRightD: Int,
            cropLeftH: Int,
            cropRightH: Int,
            cropLeftW: Int,
            cropRightW: Int,
            nIn: List[Int],
            name: String): Cropping3D =
    new Cropping3D(cropLeftD, cropRightD, cropLeftH, cropRightH, cropLeftW, cropRightW, nIn, name)
}
