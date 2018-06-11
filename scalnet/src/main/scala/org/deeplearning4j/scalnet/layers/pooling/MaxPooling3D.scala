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

import org.deeplearning4j.nn.conf.layers.Subsampling3DLayer
import org.deeplearning4j.scalnet.layers.convolutional.Convolution
import org.deeplearning4j.scalnet.layers.core.Layer

/**
  * 3D max pooling layer in neural net architectures.
  *
  * @author Max Pumperla
  */
class MaxPooling3D(kernelSize: List[Int],
                   stride: List[Int] = List(1, 1, 1),
                   padding: List[Int] = List(0, 0, 0),
                   dilation: List[Int] = List(1, 1, 1),
                   nIn: Option[List[Int]] = None,
                   override val name: String = "")
    extends Convolution(dimension = 3, kernelSize, stride, padding, dilation, 0, nIn, 0)
    with Layer {
  if (kernelSize.length != 3 || stride.length != 3 || padding.length != 3 || dilation.length != 3) {
    throw new IllegalArgumentException("Kernel, stride, padding and dilation lists must all be length 3.")
  }

  override def reshapeInput(nIn: List[Int]): MaxPooling3D =
    new MaxPooling3D(kernelSize, stride, padding, dilation, Some(nIn), name)

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new Subsampling3DLayer.Builder()
      .poolingType(Subsampling3DLayer.PoolingType.MAX)
      .kernelSize(kernelSize.head, kernelSize(1), kernelSize(2))
      .stride(stride.head, stride(1), stride(2))
      .name(name)
      .build()
}

object MaxPooling3D {
  def apply(kernelSize: List[Int],
            stride: List[Int] = List(1, 1, 1),
            padding: List[Int] = List(0, 0, 0),
            dilation: List[Int] = List(1, 1, 1),
            nIn: Option[List[Int]] = None,
            name: String = null): MaxPooling3D =
    new MaxPooling3D(kernelSize, stride, padding, dilation, nIn, name)
}
