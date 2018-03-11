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

import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.scalnet.layers.convolutional.Convolution
import org.deeplearning4j.scalnet.layers.core.Layer

/**
  * 2D max pooling in neural net architectures.
  *
  * @author David Kale
  */
class MaxPooling2D(kernelSize: List[Int],
                   stride: List[Int] = List(1, 1),
                   padding: List[Int] = List(0, 0),
                   nIn: Option[List[Int]] = None,
                   override val name: String = null)
    extends Convolution(kernelSize, stride, padding, 0, nIn)
    with Layer {
  if (kernelSize.length != 2 || stride.length != 2 || padding.length != 2) {
    throw new IllegalArgumentException("Kernel, stride, padding lists must all be length 2.")
  }

  override def reshapeInput(nIn: List[Int]): MaxPooling2D =
    new MaxPooling2D(kernelSize, stride, padding, Some(nIn), name)

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
      .kernelSize(kernelSize.head, kernelSize.last)
      .stride(stride.head, stride.last)
      .name(name)
      .build()
}

object MaxPooling2D {
  def apply(kernelSize: List[Int],
            stride: List[Int] = List(1, 1),
            padding: List[Int] = List(0, 0),
            nIn: Option[List[Int]] = None,
            name: String = null): MaxPooling2D =
    new MaxPooling2D(kernelSize, stride, padding, nIn, name)
}
