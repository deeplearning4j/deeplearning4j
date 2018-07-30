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

import org.deeplearning4j.nn.conf.layers.{ Subsampling1DLayer, SubsamplingLayer }
import org.deeplearning4j.scalnet.layers.convolutional.Convolution
import org.deeplearning4j.scalnet.layers.core.Layer

/**
  * 1D max pooling layer in neural net architectures.
  *
  * @author Max Pumperla
  */
class MaxPooling1D(kernelSize: List[Int],
                   stride: List[Int] = List(1),
                   padding: List[Int] = List(0),
                   dilation: List[Int] = List(1),
                   nIn: Option[List[Int]] = None,
                   override val name: String = "")
    extends Convolution(dimension = 1, kernelSize, stride, padding, dilation, 0, nIn, 0)
    with Layer {
  if (kernelSize.length != 1 || stride.length != 1 || padding.length != 1 || dilation.length != 1) {
    throw new IllegalArgumentException("Kernel, stride, padding and dilation lists must all be length 1.")
  }

  override def reshapeInput(nIn: List[Int]): MaxPooling1D =
    new MaxPooling1D(kernelSize, stride, padding, dilation, Some(nIn), name)

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new Subsampling1DLayer.Builder(SubsamplingLayer.PoolingType.MAX)
      .kernelSize(kernelSize.head)
      .stride(stride.head)
      .name(name)
      .build()
}

object MaxPooling1D {
  def apply(kernelSize: List[Int],
            stride: List[Int] = List(1, 1),
            padding: List[Int] = List(0, 0),
            dilation: List[Int] = List(1, 1),
            nIn: Option[List[Int]] = None,
            name: String = null): MaxPooling1D =
    new MaxPooling1D(kernelSize, stride, padding, dilation, nIn, name)
}
