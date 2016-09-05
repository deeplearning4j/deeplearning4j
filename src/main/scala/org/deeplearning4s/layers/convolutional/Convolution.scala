/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4s.layers.convolutional

import org.deeplearning4j.nn.layers.convolution.KernelValidationUtil
import org.deeplearning4s.layers.Node


abstract class Convolution(
  protected val kernelSize: List[Int],
  protected val stride: List[Int],
  protected val padding: List[Int],
  protected val nFilter: Int = 0)
  extends Node {

  if (kernelSize.length != stride.length || kernelSize.length != padding.length)
    throw new IllegalArgumentException("Kernel, stride, and padding must all have same shape.")

  override def outputShape: List[Int] = {
    if (inputShape.isEmpty)
      throw new IllegalStateException("Input shape has not been initialized.")

    KernelValidationUtil.validateShapes(inputShape.head, inputShape(1), kernelSize.head, kernelSize(1),
      stride.head, stride(1), padding.head, padding(1))

    var nOutChannels: Int = if (nFilter > 0) nFilter else inputShape.last
    List[List[Int]](inputShape.init, kernelSize, padding, stride)
      .transpose.map(x => (x(0)-x(1) + 2*x(2)) / x(3) + 1) :+ nOutChannels
  }
}
