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


abstract class ConvolutionBase(
  protected val _kernelSize: List[Int],
  protected val _stride: List[Int],
  protected val _padding: List[Int])
  extends Node {
  protected var _nFilter = 0

  if (_kernelSize.length != _stride.length || _kernelSize.length != _padding.length)
    throw new IllegalArgumentException("Kernel, stride, and padding must all have same shape.")

  override def inputShape_=(newInputShape: List[Int]): Unit = {
    if (newInputShape.length != _kernelSize.length + 1)
      throw new IllegalArgumentException("Input dimensions must match kernel dimensions plus one.")
    _inputShape = newInputShape
  }

  override def outputShape: List[Int] = {
    if (inputShape.isEmpty)
      throw new IllegalStateException("Input shape has not been initialized.")

    KernelValidationUtil.validateShapes(inputShape.head, inputShape(1), _kernelSize.head, _kernelSize(1),
      _stride.head, _stride(1), _padding.head, _padding(1))

    var nFilter: Int = if (_nFilter > 0) _nFilter else inputShape.last
    List[List[Int]](inputShape.init, _kernelSize, _padding, _stride)
      .transpose.map(x => (x(0)-x(1) + 2*x(2)) / x(3) + 1) :+ nFilter
  }
}
