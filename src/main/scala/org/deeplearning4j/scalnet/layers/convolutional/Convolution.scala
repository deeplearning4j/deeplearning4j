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

package org.deeplearning4j.scalnet.layers.convolutional

import org.deeplearning4j.nn.layers.convolution.KernelValidationUtil
import org.deeplearning4j.scalnet.layers.Node


/**
  * Base class for convolutional layers.
  *
  * @author David Kale
  */
abstract class Convolution(
    protected val kernelSize: List[Int],
    protected val stride: List[Int],
    protected val padding: List[Int],
    nChannels: Int = 0,
    protected val nFilter: Int = 0)
  extends Node {
  inputShape = List(nChannels)
  if (kernelSize.length != stride.length || kernelSize.length != padding.length)
    throw new IllegalArgumentException("Kernel, stride, and padding must all have same shape.")

  override def outputShape: List[Int] = {
    val nOutChannels: Int = if (nFilter > 0) nFilter else if (inputShape.nonEmpty) inputShape.last else 0
    if (inputShape.length == 3) {
      KernelValidationUtil.validateShapes(inputShape.head, inputShape.tail.head, kernelSize.head, kernelSize.tail.head,
        stride.head, stride.tail.head, padding.head, padding.tail.head)
      List[List[Int]](inputShape.init, kernelSize, padding, stride)
        .transpose.map(x => (x.head - x(1) + 2 * x(2)) / x(3) + 1) :+ nOutChannels
    } else if (nOutChannels > 0) List(nOutChannels)
    else List()
  }
}
