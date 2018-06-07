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

package org.deeplearning4j.scalnet.layers.convolutional

import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException
import org.deeplearning4j.scalnet.layers.core.Node
import org.deeplearning4j.util.ConvolutionUtils

/**
  * Base class for convolutional layers.
  *
  * @author David Kale, Max Pumperla
  */
abstract class Convolution(protected val dimension: Int,
                           protected val kernelSize: List[Int],
                           protected val stride: List[Int],
                           protected val padding: List[Int],
                           protected val dilation: List[Int],
                           protected val nChannels: Int = 0,
                           protected val nIn: Option[List[Int]] = None,
                           protected val nFilter: Int = 0)
  extends Node {

  override def inputShape: List[Int] = nIn.getOrElse(List(nChannels))

  if (kernelSize.lengthCompare(dimension) != 0
    || kernelSize.lengthCompare(stride.length) != 0
    || kernelSize.lengthCompare(padding.length) != 0
    || kernelSize.lengthCompare(dilation.length) != 0) {
    throw new IllegalArgumentException("Kernel, stride, dilation and padding must all have same shape.")
  }

  private def validateShapes(dimension: Int,
                             inShape: List[Int],
                             kernelSize: List[Int],
                             stride: List[Int],
                             padding: List[Int],
                             dilation: List[Int]): Unit = {

    for (i <- 0 until dimension) {
      if (kernelSize(i) > (inShape(i) + 2 * padding(i)))
        throw new InvalidInputTypeException(
          s"Invalid input: activations into layer are $inShape but kernel size is $kernelSize with padding $padding"
        )

      if (stride(i) <= 0)
        throw new InvalidInputTypeException(
          s"Invalid stride: all $stride elements should be great than 0"
        )

      if (dilation(i) <= 0)
        throw new InvalidInputTypeException(
          s"Invalid stride: all $dilation elements should be great than 0"
        )
    }

  }

  override def outputShape: List[Int] = {
    val nOutChannels: Int =
      if (nFilter > 0) nFilter
      else if (inputShape.nonEmpty) inputShape.last
      else 0
    if (inputShape.lengthCompare(dimension + 1) == 0) {
      validateShapes(dimension, inputShape, kernelSize, stride, padding, dilation)
      val effectiveKernel: Array[Int] = ConvolutionUtils.effectiveKernelSize(kernelSize.toArray, dilation.toArray)

      List[List[Int]](inputShape.init, effectiveKernel.toList, padding, stride, dilation)
        .transpose
        .map(x => (x.head - x(1) + 2 * x(2)) / x(3) + 1) :+ nOutChannels
    } else if (nOutChannels > 0) List(nOutChannels)
    else List()
  }
}
