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

import org.deeplearning4j.nn.conf.layers.{Convolution1DLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.scalnet.layers.core.Layer
import org.deeplearning4j.scalnet.regularizers.{NoRegularizer, WeightRegularizer}
import org.nd4j.linalg.activations.Activation

/**
  * 1D convolution for structured image-like inputs. Input should have
  * two dimensions: height and number of channels. Convolution is over height only.
  *
  * @author Max Pumperla
  */
class Convolution1D(nFilter: Int,
                    kernelSize: List[Int],
                    nChannels: Int = 0,
                    stride: List[Int] = List(1),
                    padding: List[Int] = List(0),
                    dilation: List[Int] = List(1),
                    nIn: Option[List[Int]] = None,
                    val weightInit: WeightInit = WeightInit.XAVIER_UNIFORM,
                    val activation: Activation = Activation.IDENTITY,
                    val regularizer: WeightRegularizer = NoRegularizer(),
                    val dropOut: Double = 0.0,
                    override val name: String = "")
  extends Convolution(dimension = 1, kernelSize, stride, padding, dilation, nChannels, nIn, nFilter)
    with Layer {

  override def reshapeInput(nIn: List[Int]): Convolution2D =
    new Convolution2D(nFilter, kernelSize, nChannels, stride, padding, dilation, Some(nIn),
      weightInit, activation, regularizer, dropOut, name)

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new Convolution1DLayer.Builder(kernelSize.head, kernelSize.last)
      .nIn(inputShape.last)
      .nOut(outputShape.last)
      .stride(stride.head)
      .padding(padding.head)
      .dilation(dilation.head)
      .weightInit(weightInit)
      .activation(activation)
      .l1(regularizer.l1)
      .l2(regularizer.l2)
      .dropOut(dropOut)
      .name(name)
      .build()
}


object Convolution1D {
  def apply(nFilter: Int,
            kernelSize: List[Int],
            nChannels: Int = 0,
            stride: List[Int] = List(1, 1),
            padding: List[Int] = List(0, 0),
            dilation: List[Int] = List(1, 1),
            nIn: Option[List[Int]] = None,
            weightInit: WeightInit = WeightInit.XAVIER_UNIFORM,
            activation: Activation = Activation.IDENTITY,
            regularizer: WeightRegularizer = NoRegularizer(),
            dropOut: Double = 0.0): Convolution1D =
    new Convolution1D(nFilter,
      kernelSize,
      nChannels,
      stride,
      padding,
      dilation,
      nIn,
      weightInit,
      activation,
      regularizer,
      dropOut)
}