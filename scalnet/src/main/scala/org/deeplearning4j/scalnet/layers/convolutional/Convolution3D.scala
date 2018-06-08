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

import org.deeplearning4j.nn.conf.layers.{ Convolution3D => JConvolution3D }
import org.deeplearning4j.nn.conf.layers.Convolution3D.DataFormat
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.scalnet.layers.core.Layer
import org.deeplearning4j.scalnet.regularizers.{NoRegularizer, WeightRegularizer}
import org.nd4j.linalg.activations.Activation

/**
  * 3D convolution for structured image-like inputs. Input should have
  * four dimensions: depth, height, width
  * and number of channels. Convolution is over depth, height and width.
  * For simplicity we assume NDHWC data format, i.e. channels last.
  *
  * @author Max Pumperla
  */
class Convolution3D(nFilter: Int,
                    kernelSize: List[Int],
                    nChannels: Int = 0,
                    stride: List[Int] = List(1, 1, 1),
                    padding: List[Int] = List(0, 0, 0),
                    dilation: List[Int] = List(1, 1, 1),
                    nIn: Option[List[Int]] = None,
                    val weightInit: WeightInit = WeightInit.XAVIER_UNIFORM,
                    val activation: Activation = Activation.IDENTITY,
                    val regularizer: WeightRegularizer = NoRegularizer(),
                    val dropOut: Double = 0.0,
                    override val name: String = "")
  extends Convolution(dimension = 3, kernelSize, stride, padding, dilation, nChannels, nIn, nFilter)
    with Layer {

  override def reshapeInput(nIn: List[Int]): Convolution3D =
    new Convolution3D(nFilter,
      kernelSize,
      nChannels,
      stride,
      padding,
      dilation,
      Some(nIn),
      weightInit,
      activation,
      regularizer,
      dropOut,
      name)

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new JConvolution3D.Builder(kernelSize.head, kernelSize(1), kernelSize(2))
      .nIn(inputShape.last)
      .nOut(outputShape.last)
      .dataFormat(DataFormat.NDHWC)
      .stride(stride.head, stride(1), stride(2))
      .padding(padding.head, padding(1), padding(2))
      .dilation(dilation.head, dilation(1), dilation(2))
      .weightInit(weightInit)
      .activation(activation)
      .l1(regularizer.l1)
      .l2(regularizer.l2)
      .dropOut(dropOut)
      .name(name)
      .build()
}

object Convolution3D {
  def apply(nFilter: Int,
            kernelSize: List[Int],
            nChannels: Int = 0,
            stride: List[Int] = List(1, 1, 1),
            padding: List[Int] = List(0, 0, 0),
            dilation: List[Int] = List(1, 1, 1),
            nIn: Option[List[Int]] = None,
            weightInit: WeightInit = WeightInit.XAVIER_UNIFORM,
            activation: Activation = Activation.IDENTITY,
            regularizer: WeightRegularizer = NoRegularizer(),
            dropOut: Double = 0.0): Convolution3D =
    new Convolution3D(nFilter,
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


