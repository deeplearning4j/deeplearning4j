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

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.scalnet.layers.core.Layer
import org.deeplearning4j.scalnet.regularizers.{ NoRegularizer, WeightRegularizer }
import org.nd4j.linalg.activations.Activation

/**
  * 2D convolution for structured image-like inputs. Input should have
  * three dimensions: height (number of rows), width (number of columns),
  * and number of channels. Convolution is over height and width.
  *
  * @author David Kale, Max Pumperla
  */
class Convolution2D(nFilter: Int,
                    kernelSize: List[Int],
                    nChannels: Int = 0,
                    stride: List[Int] = List(1, 1),
                    padding: List[Int] = List(0, 0),
                    dilation: List[Int] = List(1, 1),
                    nIn: Option[List[Int]] = None,
                    val weightInit: WeightInit = WeightInit.XAVIER_UNIFORM,
                    val activation: Activation = Activation.IDENTITY,
                    val regularizer: WeightRegularizer = NoRegularizer(),
                    val dropOut: Double = 0.0,
                    override val name: String = "")
    extends Convolution(dimension = 2, kernelSize, stride, padding, dilation, nChannels, nIn, nFilter)
    with Layer {

  override def reshapeInput(nIn: List[Int]): Convolution2D =
    new Convolution2D(nFilter,
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
    new ConvolutionLayer.Builder(kernelSize.head, kernelSize.last)
      .nIn(inputShape.last)
      .nOut(outputShape.last)
      .stride(stride.head, stride.last)
      .padding(padding.head, padding.last)
      .dilation(dilation.head, dilation.last)
      .weightInit(weightInit)
      .activation(activation)
      .l1(regularizer.l1)
      .l2(regularizer.l2)
      .dropOut(dropOut)
      .name(name)
      .build()
}

object Convolution2D {
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
            dropOut: Double = 0.0): Convolution2D =
    new Convolution2D(nFilter,
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
