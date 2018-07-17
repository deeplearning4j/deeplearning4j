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

package org.deeplearning4j.scalnet.layers.core

import org.deeplearning4j.nn.conf.layers.{ OutputLayer => JOutputLayer }
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * Extension of base layer, used to construct a DL4J OutputLayer after compilation.
  * OutputLayer has an output object and the ability to return an OutputLayer version
  * of itself, by providing a loss function.
  *
  * @author Max Pumperla
  */
trait OutputLayer extends Layer {
  def output: Output
  def toOutputLayer(lossFunction: LossFunctions.LossFunction): OutputLayer
}
