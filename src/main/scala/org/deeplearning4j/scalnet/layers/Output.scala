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

package org.deeplearning4j.scalnet.layers

import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
  * Trait for output layers in DL4J neural networks and computational graphs.
  *
  * @author David Kale
  */
trait Output {
  protected var _isOutput: Boolean = false
  protected var _lossFunction: LossFunction = _

  def isOutput: Boolean = _isOutput
  def lossFunction: LossFunction = _lossFunction

  def makeOutput(lossFunction: LossFunction): Unit = {
    _isOutput = true
    _lossFunction = lossFunction
  }
}
