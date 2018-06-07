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

import org.deeplearning4j.scalnet.layers.core.Node

/**
  * Base upsampling layer
  *
  * @author Max Pumperla
  */
class Upsampling(protected val dimension: Int,
                 protected val size: List[Int],
                 protected val nChannels: Int = 0,
                 protected val nIn: Option[List[Int]] = None,
                 override val name: String = "")
  extends Node {

  override def inputShape: List[Int] = nIn.getOrElse(List(nChannels))

  override def outputShape: List[Int] = {
    val nOutChannels: Int =
      if (inputShape.nonEmpty) inputShape.last
      else 0
    if (inputShape.lengthCompare(dimension + 1) == 0) {
      List[List[Int]](inputShape.init, size)
        .transpose
        .map(x => x.head * x(1)) :+ nOutChannels
    } else if (nOutChannels > 0) List(nOutChannels)
    else List()
  }

}


