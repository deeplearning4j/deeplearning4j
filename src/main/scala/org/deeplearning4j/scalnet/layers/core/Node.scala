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

package org.deeplearning4j.scalnet.layers.core

/**
  * Trait for node in DL4J neural networks and computational graphs.
  * Nodes are assumed to have inputs and outputs with "shapes."
  *
  * @author David Kale
  */
trait Node {

  def name: String

  def inputShape: List[Int]

  def outputShape: List[Int]

  def reshapeInput(nIn: List[Int]): Node = this

  def describe(): String = "in=" + inputShape + " out=" + outputShape

}
