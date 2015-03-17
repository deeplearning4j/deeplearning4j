/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.api.linalg

import org.nd4j.linalg.api.ndarray.INDArray

/**
 * DSL for using [[INDArray]]s with more Scala-friendly syntax
 *
 * Usage example: {{{
 *   // To use the DSL just use the following import
 *   import org.nd4j.api.linalg.DSL._
 *
 *   // You can now use Scala syntax on the arrays
 *   val a = Nd4j.create(Array[Float](1, 2), Array(2, 1))
 *   val b = a + 10
 * }}}
 */
object DSL {

  /**
   * Make [INDArray] more scala friendly
   */
  implicit def toRichNDArray(a: INDArray): RichNDArray = new RichNDArray(a)

}