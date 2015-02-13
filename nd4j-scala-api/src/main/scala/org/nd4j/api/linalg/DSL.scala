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
  implicit def extend(a: INDArray): INDArrayExt = new INDArrayExt(a)

}