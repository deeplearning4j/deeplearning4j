package org.nd4j.api.linalg

import org.nd4j.linalg.api.ndarray.INDArray

object DSL {

  /**
   * Make [INDArray] more scala friendly
   */
  implicit def extend(a: INDArray): INDArrayExt = new INDArrayExt(a)

}