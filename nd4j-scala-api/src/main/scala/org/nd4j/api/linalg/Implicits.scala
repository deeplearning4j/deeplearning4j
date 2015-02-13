package org.nd4j.api.linalg

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object Implicits {

  /**
   * Make [INDArray] more scala friendly
   */
  implicit def extend(a: INDArray): INDArrayExt = new INDArrayExt(a)

}