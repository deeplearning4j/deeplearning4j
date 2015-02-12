package org.nd4j.api.linalg

import org.nd4j.linalg.api.ndarray.{INDArray, BaseNDArray}

/**
 * Created by agibsonccc on 2/8/15.
 */
class ISNDArray extends BaseNDArray {

  def +(that: INDArray): INDArray =
      return addi(that)
  def -(that: INDArray): INDArray =
     return subi(that)
  def x(that: INDArray) : INDArray =
     return muli(that)
  def /(that: INDArray): INDArray =
     return divi(that)
  def += (that: INDArray): INDArray =
       return addi(that)
  def -= (that: INDArray): INDArray =
    return subi(that)
  def *= (that: INDArray): INDArray =
    return muli(that)
  def /= (that: INDArray): INDArray =
    return divi(that)
  def +(that: Number): INDArray =
    return addi(that)
  def -(that: Number): INDArray =
    return subi(that)
  def x(that: Number) : INDArray =
    return muli(that)
  def /(that: Number): INDArray =
    return divi(that)
  def += (that: Number): INDArray =
    return addi(that)
  def -= (that: Number): INDArray =
    return subi(that)
  def *= (that: Number): INDArray =
    return muli(that)
  def /= (that: Number): INDArray =
    return divi(that)
}
