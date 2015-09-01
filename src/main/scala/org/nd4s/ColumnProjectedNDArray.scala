package org.nd4j.api

import org.nd4j.linalg.api.ndarray.INDArray

class ColumnProjectedNDArray(val array:INDArray,filtered:Array[Int]){
  def this(ndarray: INDArray){
    this(ndarray,(0 until ndarray.columns()).toArray)
  }

  def mapi(f:INDArray => INDArray):INDArray = {
    for{
      i <- filtered
    } array.putColumn(i,f(array.getColumn(i)))
    array
  }

  def map(f:INDArray => INDArray):INDArray = new ColumnProjectedNDArray(array.dup(),filtered).flatMapi(f)

  def flatMap(f:INDArray => INDArray):INDArray = map(f)

  def flatMapi(f:INDArray => INDArray):INDArray = mapi(f)

  def withFilter(f:INDArray => Boolean):ColumnProjectedNDArray = {
    val targets = for{
      i <- filtered
      if f(array.getColumn(i))
    } yield i
    new ColumnProjectedNDArray(array,targets)
  }
}
