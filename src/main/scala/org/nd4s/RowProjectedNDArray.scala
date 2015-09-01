package org.nd4s

import org.nd4j.linalg.api.ndarray.INDArray

class RowProjectedNDArray(val array:INDArray,filtered:Array[Int]){
  def this(ndarray: INDArray){
    this(ndarray,(0 until ndarray.rows()).toArray)
  }

  def mapi(f:INDArray => INDArray):INDArray = {
    for{
      i <- filtered
    } array.putRow(i,f(array.getRow(i)))
    array
  }

  def map(f:INDArray => INDArray):INDArray = new RowProjectedNDArray(array.dup(),filtered).mapi(f)

  def flatMap(f:INDArray => INDArray):INDArray = map(f)

  def flatMapi(f:INDArray => INDArray):INDArray = mapi(f)

  def withFilter(f:INDArray => Boolean):RowProjectedNDArray = {
    val targets = for{
      i <- filtered
      if f(array.getRow(i))
    } yield i
    new RowProjectedNDArray(array,targets)
  }
}
