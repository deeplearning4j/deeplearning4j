package org.nd4j.api

import org.nd4j.linalg.api.ndarray.INDArray

class SliceProjectedNDArray(val array:INDArray,filtered:Array[Int]){
  def this(ndarray: INDArray){
    this(ndarray,(0 until ndarray.slices()).toArray)
  }

  def mapi(f:INDArray => INDArray):INDArray = {
    for{
      i <- filtered
    } array.putSlice(i,f(array.slice(i)))
    array
  }

  def map(f:INDArray => INDArray):INDArray = new SliceProjectedNDArray(array.dup(),filtered).flatMapi(f)

  def flatMap(f:INDArray => INDArray):INDArray = map(f)

  def flatMapi(f:INDArray => INDArray):INDArray = mapi(f)

  def withFilter(f:INDArray => Boolean):SliceProjectedNDArray = {
    val targets = for{
      i <- filtered
      if f(array.slice(i))
    } yield i
    new SliceProjectedNDArray(array,targets)
  }
}
