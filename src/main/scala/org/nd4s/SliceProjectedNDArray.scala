package org.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.{SpecifiedIndex, NDArrayIndex}

class SliceProjectedNDArray(val array:INDArray,filtered:Array[Int]){
  def this(ndarray: INDArray){
    this(ndarray,(0 until ndarray.slices()).toArray)
  }

  def mapi(f:INDArray => INDArray):INDArray = {
    for{
      i <- filtered
    } array.putSlice(i,f(array.slice(i)))
    array.get(new SpecifiedIndex(filtered:_*) +: NDArrayIndex.allFor(array).init:_*)
  }

  def map(f:INDArray => INDArray):INDArray = new SliceProjectedNDArray(array.dup(),filtered).flatMapi(f)

  def flatMap(f:INDArray => INDArray):INDArray = map(f)

  def flatMapi(f:INDArray => INDArray):INDArray = mapi(f)

  def foreach(f:INDArray => Unit):Unit =
    for{
      i <- filtered
    } f(array.slice(i))

  def withFilter(f:INDArray => Boolean):SliceProjectedNDArray = {
    val targets = for{
      i <- filtered
      if f(array.slice(i))
    } yield i
    new SliceProjectedNDArray(array,targets)
  }
}
