package org.nd4j.api.linalg

import java.util

import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.complex.{IComplexNumber, IComplexNDArray, BaseComplexNDArray}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.{Nd4j}
import org.nd4j.linalg.util.ArrayUtil

/**
 * Scala complex ndarray
 * @author Adam Gibson
 */
class ISComplexNDArray extends BaseComplexNDArray {


  def this(buffer: DataBuffer, shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char) {
    this()
    this.data = buffer
    this.shape = shape
    this.stride = stride
    this.offset = offset
    this.ordering = ordering
    initShape(shape)

  }

  def this(buffer : DataBuffer) {
    this()
    this.data = buffer
    initShape(Array[Int](1,buffer.length()))
  }


  /**
   * Create an ndarray from the specified slices
   * and the given shape
   * @param slices the slices of the ndarray
   * @param shape the final shape of the ndarray
   * @param stride the stride of the ndarray
   * @param ordering the ordering for the ndarray
   *
   */
  def this(slices : java.util.List[IComplexNDArray],shape : Array[Int], stride : Array[Int],ordering : Char) {
    this()

    val list = new util.ArrayList[IComplexNumber]
    for(i <-  0 until slices.size()) {
      val flattened = slices.get(i).ravel()
      for(j  <- 0 until flattened.length()) {
        list.add(flattened.getComplex(j))
      }
    }


    this.ordering = ordering
    this.data = Nd4j.createBuffer(ArrayUtil.prod(shape) * 2)
    this.stride = stride
    initShape(shape)

    var count = 0
    for (i <- 0 until list.size()) {
      putScalar(count,list.get(i))
      count += 1
    }
  }






  def this(shape: Array[Int], offset: Int, ordering: Char) {
    this(Nd4j.createBuffer(ArrayUtil.prod(shape) * 2), shape, Nd4j.getComplexStrides(shape, ordering), offset, ordering)
  }




  def this(shape: Array[Int], ordering: Char) {
    this(Nd4j.createBuffer(ArrayUtil.prod(shape) * 2), shape, Nd4j.getComplexStrides(shape, ordering), 0, ordering)
  }

  def this(shape : Array[Int],stride : Array[Int],offset : Int,ordering : Char) {
    this(Nd4j.createBuffer(ArrayUtil.prod(shape)),shape,stride,offset,ordering)
  }

  def this(shape : Array[Int],stride : Array[Int],ordering : Char) {
    this(shape,stride,0,ordering)
  }

  /**
   * Initialize the given ndarray as the real component
   * @param m the real component
   * @param stride the stride of the ndarray
   * @param ordering the ordering for the ndarray
   */
  def this(m: INDArray, stride: Array[Int], ordering: Char) {
    this(m.shape, stride, ordering)
    copyFromReal(m)
  }

  /** Construct a complex matrix from a realComponent matrix. */
  def this(m: INDArray, ordering: Char) {
    this(m.shape, ordering)
    copyFromReal(m)
  }

  /** Construct a complex matrix from a realComponent matrix. */
  def this(m: INDArray) {
    this(m, Nd4j.order)
  }

  /**
   * Create with the specified ndarray as the real component
   * and the given stride
   * @param m the ndarray to use as the stride
   * @param stride the stride of the ndarray
   */
  def this(m: INDArray, stride: Array[Int]) {
    this(m, stride, Nd4j.order)
  }






  def this(data: Array[Float], shape: Array[Int], stride: Array[Int], offset: Int, order: Character) {
    this()
    this.data = Nd4j.createBuffer(data)
    this.stride = stride
    this.offset = offset
    this.ordering = order
    initShape(shape)
  }


  def this(data: DataBuffer, shape: Array[Int], stride: Array[Int], offset: Int) {
    this()
    this.data = data
    this.stride = stride
    this.offset = offset
    this.ordering = Nd4j.order
    initShape(shape)
  }




  def this(data: Array[IComplexNumber], shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char) {
    this(shape, stride, offset, ordering)
    for(i <- 0 until data.length) {}
  }

  def this(data: Array[IComplexNumber], shape: Array[Int], stride: Array[Int], offset: Int) {
    this(data, shape, stride, offset, Nd4j.order)
  }

  def this(data: Array[IComplexNumber], shape: Array[Int], offset: Int, ordering: Char) {
    this(data, shape, Nd4j.getComplexStrides(shape), offset, ordering)
  }

  def this(buffer: DataBuffer, shape: Array[Int], offset: Int, ordering: Char) {
    this(buffer, shape, Nd4j.getComplexStrides(shape), offset, ordering)
  }

  def this(buffer: DataBuffer, shape: Array[Int], offset: Int) {
    this(buffer, shape, Nd4j.getComplexStrides(shape), offset, Nd4j.order)
  }


  def this(data: Array[Float], shape: Array[Int], stride: Array[Int], ordering: Char) {
    this(data, shape, stride, 0, ordering)
  }

  def this(data: DataBuffer, shape: Array[Int], stride: Array[Int]) {
    this(data, shape, stride, 0, Nd4j.order)
  }


  /**
   * Create this ndarray with the given data and shape and 0 offset
   *
   * @param data     the data to use
   * @param shape    the shape of the ndarray
   * @param ordering
   */
  def this(data: Array[Float], shape: Array[Int], ordering: Char) {
    this(data, shape, Nd4j.getComplexStrides(shape, ordering), 0, ordering)
  }

  def this(shape: Array[Int]) {
    this(Nd4j.createBuffer(ArrayUtil.prod(shape) * 2), shape, Nd4j.getComplexStrides(shape))
  }

  def this(data: DataBuffer, shape: Array[Int]) {
    this(shape)
    this.data = data
  }

  def this(data: Array[Float], order: Character) {
    this(data, Array[Int](data.length / 2), order)
  }
  /**
   * Create a complex ndarray with the given complex doubles.
   * Note that this maybe an easier setup than the new float
   * @param newData the new data for this array
   * @param shape the shape of the ndarray
   */
  def this(newData : Array[IComplexNumber],shape : Array[Int]) {
    this()
    this.data = Nd4j.createBuffer(Array[Float](ArrayUtil.prod(shape) * 2))
    initShape(shape)
    var i : Int = 0
    for( i <- 0  until length) {
      this.putScalar(i,newData(i))
    }

  }




}
