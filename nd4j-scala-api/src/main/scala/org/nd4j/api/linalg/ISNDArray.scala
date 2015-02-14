package org.nd4j.api.linalg

import org.nd4j.linalg.api.buffer.{FloatBuffer, DoubleBuffer, DataBuffer}
import org.nd4j.linalg.api.ndarray.{INDArray, BaseNDArray}
import org.nd4j.linalg.factory.{NDArrayFactory, Nd4j}
import org.nd4j.linalg.util.ArrayUtil
import org.nd4j.linalg.util.ArrayUtil._


/**
 * Operator overloading with ndarrays
 * @author Adam Gibson
 */
class ISNDArray extends BaseNDArray {

  def this(buffer: DataBuffer) {
    this()
    this.data = buffer
    initShape(Array[Int](1, buffer.length))
  }

  def this(buffer: DataBuffer, shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char) {
    this()
    this.data = buffer
    if (ArrayUtil.prod(shape) > buffer.length) throw new IllegalArgumentException("Shape must be <= buffer length")
    this.stride = stride
    this.offset = offset
    this.ordering = ordering
    initShape(shape)
  }

  /**
   * Create an ndarray from the specified slices.
   * This will go through and merge all of the
   * data from each slice in to one ndarray
   * which will then take the specified shape
   * @param slices the slices to merge
   * @param shape the shape of the ndarray
   */
  def this(slices: java.util.List[INDArray], shape: Array[Int], stride: Array[Int], offset : Int,ordering: Char) {
    this()
    val ret: DataBuffer = if (slices.get(0).data.dataType == (DataBuffer.FLOAT)) Nd4j.createBuffer(new Array[Float](ArrayUtil.prod(shape))) else Nd4j.createBuffer(new Array[Double](ArrayUtil.prod(shape)))
    this.stride = stride
    this.ordering = ordering
    this.data = ret
    this.offset = offset
    initShape(shape)

      var i: Int = 0
      while (i < slices.size()) {
        {
          putSlice(i, slices.get(i))
        }
        ({
          i += 1; i - 1
        })
      }

  }




  /**
   * Create with the specified shape and buffer
   * @param shape the shape
   * @param buffer the buffer
   */
  def this(shape: Array[Int], buffer: DataBuffer) {
    this()
    this.data = buffer
    initShape(shape)
  }




  /**
   * Construct an ndarray of the specified shape
   * with an empty data array
   * @param shape the shape of the ndarray
   * @param stride the stride of the ndarray
   * @param offset the desired offset
   * @param ordering the ordering of the ndarray
   */
  def this(shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char) {
    this(Nd4j.createBuffer(ArrayUtil.prod(shape)), shape, stride, offset, ordering)
  }

  /**
   * Create the ndarray with
   * the specified shape and stride and an offset of 0
   * @param shape the shape of the ndarray
   * @param stride the stride of the ndarray
   * @param ordering the ordering of the ndarray
   */
  def this(shape: Array[Int], stride: Array[Int], ordering: Char) {
    this(shape, stride, 0, ordering)
  }

  def this(shape: Array[Int], offset: Int, ordering: Char) {
    this(shape, Nd4j.getStrides(shape, ordering), offset, ordering)
  }

  def this(shape: Array[Int]) {
    this(shape, 0, Nd4j.order)
  }

  def this(rows : Int, cols : Int) {
    this(Array[Int](rows,cols))
  }

  def this(data: Array[Array[Double]]) {
    this(data.length, data(0).length)

  }

  /**
   * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
   *
   * @param newRows    the number of rows (<i>n</i>) of the new matrix.
   * @param newColumns the number of columns (<i>m</i>) of the new matrix.
   */
  def this(newRows: Int, newColumns: Int, ordering: Char) {
    this()
    this.ordering = ordering
    initShape(Array[Int](newRows, newColumns))
  }




  /**
   * Create an ndarray from the specified slices.
   * This will go through and merge all of the
   * data from each slice in to one ndarray
   * which will then take the specified shape
   * @param slices the slices to merge
   * @param shape the shape of the ndarray
   */
  def this(slices: java.util.List[INDArray], shape: Array[Int], stride: Array[Int], ordering: Char) {
    this()
    val ret: DataBuffer = if (slices.get(0).data.dataType == (DataBuffer.FLOAT)) Nd4j.createBuffer(new Array[Float](ArrayUtil.prod(shape))) else Nd4j.createBuffer(new Array[Double](ArrayUtil.prod(shape)))
    this.stride = stride
    this.ordering = ordering
    this.data = ret
    initShape(shape)
    var i: Int = 0

    for( a <- 1 until slices.size()){
      putSlice(i, slices.get(i))
    }



  }
  def this(slices : java.util.List[INDArray],shape : Array[Int]) {
    this(slices,shape,Nd4j.getStrides(shape),0,Nd4j.order())
  }



  def this(data: Array[Float], shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char) {
    this()
    this.offset = offset
    this.stride = stride
    this.ordering = ordering
    initShape(shape)
    if (data != null && data.length > 0) {
      this.data = Nd4j.createBuffer(data)
      if (offset >= data.length) throw new IllegalArgumentException("invalid offset: must be < data.length")
    }
  }

  def this(data: DataBuffer, shape: Array[Int], stride: Array[Int], offset: Int) {
    this()
    this.data = data
    this.stride = stride
    this.offset = offset
    this.ordering = Nd4j.order
    initShape(shape)
  }



  def this(data: DataBuffer, shape: Array[Int]) {
    this(data, shape, Nd4j.getStrides(shape), 0, Nd4j.order)
  }

  def this(buffer: DataBuffer, shape: Array[Int], offset: Int) {
    this(buffer, shape, Nd4j.getStrides(shape), offset)
  }

  def this(data: Array[Double], shape: Array[Int], ordering: Char) {
    this(new DoubleBuffer(data), shape, ordering)
  }

  def this(data: Array[Double], shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char) {
    this(new DoubleBuffer(data), shape, stride, offset, ordering)
  }


  def this(floatBuffer: FloatBuffer, order: Char) {
    this(floatBuffer, Array[Int](floatBuffer.length), Nd4j.getStrides(Array[Int](floatBuffer.length)), 0, order)
  }

  def this(buffer: DataBuffer, shape: Array[Int], strides: Array[Int]) {
    this(buffer, shape, strides, 0, Nd4j.order)
  }

  def this(data: Array[Float], order: Char) {
    this(new FloatBuffer(data), order)
  }






  def this(data: Array[Int], shape: Array[Int], strides: Array[Int]) {
    this(Nd4j.createBuffer(data), shape, strides)
  }




  def this(data: Array[Float],shape: Array[Int],strides : Array[Int],offset: Int) {
    this(Nd4j.createBuffer(data),shape,strides,offset,Nd4j.order())
  }

  /**
   *
   * @param data the data to use
   * @param shape the shape of the ndarray
   * @param offset the desired offset
   * @param ordering the ordering of the ndarray
   */
  def this(data: Array[Float], shape: Array[Int], offset: Int, ordering: Char) {
    this(data, shape, if (ordering == NDArrayFactory.C) calcStrides(shape) else calcStridesFortran(shape), offset)
  }


  /**
   * Create this ndarray with the given data and shape and 0 offset
   * @param data the data to use
   * @param shape the shape of the ndarray
   */
  def this(data: Array[Float], shape: Array[Int], ordering: Char) {
    this(data, shape, 0, ordering)
  }




  /**
   * Create an ndarray from the specified slices.
   * This will go through and merge all of the
   * data from each slice in to one ndarray
   * which will then take the specified shape
   * @param slices the slices to merge
   * @param shape the shape of the ndarray
   */
  def this(slices: java.util.List[INDArray], shape: Array[Int], ordering: Char) {
    this(slices,shape,Nd4j.getStrides(shape),0,ordering)
  }


  /**
   *
   * @param that
   * @return
   */
  def +(that: INDArray): ISNDArray =
    return  add(that).asInstanceOf[ISNDArray]

  /**
   *
   * @param that
   * @return
   */
  def -(that: INDArray): ISNDArray =
    return sub(that).asInstanceOf[ISNDArray]

  /**
   *
   * @param that
   * @return
   */
  def *(that: INDArray) : ISNDArray =
    return mul(that).asInstanceOf[ISNDArray]

  /**
   *
   * @param that
   * @return
   */
  def /(that: INDArray): ISNDArray =
    return div(that).asInstanceOf[ISNDArray]

  /**
   *
   * @param that
   * @return
   */
  def += (that: INDArray): ISNDArray =
    return addi(that).asInstanceOf[ISNDArray]

  /**
   *
   * @param that
   * @return
   */
  def -= (that: INDArray): ISNDArray =
    return subi(that).asInstanceOf[ISNDArray]

  /**
   *
   * @param that
   * @return
   */
  def *= (that: INDArray): ISNDArray =
    return muli(that).asInstanceOf[ISNDArray]

  /**
   *
   * @param that
   * @return
   */
  def /= (that: INDArray): ISNDArray =
    return divi(that).asInstanceOf[ISNDArray]

  /**
   *
   * @param that
   * @return
   */
  def +(that: Number): ISNDArray =
    return add(that).asInstanceOf[ISNDArray]

  /**
   *
   * @param that
   * @return
   */
  def -(that: Number): ISNDArray =
    return sub(that).asInstanceOf[ISNDArray]

  /**
   *
   *
   * @param that
   * @return
   */
  def *(that: Number) : ISNDArray =
    return mul(that).asInstanceOf[ISNDArray]

  /**
   *
   * @param that
   * @return
   */
  def /(that: Number): ISNDArray =
    return div(that).asInstanceOf[ISNDArray]

  /**
   *
   * @param that
   * @return
   */
  def += (that: Number): ISNDArray =
    return addi(that).asInstanceOf[ISNDArray]

  /**
   *
   * @param that
   * @return
   */
  def -= (that: Number): ISNDArray =
    return subi(that).asInstanceOf[ISNDArray]

  /**
   *
   * @param that
   * @return
   */
  def *= (that: Number): ISNDArray =
    return muli(that).asInstanceOf[ISNDArray]

  /**
   *
   * @param that
   * @return
   */
  def /= (that: Number): ISNDArray =
    return divi(that).asInstanceOf[ISNDArray]

  def T(): ISNDArray =
    return transposei().asInstanceOf[ISNDArray]

}
