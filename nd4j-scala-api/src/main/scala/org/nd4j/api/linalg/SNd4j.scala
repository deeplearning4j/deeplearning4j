/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.api.linalg

import java.io._
import java.util

import org.nd4j.api.linalg.{ISComplexNDArray, ISNDArray}

import org.apache.commons.math3.distribution.RealDistribution
import org.apache.commons.math3.random.{MersenneTwister, RandomGenerator}
import org.nd4j.linalg.api.buffer.{FloatBuffer, DataBuffer}
import org.nd4j.linalg.api.complex.{IComplexDouble, IComplexNumber, IComplexFloat, IComplexNDArray}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.convolution.ConvolutionInstance
import org.nd4j.linalg.factory.{NDArrayFactory, Nd4j}
import org.nd4j.linalg.fft.FFTInstance
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.indexing.functions.Value
import org.nd4j.linalg.util.{ArrayUtil, Shape}

/**
 * Over rides nd4j to create scala based ndarrays
 * @author Adam Gibson
 */
object SNd4j extends Nd4j {

  initContext()

  /**
   *
   */
  override def initContext(): Unit = {
    val clazz = classOf[SNDArrayFactory]
    Nd4j.setNdArrayFactoryClazz(clazz)
    super.initContext()

  }

  /**
   * Set a convolution instance
   * @param convolutionInstance
   */
  def setConvolution(convolutionInstance: ConvolutionInstance) {
    Nd4j.setConvolution(convolutionInstance)
  }

  def setNdArrayFactoryClazz(clazz: Class[_ <: NDArrayFactory]) {
    Nd4j.setNdArrayFactoryClazz(clazz)
  }

  def getConvolution: ConvolutionInstance = {
    return Nd4j.getConvolution
  }

  /**
   * Returns the fft instance
   * @return the fft instance
   */
  def getFFt: FFTInstance = {
    return Nd4j.getFFt
  }

  /**
   *
   * @param fftInstance
   */
  def setFft(fftInstance: FFTInstance) {
    Nd4j.setFft(fftInstance)
  }

  /**
   * Given a sequence of Iterators over a applyTransformToDestination of matrices, fill in all of
   * the matrices with the entries in the theta vector.  Errors are
   * thrown if the theta vector does not exactly fill the matrices.
   */
  def setParams(theta: INDArray, matrices: java.util.Collection[INDArray]) {
    Nd4j.setParams(theta,matrices)
  }

  /**
   * Given a sequence of Iterators over a applyTransformToDestination of matrices, fill in all of
   * the matrices with the entries in the theta vector.  Errors are
   * thrown if the theta vector does not exactly fill the matrices.
   */
  def setParams(theta: INDArray, matrices: java.util.Iterator[_ <: INDArray]) {
    Nd4j.setParams(theta,matrices)
  }

  def factory: NDArrayFactory = {
    return Nd4j.factory()
  }

  def cumsum(compute: INDArray): ISNDArray = {
    return compute.cumsum(Integer.MAX_VALUE).asInstanceOf[ISNDArray]
  }

  def max(compute: INDArray): ISNDArray = {
    return compute.max(Integer.MAX_VALUE).asInstanceOf[ISNDArray]
  }

  def min(compute: INDArray): ISNDArray = {
    return compute.min(Integer.MAX_VALUE).asInstanceOf[ISNDArray]
  }

  def prod(compute: INDArray): ISNDArray = {
    return compute.prod(Integer.MAX_VALUE).asInstanceOf[ISNDArray]
  }

  def normmax(compute: INDArray): ISNDArray = {
    return compute.normmax(Integer.MAX_VALUE).asInstanceOf[ISNDArray]
  }

  def norm2(compute: INDArray): ISNDArray = {
    return compute.norm2(Integer.MAX_VALUE).asInstanceOf[ISNDArray]
  }

  def norm1(compute: INDArray): ISNDArray = {
    return compute.norm1(Integer.MAX_VALUE).asInstanceOf[ISNDArray]
  }

  def std(compute: INDArray): ISNDArray = {
    return compute.std(Integer.MAX_VALUE).asInstanceOf[ISNDArray]
  }

  def `var`(compute: INDArray): ISNDArray = {
    return compute.`var`(Integer.MAX_VALUE).asInstanceOf[ISNDArray]
  }

  def sum(compute: INDArray): ISNDArray = {
    return compute.sum(Integer.MAX_VALUE).asInstanceOf[ISNDArray]
  }

  def mean(compute: INDArray): ISNDArray = {
    return compute.mean(Integer.MAX_VALUE).asInstanceOf[ISNDArray]
  }

  def cumsum(compute: IComplexNDArray): ISComplexNDArray = {
    return compute.cumsum(Integer.MAX_VALUE).asInstanceOf[ISComplexNDArray]
  }

  def max(compute: IComplexNDArray): ISComplexNDArray = {
    return compute.max(Integer.MAX_VALUE).asInstanceOf[ISComplexNDArray]
  }

  def min(compute: IComplexNDArray): ISComplexNDArray = {
    return compute.min(Integer.MAX_VALUE).asInstanceOf[ISComplexNDArray]
  }

  def prod(compute: IComplexNDArray): ISComplexNDArray = {
    return compute.prod(Integer.MAX_VALUE).asInstanceOf[ISComplexNDArray]
  }

  def normmax(compute: IComplexNDArray): ISComplexNDArray = {
    return compute.normmax(Integer.MAX_VALUE).asInstanceOf[ISComplexNDArray]
  }

  def norm2(compute: IComplexNDArray): ISComplexNDArray = {
    return compute.norm2(Integer.MAX_VALUE).asInstanceOf[ISComplexNDArray]
  }

  def norm1(compute: IComplexNDArray): ISComplexNDArray = {
    return compute.norm1(Integer.MAX_VALUE).asInstanceOf[ISComplexNDArray]
  }

  def std(compute: IComplexNDArray): ISComplexNDArray = {
    return compute.std(Integer.MAX_VALUE).asInstanceOf[ISComplexNDArray]
  }

  def `var`(compute: IComplexNDArray): ISComplexNDArray = {
    return compute.`var`(Integer.MAX_VALUE).asInstanceOf[ISComplexNDArray]
  }

  def sum(compute: IComplexNDArray): ISComplexNDArray = {
    return compute.sum(Integer.MAX_VALUE).asInstanceOf[ISComplexNDArray]
  }

  def mean(compute: IComplexNDArray): ISComplexNDArray = {
    return compute.mean(Integer.MAX_VALUE).asInstanceOf[ISComplexNDArray]
  }

  def cumsum(compute: INDArray, dimension: Int): ISNDArray = {
    return compute.cumsum(dimension).asInstanceOf[ISNDArray]
  }

  def max(compute: INDArray, dimension: Int): ISNDArray = {
    return compute.max(dimension).asInstanceOf[ISNDArray]
  }

  def min(compute: INDArray, dimension: Int): ISNDArray = {
    return compute.min(dimension).asInstanceOf[ISNDArray]
  }

  def prod(compute: INDArray, dimension: Int): ISNDArray = {
    return compute.prod(dimension).asInstanceOf[ISNDArray]
  }

  def normmax(compute: INDArray, dimension: Int): ISNDArray = {
    return compute.normmax(dimension).asInstanceOf[ISNDArray]
  }

  def norm2(compute: INDArray, dimension: Int): ISNDArray = {
    return compute.norm2(dimension).asInstanceOf[ISNDArray]
  }

  def norm1(compute: INDArray, dimension: Int): ISNDArray = {
    return compute.norm1(dimension).asInstanceOf[ISNDArray]
  }

  def std(compute: INDArray, dimension: Int): ISNDArray = {
    return compute.std(dimension).asInstanceOf[ISNDArray]
  }

  def `var`(compute: INDArray, dimension: Int): ISNDArray = {
    return compute.`var`(dimension).asInstanceOf[ISNDArray]
  }

  def sum(compute: INDArray, dimension: Int): ISNDArray = {
    return compute.sum(dimension).asInstanceOf[ISNDArray]
  }

  def mean(compute: INDArray, dimension: Int): ISNDArray = {
    return compute.mean(dimension).asInstanceOf[ISNDArray]
  }

  def cumsum(compute: IComplexNDArray, dimension: Int): ISComplexNDArray = {
    return compute.cumsum(dimension).asInstanceOf[ISComplexNDArray]
  }

  def max(compute: IComplexNDArray, dimension: Int): ISComplexNDArray = {
    return compute.max(dimension).asInstanceOf[ISComplexNDArray]
  }

  def min(compute: IComplexNDArray, dimension: Int): ISComplexNDArray = {
    return compute.min(dimension).asInstanceOf[ISComplexNDArray]
  }

  def prod(compute: IComplexNDArray, dimension: Int): ISComplexNDArray = {
    return compute.prod(dimension).asInstanceOf[ISComplexNDArray]
  }

  def normmax(compute: IComplexNDArray, dimension: Int): ISComplexNDArray = {
    return compute.normmax(dimension).asInstanceOf[ISComplexNDArray]
  }

  def norm2(compute: IComplexNDArray, dimension: Int): ISComplexNDArray = {
    return compute.norm2(dimension).asInstanceOf[ISComplexNDArray]
  }

  def norm1(compute: IComplexNDArray, dimension: Int): ISComplexNDArray = {
    return compute.norm1(dimension).asInstanceOf[ISComplexNDArray]
  }

  def std(compute: IComplexNDArray, dimension: Int): ISComplexNDArray = {
    return compute.std(dimension).asInstanceOf[ISComplexNDArray]
  }

  def `var`(compute: IComplexNDArray, dimension: Int): ISComplexNDArray = {
    return compute.`var`(dimension).asInstanceOf[ISComplexNDArray]
  }

  def sum(compute: IComplexNDArray, dimension: Int): ISComplexNDArray = {
    return compute.sum(dimension).asInstanceOf[ISComplexNDArray]
  }

  def mean(compute: IComplexNDArray, dimension: Int): ISComplexNDArray = {
    return compute.mean(dimension).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create a buffer equal of length prod(shape)
   * @param shape the shape of the buffer to create
   * @param type the type to create
   * @return the created buffer
   */
  def createBuffer(shape: Array[Int], `type`: Int): DataBuffer = {
    val length: Int = ArrayUtil.prod(shape)
    return if (`type` == DataBuffer.DOUBLE) createBuffer(new Array[Double](length)) else createBuffer(new Array[Float](length))
  }

  /**
   * Create a buffer equal of length prod(shape)
   * @param shape the shape of the buffer to create
   * @return the created buffer
   */
  def createBuffer(shape: Array[Int]): DataBuffer = {
    val length: Int = ArrayUtil.prod(shape)
    return createBuffer(length)
  }

  /**
   * Creates a buffer of the specified length based on the data type
   * @param length the length of te buffer
   * @return the buffer to create
   */
  def createBuffer(length: Long): DataBuffer = {
    return Nd4j.createBuffer(length)
  }

  def createBuffer(data: Array[Float]): DataBuffer = {
    return Nd4j.createBuffer(data)
  }

  def createBuffer(data: Array[Double]): DataBuffer = {
    return Nd4j.createBuffer(data)
  }

  def createBuffer[E](data: Array[E]): DataBuffer = {
    throw new UnsupportedOperationException
  }



  /**
   * Returns the ordering of the ndarrays
   * @return the ordering of the ndarrays
   */
  def order: Character = {
    return factory.order
  }


  /**
   * Create a complex ndarray based on the
   * real and imaginary
   * @param real the real numbers
   * @param imag the imaginary components
   * @return the complex
   */
  def createComplex(real: INDArray, imag: INDArray): ISComplexNDArray = {
    return Nd4j.createComplex(real,imag).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create an n x (shape)
   * ndarray where the ndarray is repeated num times
   * @param n the ndarray to replicate
   * @param num the number of copies to repeat
   * @return the repeated ndarray
   */
  def repeat(n: IComplexNDArray, num: Int): ISComplexNDArray = {
    return Nd4j.repeat(n,num).asInstanceOf[ISComplexNDArray]
  }




  /**
   * Create an n x (shape)
   * ndarray where the ndarray is repeated num times
   * @param n the ndarray to replicate
   * @param num the number of copies to repeat
   * @return the repeated ndarray
   */
  def repeat(n: INDArray, num: Int): ISNDArray = {
    return repeat(n,num).asInstanceOf[ISNDArray]
  }

  /**
   * Generate a linearly spaced vector
   * @param lower upper bound
   * @param upper lower bound
   * @param num the step size
   * @return the linearly spaced vector
   */
  def linspace(lower: Int, upper: Int, num: Int): ISNDArray = {
    return Nd4j.linspace(lower, upper, num).asInstanceOf[ISNDArray]
  }

  def toFlattened(matrices: util.Collection[INDArray]): ISNDArray = {
    return Nd4j.toFlattened(matrices).asInstanceOf[ISNDArray]
  }

  def complexFlatten(flatten: java.util.List[IComplexNDArray]): ISComplexNDArray = {
    return Nd4j.complexFlatten(flatten).asInstanceOf[ISComplexNDArray]
  }

  def complexFlatten(flatten: IComplexNDArray): ISComplexNDArray = {
    return Nd4j.complexFlatten(flatten).asInstanceOf[ISComplexNDArray]
  }

  def toFlattened(length: Int, matrices: java.util.Iterator[_ <: INDArray]): ISNDArray = {
    return Nd4j.toFlattened(length, matrices).asInstanceOf[ISNDArray]
  }

  /**
   * Sort an ndarray along a particular dimension
   *
   * @param ndarray   the ndarray to sort
   * @param dimension the dimension to sort
   * @return an array with indices and the sorted ndarray
   */
  def sortWithIndices(ndarray: IComplexNDArray, dimension: Int, ascending: Boolean): Array[INDArray] = {
    return Nd4j.sortWithIndices(ndarray,dimension,ascending)
  }

  /**
   * Sort an ndarray along a particular dimension
   *
   * @param ndarray   the ndarray to sort
   * @param dimension the dimension to sort
   * @return the indices and the sorted ndarray
   */
  def sortWithIndices(ndarray: INDArray, dimension: Int, ascending: Boolean): Array[INDArray] = {
    return Nd4j.sortWithIndices(ndarray,dimension,ascending)
  }



  /**
   * Sort an ndarray along a particular dimension
   *
   * @param ndarray   the ndarray to sort
   * @param dimension the dimension to sort
   * @return the sorted ndarray
   */
  def sort(ndarray: IComplexNDArray, dimension: Int, ascending: Boolean): IComplexNDArray = {
    return Nd4j.sort(ndarray,dimension,ascending)
  }

  /**
   * Sort an ndarray along a particular dimension
   *
   * @param ndarray   the ndarray to sort
   * @param dimension the dimension to sort
   * @return the sorted ndarray
   */
  def sort(ndarray: INDArray, dimension: Int, ascending: Boolean): INDArray = {
    return Nd4j.sort(ndarray,dimension,ascending)
  }

  /**
   * Returns a column vector where each entry is the nth bilinear
   * product of the nth slices of the two tensors.
   */
  def bilinearProducts(curr: INDArray, in: INDArray): ISNDArray = {
    return Nd4j.bilinearProducts(curr, in).asInstanceOf[ISNDArray]
  }

  def toFlattened(matrices: INDArray): ISNDArray = {
    return Nd4j.toFlattened(matrices).asInstanceOf[ISNDArray]
  }

  /**
   * Create the identity ndarray
   * @param n the number for the identity
   * @return
   */
  def eye(n: Int): ISNDArray = {
    return Nd4j.eye(n).asInstanceOf[ISNDArray]
  }

  /**
   * Rotate a matrix 90 degrees
   * @param toRotate the matrix to rotate
   * @return the rotated matrix
   */
  def rot90(toRotate: INDArray) {
    Nd4j.rot90(toRotate)
  }

  /**
   * Read line via input streams
   * @param filePath the input stream ndarray
   * @param split the split separator
   * @return the read txt method
   *
   */
  @throws(classOf[IOException])
  def writeTxt(write: INDArray, filePath: String, split: String) {
    Nd4j.writeTxt(write,filePath,split)
  }

  /**
   * Read line via input streams
   * @param filePath the input stream ndarray
   * @param split the split separator
   * @return the read txt method
   *
   */
  @throws(classOf[IOException])
  def readTxt(filePath: InputStream, split: String): ISNDArray = {
    return Nd4j.readTxt(filePath,split).asInstanceOf[ISNDArray]
  }



  /**
   * Read line via input streams
   * @param filePath the input stream ndarray
   * @param split the split separator
   * @return the read txt method
   *
   */
  @throws(classOf[IOException])
  def readTxt(filePath: String, split: String): ISNDArray = {
    return readTxt(new FileInputStream(filePath), split)
  }

  /**
   * Read line via input streams
   * @param filePath the input stream ndarray
   * @return the read txt method
   *
   */
  @throws(classOf[IOException])
  def readTxt(filePath: String): ISNDArray = {
    return readTxt(filePath, "\t")
  }



  /**
   * Read in an ndarray from a data input stream
   * @param dis the data input stream to read from
   * @return the ndarray
   * @throws IOException
   */
  @throws(classOf[IOException])
  def read(dis: DataInputStream): ISNDArray = {
    return read(dis).asInstanceOf[ISNDArray]
  }

  /**
   * Write an ndarray to the specified outputs tream
   * @param arr the array to write
   * @param dataOutputStream the data output stream to write to
   * @throws IOException
   */
  @throws(classOf[IOException])
  def write(arr: INDArray, dataOutputStream: DataOutputStream) {
    Nd4j.write(arr,dataOutputStream)
  }

  /**
   * Clear nans from an ndarray
   * @param arr the array to clear
   */
  def clearNans(arr: INDArray) {
    BooleanIndexing.applyWhere(arr, Conditions.isNan, new Value(Nd4j.EPS_THRESHOLD))
  }

  /**
   * Create an ndarray based on the given data layout
   * @param data the data to use
   * @return an ndarray with the given data layout
   */
  def create(data: Array[Array[Double]]): ISNDArray = {
    return Nd4j.create(data).asInstanceOf[ISNDArray]
  }

  /**
   * Read in an ndarray from a data input stream
   * @param dis the data input stream to read from
   * @return the ndarray
   * @throws IOException
   */
  @throws(classOf[IOException])
  def readComplex(dis: DataInputStream): ISComplexNDArray = {
    return readComplex(dis).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Write an ndarray to the specified outputs tream
   * @param arr the array to write
   * @param dataOutputStream the data output stream to write to
   * @throws IOException
   */
  @throws(classOf[IOException])
  def writeComplex(arr: IComplexNDArray, dataOutputStream: DataOutputStream) {
    Nd4j.writeComplex(arr,dataOutputStream)
  }

  /**
   * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
   * @param reverse the matrix to reverse
   * @return the reversed matrix
   */
  def rot(reverse: INDArray): ISNDArray = {
    return Nd4j.rot(reverse).asInstanceOf[ISNDArray]
  }

  /**
   * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
   * @param reverse the matrix to reverse
   * @return the reversed matrix
   */
  def reverse(reverse: INDArray): ISNDArray = {
    return Nd4j.reverse(reverse).asInstanceOf[ISNDArray]
  }

  /**
   * Array of evenly spaced values.
   * @param begin the begin of the range
   * @param end the end of the range
   * @return the range vector
   */
  def arange(begin: Double, end: Double): ISNDArray = {
    return Nd4j.arange(begin, end).asInstanceOf[ISNDArray]
  }

  /**
   * Create double
   * @param real real component
   * @param imag imag component
   * @return
   */
  def createComplexNumber(real: Number, imag: Number): IComplexNumber = {
    return Nd4j.createComplexNumber(real,imag)
  }

  /**
   * Create double
   * @param real real component
   * @param imag imag component
   * @return
   */
  def createFloat(real: Float, imag: Float): IComplexFloat = {
    return Nd4j.createFloat(real, imag)
  }

  /**
   * Create an instance of a complex double
   * @param real the real component
   * @param imag the imaginary component
   * @return a new imaginary double with the specified real and imaginary components
   */
  def createDouble(real: Double, imag: Double): IComplexDouble = {
    return Nd4j.createDouble(real, imag)
  }

  /**
   * Copy a to b
   * @param a the origin matrix
   * @param b the destination matrix
   */
  def copy(a: INDArray, b: INDArray) {
    Nd4j.copy(a, b)
  }

  /**
   * Generates a random matrix between min and max
   * @param shape the number of rows of the matrix
   * @param min the minimum number
   * @param max the maximum number
   * @param rng the rng to use
   * @return a drandom matrix of the specified shape and range
   */
  def rand(shape: Array[Int], min: Double, max: Double, rng: RandomGenerator): ISNDArray = {
    return Nd4j.rand(shape, min, max, rng).asInstanceOf[ISNDArray]
  }

  /**
   * Generates a random matrix between min and max
   * @param rows the number of rows of the matrix
   * @param columns the number of columns in the matrix
   * @param min the minimum number
   * @param max the maximum number
   * @param rng the rng to use
   * @return a drandom matrix of the specified shape and range
   */
  def rand(rows: Int, columns: Int, min: Double, max: Double, rng: RandomGenerator): ISNDArray = {
    return Nd4j.rand(rows, columns, min, max, rng).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a new matrix where the values of the given vector are the diagonal values of
   * the matrix if a vector is passed in, if a matrix is returns the kth diagonal
   * in the matrix
   * @param x the diagonal values
   * @param k the kth diagonal to getDouble
   * @return new matrix
   */
  def diag(x: IComplexNDArray, k: Int): ISComplexNDArray = {
    return diag(x,k).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates a new matrix where the values of the given vector are the diagonal values of
   * the matrix if a vector is passed in, if a matrix is returns the kth diagonal
   * in the matrix
   * @param x the diagonal values
   * @param k the kth diagonal to getDouble
   * @return new matrix
   */
  def diag(x: INDArray, k: Int): ISNDArray = {
    return Nd4j.diag(x,k).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a new matrix where the values of the given vector are the diagonal values of
   * the matrix if a vector is passed in, if a matrix is returns the kth diagonal
   * in the matrix
   * @param x the diagonal values
   * @return new matrix
   */
  def diag(x: IComplexNDArray): ISComplexNDArray = {
    return diag(x, 0)
  }

  /**
   * Creates a new matrix where the values of the given vector are the diagonal values of
   * the matrix if a vector is passed in, if a matrix is returns the kth diagonal
   * in the matrix
   * @param x the diagonal values
   * @return new matrix
   */
  def diag(x: INDArray): ISNDArray = {
    return diag(x, 0)
  }

  def appendBias(vectors: INDArray): ISNDArray = {
    return Nd4j.appendBias(vectors).asInstanceOf[ISNDArray]
  }

  /**
   * Perform an operation along a diagonal
   * @param x the ndarray to perform the operation on
   * @param func the operation to perform
   */
  def doAlongDiagonal(x: INDArray, func: Function[Number, Number]) {
    if (x.isMatrix) {
      var i: Int = 0
      while (i < x.rows) {
        x.put(i, i, func.apply(x.getDouble(i, i)))
        ({
          i += 1; i - 1
        })
      }
    }
  }

  /**
   * Perform an operation along a diagonal
   * @param x the ndarray to perform the operation on
   * @param func the operation to perform
   */
  def doAlongDiagonal(x: IComplexNDArray, func: Function[IComplexNumber, IComplexNumber]) {
    if (x.isMatrix) {
      var i: Int = 0
      while (i < x.rows) {
        x.putScalar(i, i, func.apply(x.getComplex(i, i)))
        ({
          i += 1; i - 1
        })
      }
    }
  }

  /**
   * Create a complex ndarray from the passed in indarray
   * @param arr the arr to wrap
   * @return the complex ndarray with the specified ndarray as the
   *         real components
   */
  def createComplex(arr: INDArray): ISComplexNDArray = {
    if (arr.isInstanceOf[IComplexNDArray]) return arr.asInstanceOf[ISComplexNDArray]
    return Nd4j.createComplex(arr).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create a complex ndarray from the passed in indarray
   * @param data the data to wrap
   * @return the complex ndarray with the specified ndarray as the
   *         real components
   */
  def createComplex(data: Array[IComplexNumber], shape: Array[Int]): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create a complex ndarray from the passed in indarray
   * @param data the data to wrap
   * @return the complex ndarray with the specified ndarray as the
   *         real components
   */
  def createComplex(data: Array[IComplexNumber], shape: Array[Int], offset: Int, ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape, offset, ordering).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create a complex ndarray from the passed in indarray
   * @param arrs the arr to wrap
   * @return the complex ndarray with the specified ndarray as the
   *         real components
   */
  def createComplex(arrs: java.util.List[IComplexNDArray], shape: Array[Int]): ISComplexNDArray = {
    return Nd4j.createComplex(arrs, shape).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create a random ndarray with the given shape using the given rng
   * @param rows the number of rows in the matrix
   * @param columns the number of columns in the matrix
   * @param r the random generator to use
   * @return the random ndarray with the specified shape
   */
  def rand(rows: Int, columns: Int, r: RandomGenerator): ISNDArray = {
    return Nd4j.rand(rows, columns, r).asInstanceOf[ISNDArray]
  }

  /**
   * Create a random ndarray with the given shape using the given rng
   * @param rows the number of rows in the matrix
   * @param columns the columns of the ndarray
   * @param seed the  seed to use
   * @return the random ndarray with the specified shape
   */
  def rand(rows: Int, columns: Int, seed: Long): ISNDArray = {
    return Nd4j.rand(rows, columns, seed).asInstanceOf[ISNDArray]
  }

  /**
   * Create a random ndarray with the given shape using
   * the current time as the seed
   * @param rows the number of rows in the matrix
   * @param columns the number of columns in the matrix
   * @return the random ndarray with the specified shape
   */
  def rand(rows: Int, columns: Int): ISNDArray = {
    return Nd4j.rand(rows, columns).asInstanceOf[ISNDArray]
  }

  /**
   * Random normal using the given rng
   * @param rows the number of rows in the matrix
   * @param columns the number of columns in the matrix
   * @param r the random generator to use
   * @return
   */
  def randn(rows: Int, columns: Int, r: RandomGenerator): ISNDArray = {
    return Nd4j.randn(rows, columns, r).asInstanceOf[ISNDArray]
  }

  /**
   * Random normal using the current time stamp
   * as the seed
   * @param rows the number of rows in the matrix
   * @param columns the number of columns in the matrix
   * @return
   */
  def randn(rows: Int, columns: Int): ISNDArray = {
    return Nd4j.randn(rows, columns).asInstanceOf[ISNDArray]
  }

  /**
   * Random normal using the specified seed
   * @param rows the number of rows in the matrix
   * @param columns the number of columns in the matrix
   * @return
   */
  def randn(rows: Int, columns: Int, seed: Long): ISNDArray = {
    return Nd4j.randn(rows, columns, seed).asInstanceOf[ISNDArray]
  }

  /**
   * Create a random ndarray with the given shape using the given rng
   * @param shape the shape of the ndarray
   * @param r the random generator to use
   * @return the random ndarray with the specified shape
   */
  def rand(shape: Array[Int], r: RealDistribution): ISNDArray = {
    return Nd4j.rand(shape, r).asInstanceOf[ISNDArray]
  }

  /**
   * Create a random ndarray with the given shape using the given rng
   * @param shape the shape of the ndarray
   * @param r the random generator to use
   * @return the random ndarray with the specified shape
   */
  def rand(shape: Array[Int], r: RandomGenerator): ISNDArray = {
    return Nd4j.rand(shape, r).asInstanceOf[ISNDArray]
  }

  /**
   * Create a random ndarray with the given shape using the given rng
   * @param shape the shape of the ndarray
   * @param seed the  seed to use
   * @return the random ndarray with the specified shape
   */
  def rand(shape: Array[Int], seed: Long): ISNDArray = {
    return Nd4j.rand(shape, seed).asInstanceOf[ISNDArray]
  }

  /**
   * Create a random ndarray with the given shape using
   * the current time as the seed
   * @param shape the shape of the ndarray
   * @return the random ndarray with the specified shape
   */
  def rand(shape: Array[Int]): ISNDArray = {
    return Nd4j.rand(shape).asInstanceOf[ISNDArray]
  }

  /**
   * Random normal using the given rng
   * @param shape the shape of the ndarray
   * @param r the random generator to use
   * @return
   */
  def randn(shape: Array[Int], r: RandomGenerator): ISNDArray = {
    return Nd4j.randn(shape, r).asInstanceOf[ISNDArray]
  }

  /**
   * Random normal using the current time stamp
   * as the seed
   * @param shape the shape of the ndarray
   * @return
   */
  def randn(shape: Array[Int]): ISNDArray = {
    return Nd4j.randn(shape).asInstanceOf[ISNDArray]
  }

  /**
   * Random normal using the specified seed
   * @param shape the shape of the ndarray
   * @return
   */
  def randn(shape: Array[Int], seed: Long): ISNDArray = {
    return randn(shape, new MersenneTwister(seed)).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a row vector with the data
   * @param data the columns of the ndarray
   * @return  the created ndarray
   */
  def create(data: Array[Float], order: Char): ISNDArray = {
    return Nd4j.create(data).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a row vector with the data
   * @param data the columns of the ndarray
   * @return  the created ndarray
   */
  def create(data: Array[Double], order: Char): ISNDArray = {
    return Nd4j.create(data).asInstanceOf[ISNDArray]
  }

  /**
   * Creates an ndarray with the specified data
   * @param data the number of columns in the row vector
   * @return ndarray
   */
  def createComplex(data: Array[Double], order: Char): ISComplexNDArray = {
    return Nd4j.createComplex(data).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates a row vector with the specified number of columns
   * @param columns the columns of the ndarray
   * @return  the created ndarray
   */
  def create(columns: Int, order: Char): ISNDArray = {
    return Nd4j.create(columns).asInstanceOf[ISNDArray]
  }

  /**
   * Creates an ndarray
   * @param columns the number of columns in the row vector
   * @return ndarray
   */
  def createComplex(columns: Int, order: Char): ISComplexNDArray = {
    return Nd4j.createComplex(columns).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates a row vector with the data
   * @param data the columns of the ndarray
   * @return  the created ndarray
   */
  def create(data: Array[Float]): ISNDArray = {
    return create(data, order)
  }

  /**
   * Creates a row vector with the data
   * @param data the columns of the ndarray
   * @return  the created ndarray
   */
  def create(data: Array[Double]): ISNDArray = {
    return create(data, order)
  }





  /**
   * Creates an ndarray
   * @param columns the number of columns in the row vector
   * @return ndarray
   */
  def createComplex(columns: Int): ISComplexNDArray = {
    return createComplex(columns, order).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Returns true if the given ndarray has either
   * an infinite or nan
   * @param num the ndarray to test
   * @return true if the given ndarray has either infinite or nan
   *         false otherwise
   */
  def hasInvalidNumber(num: INDArray): Boolean = {
    return Nd4j.hasInvalidNumber(num)
  }

  /**
   * Creates a row vector with the specified number of columns
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @return  the created ndarray
   */
  def zeros(rows: Int, columns: Int): ISNDArray = {
    return Nd4j.zeros(rows, columns).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a matrix of zeros
   * @param rows te number of rows in the matrix
   * @param columns the number of columns in the row vector
   * @return ndarray
   */
  def complexZeros(rows: Int, columns: Int): ISNDArray = {
    return Nd4j.complexZeros(rows, columns).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a row vector with the specified number of columns
   * @param columns the columns of the ndarray
   * @return  the created ndarray
   */
  def zeros(columns: Int): ISNDArray = {
    return Nd4j.zeros(columns).asInstanceOf[ISNDArray]
  }

  /**
   * Creates an ndarray
   * @param columns the number of columns in the row vector
   * @return ndarray
   */
  def complexZeros(columns: Int): ISNDArray = {
    return Nd4j.complexZeros(columns).asInstanceOf[ISNDArray]
  }

  def complexValueOf(num: Int, value: IComplexNumber): ISComplexNDArray = {
    return Nd4j.complexValueOf(num, value).asInstanceOf[ISComplexNDArray]
  }

  def complexValueOf(shape: Array[Int], value: IComplexNumber): ISComplexNDArray = {
    return Nd4j.complexValueOf(shape, value).asInstanceOf[ISComplexNDArray]
  }

  def complexValueOf(num: Int, value: Double): ISComplexNDArray = {
    return Nd4j.complexValueOf(num, value).asInstanceOf[ISComplexNDArray]
  }

  def complexValueOf(shape: Array[Int], value: Double): ISComplexNDArray = {
    return Nd4j.complexValueOf(shape, value).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified value
   * as the  only value in the ndarray
   * @param shape the shape of the ndarray
   * @param value the value to assign
   * @return  the created ndarray
   */
  def valueArrayOf(shape: Array[Int], value: Double): ISNDArray = {
    return Nd4j.valueArrayOf(shape, value).asInstanceOf[ISNDArray]
  }

  /**
   * Creates an ndarray with the specified value
   * as the  only value in the ndarray
   * @param num number of columns
   * @param value the value to assign
   * @return  the created ndarray
   */
  def valueArrayOf(num: Int, value: Double): ISNDArray = {
    return Nd4j.valueArrayOf(Array[Int](1, num), value).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a row vector with the specified number of columns
   * @param rows the number of rows in the matrix
   * @param columns the columns of the ndarray
   * @param value the value to assign
   * @return  the created ndarray
   */
  def valueArrayOf(rows: Int, columns: Int, value: Double): ISNDArray = {
    return Nd4j.valueArrayOf(rows, columns, value).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a row vector with the specified number of columns
   * @param rows the number of rows in the matrix
   * @param columns the columns of the ndarray
   * @return  the created ndarray
   */
  def ones(rows: Int, columns: Int): ISNDArray = {
    return Nd4j.ones(rows, columns).asInstanceOf[ISNDArray]
  }

  /**
   * Creates an ndarray
   * @param rows the number of rows in the matrix
   * @param columns the number of columns in the row vector
   * @return ndarray
   */
  def complexOnes(rows: Int, columns: Int): ISNDArray = {
    return Nd4j.complexOnes(rows, columns).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a row vector with the specified number of columns
   * @param columns the columns of the ndarray
   * @return  the created ndarray
   */
  def ones(columns: Int): ISNDArray = {
    return Nd4j.ones(columns).asInstanceOf[ISNDArray]
  }

  /**
   * Creates an ndarray
   * @param columns the number of columns in the row vector
   * @return ndarray
   */
  def complexOnes(columns: Int): ISComplexNDArray = {
    return Nd4j.complexOnes(columns).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Concatenates two matrices horizontally. Matrices must have identical
   * numbers of rows.
   * @param arrs the first matrix to concat
   *
   */
  def hstack(arrs: INDArray): ISNDArray = {
    return Nd4j.hstack(arrs).asInstanceOf
  }

  /**
   * Concatenates two matrices vertically. Matrices must have identical
   * numbers of columns.
   * @param arrs
   *
   */
  def vstack(arrs: INDArray): ISNDArray = {
    return Nd4j.vstack(arrs).asInstanceOf[ISNDArray]
  }

  /**
   * Concatneate ndarrays along a dimension
   * @param dimension the dimension to concatneate along
   * @param toConcat the ndarrays to concat
   * @return the concatted ndarrays with an output shape of
   *         the ndarray shapes save the dimension shape specified
   *         which is then the sum of the sizes along that dimension
   */
  def concat(dimension: Int, toConcat: INDArray): ISNDArray = {
    return Nd4j.concat(dimension, toConcat).asInstanceOf[ISNDArray]
  }

  /**
   * Concatneate ndarrays along a dimension
   * @param dimension the dimension to concatneate along
   * @param toConcat the ndarrays to concat
   * @return the concatted ndarrays with an output shape of
   *         the ndarray shapes save the dimension shape specified
   *         which is then the sum of the sizes along that dimension
   */
  def concat(dimension: Int, toConcat: IComplexNDArray): ISComplexNDArray = {
    return Nd4j.concat(dimension, toConcat).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create an ndarray of zeros
   * @param shape the shape of the ndarray
   * @return an ndarray with ones filled in
   */
  def zeros(shape: Array[Int]): ISNDArray = {
    return Nd4j.zeros(shape).asInstanceOf[ISNDArray]
  }

  /**
   * Create an ndarray of ones
   * @param shape the shape of the ndarray
   * @return an ndarray with ones filled in
   */
  def complexZeros(shape: Array[Int]): ISComplexNDArray = {
    return Nd4j.complexZeros(shape).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create an ndarray of ones
   * @param shape the shape of the ndarray
   * @return an ndarray with ones filled in
   */
  def ones(shape: Array[Int]): ISNDArray = {
    return Nd4j.ones(shape).asInstanceOf[ISNDArray]
  }

  /**
   * Create an ndarray of ones
   * @param shape the shape of the ndarray
   * @return an ndarray with ones filled in
   */
  def complexOnes(shape: Array[Int]): ISComplexNDArray = {
    return Nd4j.complexOnes(shape).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param data the data to use with the ndarray
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def createComplex(data: Array[Float], rows: Int, columns: Int, stride: Array[Int], offset: Int): ISComplexNDArray = {
    return Nd4j.createComplex(data, rows, columns, stride, offset).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param data  the data to use with the ndarray
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def create(data: Array[Float], rows: Int, columns: Int, stride: Array[Int], offset: Int): ISNDArray = {
    return Nd4j.create(data, rows, columns, stride, offset).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param data the data to use with the ndarray
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @param offset  the offset of the ndarray
   * @return the instance
   */
  def createComplex(data: Array[Double], shape: Array[Int], stride: Array[Int], offset: Int): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape, stride, offset).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def create(data: Array[Double], shape: Array[Int], stride: Array[Int], offset: Int): ISNDArray = {
    return Nd4j.create(data, shape, stride, offset).asInstanceOf[ISNDArray]
  }

  /**
   * Create an ndrray with the specified shape
   * @param data the data to use with tne ndarray
   * @param shape the shape of the ndarray
   * @return the created ndarray
   */
  def create(data: Array[Double], shape: Array[Int]): ISNDArray = {
    return Nd4j.create(data, shape).asInstanceOf[ISNDArray]
  }

  /**
   * Create an ndrray with the specified shape
   * @param data the data to use with tne ndarray
   * @param shape the shape of the ndarray
   * @return the created ndarray
   */
  def create(data: Array[Float], shape: Array[Int]): ISNDArray = {
    return Nd4j.create(data, shape).asInstanceOf[ISNDArray]
  }

  /**
   * Create an ndrray with the specified shape
   * @param data the data to use with tne ndarray
   * @param shape the shape of the ndarray
   * @return the created ndarray
   */
  def createComplex(data: Array[Float], shape: Array[Int]): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create an ndrray with the specified shape
   * @param data the data to use with tne ndarray
   * @param shape the shape of the ndarray
   * @return the created ndarray
   */
  def createComplex(data: Array[Double], shape: Array[Int]): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create an ndrray with the specified shape
   * @param data the data to use with tne ndarray
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @return the created ndarray
   */
  def createComplex(data: Array[Float], shape: Array[Int], stride: Array[Int]): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape, stride).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create an ndrray with the specified shape
   * @param data the data to use with tne ndarray
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @return the created ndarray
   */
  def createComplex(data: Array[Double], shape: Array[Int], stride: Array[Int]): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape, stride).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def createComplex(data: Array[Double], rows: Int, columns: Int, stride: Array[Int], offset: Int): ISComplexNDArray = {
    return Nd4j.createComplex(data, rows, columns, stride, offset).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param data the data to use with tne ndarray
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def create(data: Array[Double], rows: Int, columns: Int, stride: Array[Int], offset: Int): ISNDArray = {
    return Nd4j.create(data, rows, columns, stride, offset).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @param offset  the offset of the ndarray
   * @return the instance
   */
  def createComplex(data: Array[Float], shape: Array[Int], stride: Array[Int], offset: Int): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape, stride, offset).asInstanceOf[ISComplexNDArray]
  }


  /**
   * Creates an ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def create(data: Array[Double], shape: Array[Int], offset: Int, ordering: Char): ISNDArray = {
    return Nd4j.create(data, shape, Nd4j.getStrides(shape), offset).asInstanceOf[ISNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def create(data: Array[Float], shape: Array[Int], stride: Array[Int], offset: Int): ISNDArray = {
    return Nd4j.create(data, shape, stride, offset).asInstanceOf[ISNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @return the instance
   */
  def create(list: java.util.List[INDArray], shape: Array[Int]): ISNDArray = {
    return Nd4j.create(list, shape).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def createComplex(rows: Int, columns: Int, stride: Array[Int], offset: Int): ISComplexNDArray = {
    return Nd4j.createComplex(rows, columns, stride, offset).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def create(rows: Int, columns: Int, stride: Array[Int], offset: Int): ISNDArray = {
    return Nd4j.create(rows, columns, stride, offset).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @param offset  the offset of the ndarray
   * @return the instance
   */
  def createComplex(shape: Array[Int], stride: Array[Int], offset: Int): ISComplexNDArray = {
    return Nd4j.createComplex(shape, stride, offset).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def create(shape: Array[Int], stride: Array[Int], offset: Int): ISNDArray = {
    return Nd4j.create(shape, stride, offset).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @return the instance
   */
  def createComplex(rows: Int, columns: Int, stride: Array[Int]): ISComplexNDArray = {
    return createComplex(rows, columns, stride, order).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @return the instance
   */
  def create(rows: Int, columns: Int, stride: Array[Int]): ISNDArray = {
    return create(rows, columns, stride, order)
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @return the instance
   */
  def createComplex(shape: Array[Int], stride: Array[Int]): ISComplexNDArray = {
    return createComplex(shape, stride, order).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @return the instance
   */
  def create(shape: Array[Int], stride: Array[Int]): ISNDArray = {
    return create(shape, stride, order).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @return the instance
   */
  def createComplex(rows: Int, columns: Int): ISComplexNDArray = {
    return createComplex(rows, columns, order).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @return the instance
   */
  def create(rows: Int, columns: Int): ISNDArray = {
    return create(rows, columns, order).asInstanceOf[ISNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @return the instance
   */
  def create(shape: Int): ISNDArray = {
    return create(shape, order).asInstanceOf[ISNDArray]
  }

  /**
   * Create a scalar ndarray with the specified offset
   * @param value the value to initialize the scalar with
   * @param offset the offset of the ndarray
   * @return the created ndarray
   */
  def scalar(value: Number, offset: Int): ISNDArray = {
    return Nd4j.scalar(value, offset).asInstanceOf[ISNDArray]
  }

  /**
   * Create a scalar ndarray with the specified offset
   * @param value the value to initialize the scalar with
   * @param offset the offset of the ndarray
   * @return the created ndarray
   */
  def complexScalar(value: Number, offset: Int): ISComplexNDArray = {
    return Nd4j.complexScalar(value, offset).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create a scalar ndarray with the specified offset
   * @param value the value to initialize the scalar with
   * @return the created ndarray
   */
  def complexScalar(value: Number): ISComplexNDArray = {
    return Nd4j.complexScalar(value).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create a scalar nd array with the specified value and offset
   * @param value the value of the scalar
   * @param offset the offset of the ndarray
   * @return the scalar nd array
   */
  def scalar(value: Double, offset: Int): ISNDArray = {
    return Nd4j.scalar(value, offset).asInstanceOf[ISNDArray]
  }

  /**
   * Create a scalar nd array with the specified value and offset
   * @param value the value of the scalar
   * @param offset the offset of the ndarray
   * @return the scalar nd array
   */
  def scalar(value: Float, offset: Int): ISNDArray = {
    return Nd4j.scalar(value, offset).asInstanceOf[ISNDArray]
  }

  /**
   * Create a scalar ndarray with the specified offset
   * @param value the value to initialize the scalar with
   * @return the created ndarray
   */
  def scalar(value: Number): ISNDArray = {
    return Nd4j.scalar(value).asInstanceOf[ISNDArray]
  }

  /**
   * Create a scalar nd array with the specified value and offset
   * @param value the value of the scalar
    =     * @return the scalar nd array
   */
  def scalar(value: Double): ISNDArray = {
    return Nd4j.scalar(value).asInstanceOf[ISNDArray]
  }

  /**
   * Create a scalar nd array with the specified value and offset
   * @param value the value of the scalar
    =     * @return the scalar nd array
   */
  def scalar(value: Float): ISNDArray = {
    return Nd4j.scalar(value).asInstanceOf[ISNDArray]
  }

  /**
   * Create a scalar ndarray with the specified offset
   * @param value the value to initialize the scalar with
   * @param offset the offset of the ndarray
   * @return the created ndarray
   */
  def scalar(value: IComplexNumber, offset: Int): ISComplexNDArray = {
    return Nd4j.scalar(value, offset).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create a scalar nd array with the specified value and offset
   * @param value the value of the scalar
   * @return the scalar nd array
   */
  def scalar(value: IComplexFloat): ISComplexNDArray = {
    return Nd4j.scalar(value).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create a scalar nd array with the specified value and offset
   * @param value the value of the scalar
    =     * @return the scalar nd array
   */
  def scalar(value: IComplexDouble): ISComplexNDArray = {
    return Nd4j.scalar(value).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create a scalar ndarray with the specified offset
   * @param value the value to initialize the scalar with
   * @return the created ndarray
   */
  def scalar(value: IComplexNumber): ISComplexNDArray = {
    return Nd4j.scalar(value).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create a scalar nd array with the specified value and offset
   * @param value the value of the scalar
   * @param offset the offset of the ndarray
   * @return the scalar nd array
   */
  def scalar(value: IComplexFloat, offset: Int): ISComplexNDArray = {
    return Nd4j.scalar(value, offset).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create a scalar nd array with the specified value and offset
   * @param value the value of the scalar
   * @param offset the offset of the ndarray
   * @return the scalar nd array
   */
  def scalar(value: IComplexDouble, offset: Int): ISComplexNDArray = {
    return Nd4j.scalar(value, offset).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Get the strides for the given order and shape
   * @param shape the shape of the ndarray
   * @param order the order to getScalar the strides for
   * @return the strides for the given shape and order
   */
  def getStrides(shape: Array[Int], order: Char): Array[Int] = {
    return Nd4j.getStrides(shape,order)
  }

  /**
   * Get the strides based on the shape
   * and NDArrays.order()
   * @param shape the shape of the ndarray
   * @return the strides for the given shape
   *         and order specified by NDArrays.order()
   */
  def getStrides(shape: Array[Int]): Array[Int] = {
    return getStrides(shape, Nd4j.order).asInstanceOf[Array[Int]]
  }

  /**
   * An alias for repmat
   * @param tile the ndarray to tile
   * @param repeat the shape to repeat
   * @return the tiled ndarray
   */
  def tile(tile: INDArray, repeat: Array[Int]): ISNDArray = {
    return tile.repmat(repeat).asInstanceOf[ISNDArray]
  }

  /**
   * Get the strides for the given order and shape
   * @param shape the shape of the ndarray
   * @param order the order to getScalar the strides for
   * @return the strides for the given shape and order
   */
  def getComplexStrides(shape: Array[Int], order: Char): Array[Int] = {
    if (order == NDArrayFactory.FORTRAN) return ArrayUtil.calcStridesFortran(shape, 2)
    return ArrayUtil.calcStrides(shape, 2)
  }

  /**
   * Get the strides based on the shape
   * and NDArrays.order()
   * @param shape the shape of the ndarray
   * @return the strides for the given shape
   *         and order specified by NDArrays.order()
   */
  def getComplexStrides(shape: Array[Int]): Array[Int] = {
    return getComplexStrides(shape, Nd4j.order).asInstanceOf[Array[Int]]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param data the data to use with the ndarray
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def createComplex(data: Array[Float], rows: Int, columns: Int, stride: Array[Int], offset: Int, ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(data, rows, columns, stride, offset).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param data  the data to use with the ndarray
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def create(data: Array[Float], rows: Int, columns: Int, stride: Array[Int], offset: Int, ordering: Char): ISNDArray = {
    return Nd4j.create(data, rows, columns, stride, offset, ordering).asInstanceOf[ISNDArray]
  }

  def create(shape: Array[Int], dataType: Int): ISNDArray = {
    return Nd4j.create(shape, dataType).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param data the data to use with the ndarray
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @param offset  the offset of the ndarray
   * @return the instance
   */
  def createComplex(data: Array[Float], shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape, stride, offset, ordering).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def create(data: Array[Double], shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char): ISNDArray = {
    return Nd4j.create(data, shape, stride, offset, ordering).asInstanceOf[ISNDArray]
  }

  /**
   * Create an ndrray with the specified shape
   * @param data the data to use with tne ndarray
   * @param shape the shape of the ndarray
   * @return the created ndarray
   */
  def create(data: Array[Double], shape: Array[Int], ordering: Char): ISNDArray = {
    return Nd4j.create(data, shape, ordering).asInstanceOf[ISNDArray]
  }

  /**
   * Create an ndrray with the specified shape
   * @param data the data to use with tne ndarray
   * @param shape the shape of the ndarray
   * @return the created ndarray
   */
  def create(data: Array[Float], shape: Array[Int], ordering: Char): ISNDArray = {
    return Nd4j.create(data, shape, ordering).asInstanceOf[ISNDArray]
  }

  /**
   * Create an ndrray with the specified shape
   * @param data the data to use with tne ndarray
   * @param shape the shape of the ndarray
   * @return the created ndarray
   */
  def createComplex(data: Array[Float], shape: Array[Int], ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape, ordering).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create an ndrray with the specified shape
   * @param data the data to use with tne ndarray
   * @param shape the shape of the ndarray
   * @return the created ndarray
   */
  def createComplex(data: Array[Double], shape: Array[Int], ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape, ordering).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create an ndrray with the specified shape
   * @param data the data to use with tne ndarray
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @return the created ndarray
   */
  def createComplex(data: Array[Float], shape: Array[Int], stride: Array[Int], ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape, stride, ordering).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Create an ndrray with the specified shape
   * @param data the data to use with tne ndarray
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @return the created ndarray
   */
  def createComplex(data: Array[Double], shape: Array[Int], stride: Array[Int], ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape, stride, ordering).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def createComplex(data: Array[Double], rows: Int, columns: Int, stride: Array[Int], offset: Int, ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(data, rows, columns, stride, offset).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param data the data to use with tne ndarray
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def create(data: Array[Double], rows: Int, columns: Int, stride: Array[Int], offset: Int, ordering: Char): ISNDArray = {
    return Nd4j.create(data, rows, columns, stride, offset).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @param offset  the offset of the ndarray
   * @return the instance
   */
  def createComplex(data: Array[Double], shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape, stride, offset).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def create(data: Array[Float], shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char): ISNDArray = {
    return Nd4j.create(data, shape, stride, offset, ordering).asInstanceOf[ISNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @return the instance
   */
  def create(list: java.util.List[INDArray], shape: Array[Int], ordering: Char): ISNDArray = {
    return Nd4j.create(list, shape, ordering).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def createComplex(rows: Int, columns: Int, stride: Array[Int], offset: Int, ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(rows, columns, stride, offset).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def create(rows: Int, columns: Int, stride: Array[Int], offset: Int, ordering: Char): ISNDArray = {
    return Nd4j.create(rows, columns, stride, offset).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @param offset  the offset of the ndarray
   * @return the instance
   */
  def createComplex(shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(shape, stride, offset).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @param offset the offset of the ndarray
   * @return the instance
   */
  def create(shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char): ISNDArray = {
    return Nd4j.create(shape, stride, offset).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @return the instance
   */
  def createComplex(rows: Int, columns: Int, stride: Array[Int], ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(rows, columns, stride).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @param stride the stride for the ndarray
   * @return the instance
   */
  def create(rows: Int, columns: Int, stride: Array[Int], ordering: Char): ISNDArray = {
    return Nd4j.create(rows, columns, stride).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @return the instance
   */
  def createComplex(shape: Array[Int], stride: Array[Int], ordering: Char): ISComplexNDArray = {
    return createComplex(shape, stride, 0).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @param stride the stride for the ndarray
   * @return the instance
   */
  def create(shape: Array[Int], stride: Array[Int], ordering: Char): ISNDArray = {
    return Nd4j.create(shape, stride).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @return the instance
   */
  def createComplex(rows: Int, columns: Int, ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(rows, columns).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param rows the rows of the ndarray
   * @param columns the columns of the ndarray
   * @return the instance
   */
  def create(rows: Int, columns: Int, ordering: Char): ISNDArray = {
    return Nd4j.create(rows, columns, ordering).asInstanceOf[ISNDArray]
  }

  /**
   * Creates a complex ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @return the instance
   */
  def createComplex(shape: Array[Int], ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(createBuffer(ArrayUtil.prod(shape) * 2), shape, 0, ordering).asInstanceOf[ISComplexNDArray]
  }

  /**
   * Creates an ndarray with the specified shape
   * @param shape the shape of the ndarray
   * @return the instance
   */
  def create(shape: Array[Int], ordering: Char): ISNDArray = {
    return Nd4j.create(shape, ordering).asInstanceOf[ISNDArray]
  }

  def createComplex(data: Array[Float], ints: Array[Int], offset: Int, ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(data, ints, ArrayUtil.calcStrides(ints, 2), offset, ordering).asInstanceOf[ISComplexNDArray]
  }

  def createComplex(data: Array[Double], shape: Array[Int], offset: Int): ISComplexNDArray = {
    return createComplex(data, shape, offset, Nd4j.order).asInstanceOf[ISComplexNDArray]
  }

  def create(data: Array[Double], shape: Array[Int], offset: Int): ISNDArray = {
    return Nd4j.create(data, shape, offset).asInstanceOf[ISNDArray]
  }

  def createComplex(data: Array[Double], ints: Array[Int], offset: Int, ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(data, ints, offset, ordering).asInstanceOf[ISComplexNDArray]
  }

  def createComplex(dim: Array[Double]): ISComplexNDArray = {
    return Nd4j.createComplex(dim, Array[Int](1, dim.length / 2)).asInstanceOf[ISComplexNDArray]
  }

  def createComplex(data: Array[Float], shape: Array[Int], offset: Int): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape, offset).asInstanceOf[ISComplexNDArray]
  }

  def create(doubles: Array[Array[Float]]): ISNDArray = {
    return Nd4j.create(doubles).asInstanceOf[ISNDArray]
  }

  def complexLinSpace(i: Int, i1: Int, i2: Int): ISComplexNDArray = {
    return Nd4j.createComplex(Nd4j.linspace(i, i1, i2)).asInstanceOf[ISComplexNDArray]
  }

  def create(data: Array[Float], shape: Array[Int], stride: Array[Int], ordering: Char, offset: Int): ISNDArray = {
    return Nd4j.create(data, shape, stride, offset, ordering).asInstanceOf[ISNDArray]
  }

  def create(data: Array[Float], shape: Array[Int], ordering: Char, offset: Int): ISNDArray = {
    return Nd4j.create(data, shape, getStrides(shape, ordering), offset, ordering).asInstanceOf[ISNDArray]
  }

  def create(data: DataBuffer, shape: Array[Int], strides: Array[Int], offset: Int): ISNDArray = {
    return Nd4j.create(data, shape, strides, offset).asInstanceOf[ISNDArray]
  }

  def create(data: DataBuffer, shape: Array[Int], offset: Int): ISNDArray = {
    return Nd4j.create(data, shape, getStrides(shape), offset).asInstanceOf[ISNDArray]
  }

  def create(data: DataBuffer, newShape: Array[Int], newStride: Array[Int], offset: Int, ordering: Char): ISNDArray = {
    return Nd4j.create(data, newShape, newStride, offset, ordering).asInstanceOf[ISNDArray]
  }

  def createComplex(data: DataBuffer, newShape: Array[Int], newStrides: Array[Int], offset: Int): ISComplexNDArray = {
    return Nd4j.createComplex(data, newShape, newStrides, offset).asInstanceOf[ISComplexNDArray]
  }

  def createComplex(data: DataBuffer, shape: Array[Int], offset: Int): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape, offset).asInstanceOf[ISComplexNDArray]
  }

  def createComplex(data: DataBuffer, newDims: Array[Int], newStrides: Array[Int], offset: Int, ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(data, newDims, newStrides, offset, ordering).asInstanceOf[ISComplexNDArray]
  }

  def createComplex(data: DataBuffer, shape: Array[Int], offset: Int, ordering: Char): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape, offset, ordering).asInstanceOf[ISComplexNDArray]
  }

  def create(data: DataBuffer, shape: Array[Int]): ISNDArray = {
    return Nd4j.create(data, shape).asInstanceOf[ISNDArray]
  }

  def createComplex(data: DataBuffer, shape: Array[Int]): ISComplexNDArray = {
    return Nd4j.createComplex(data, shape).asInstanceOf[ISComplexNDArray]
  }

  def create(buffer: DataBuffer): ISNDArray = {
    return Nd4j.create(buffer).asInstanceOf[ISNDArray]
  }



}