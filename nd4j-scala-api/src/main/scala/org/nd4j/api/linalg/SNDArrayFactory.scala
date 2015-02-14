package org.nd4j.api.linalg

import java.util

import org.nd4j.api.linalg.complex.{SComplexDouble, SComplexFloat}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.complex.{IComplexFloat, IComplexDouble, IComplexNumber, IComplexNDArray}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.{Nd4j, BaseNDArrayFactory}
import org.nd4j.linalg.util.ArrayUtil


/**
 * Created by agibsonccc on 2/13/15.
 */
class SNDArrayFactory extends  BaseNDArrayFactory {

  def this(dtype : Int,order : Char) {
    this()
    this.dtype = dtype
    this.order = order
  }

  def this(dtype : java.lang.Integer,order : java.lang.Character) {
    this()
    this.dtype = dtype
    this.order = order
  }



  def createFloat(real: java.lang.Float, imag: java.lang.Float): IComplexFloat =
   return new SComplexFloat(real,imag)

  override def createDouble(real: Double, imag: Double): IComplexDouble =
    return new SComplexDouble(real,imag)

  override def create(data: Array[Float], shape: Array[Int], stride: Array[Int], offset: Int): INDArray =
    return new ISNDArray(data,shape,stride,offset)

  override def create(data: Array[Double], shape: Array[Int], stride: Array[Int], offset: Int): INDArray =
    return new ISNDArray(Nd4j.createBuffer(data),shape,stride,offset)


  override def createComplex(data: Array[Double], shape: Array[Int], stride: Array[Int], offset: Int): IComplexNDArray = {
    return new ISComplexNDArray(Nd4j.createBuffer(data), shape, stride, offset)
  }

  override def createComplex(arr: INDArray): IComplexNDArray =
    return new ISComplexNDArray(arr)

  override def createComplex(data: Array[IComplexNumber], shape: Array[Int]): IComplexNDArray =
    return new ISComplexNDArray(data,shape)


  override def createComplex(data: Array[Float], shape: Array[Int], stride: Array[Int], offset: Int): IComplexNDArray =
    return new ISComplexNDArray(Nd4j.createBuffer(data),shape,stride,offset)

  override def createComplex(data: Array[Double], shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char): IComplexNDArray =
    return new ISComplexNDArray(Nd4j.createBuffer(data),shape,stride,offset,ordering)

  override def create(data: Array[Double], shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char): INDArray =
    return new ISNDArray(data,shape,stride,offset,ordering)

  override def create(data: Array[Double], shape: Array[Int], offset: Int): INDArray =
    return new ISNDArray(Nd4j.createBuffer(data),shape,offset)


  override def create(data: Array[Double], shape: Array[Int], ordering: Char): INDArray =
    return new ISNDArray(data,shape,ordering)

  override def create(data: Array[Float], rows: Int, columns: Int, stride: Array[Int], offset: Int, ordering: Char): INDArray =
    return new ISNDArray(data,Array[Int](rows,columns),stride,offset,ordering)

  override def create(data: Array[Float], shape: Array[Int], offset: Int, order: Character): INDArray =
    return new ISNDArray(data,shape,offset,order)

  override def create(data: DataBuffer, newShape: Array[Int], newStride: Array[Int], offset: Int, ordering: Char): INDArray =
    return new ISNDArray(data,newShape,newStride,offset,ordering)

  override def create(shape: Array[Int], ordering: Char): INDArray =
    return new ISNDArray(Nd4j.createBuffer(ArrayUtil.prod(shape)),shape,Nd4j.getStrides(shape),0,ordering)

  override def create(buffer: DataBuffer, shape: Array[Int], offset: Int): INDArray =
    return new ISNDArray(buffer,shape,offset)

  override def create(data: DataBuffer, rows: Int, columns: Int, stride: Array[Int], offset: Int): INDArray =
    return new ISNDArray(data,Array[Int](rows,columns),stride,offset)

  override def create(data: DataBuffer): INDArray =
    return new ISNDArray(data)

  override def create(data: Array[Array[Double]]): INDArray =
    return new ISNDArray(data)

  override def create(shape: Array[Int], buffer: DataBuffer): INDArray =
    return new ISNDArray(shape,buffer)

  override def create(data: DataBuffer, shape: Array[Int]): INDArray =
    return new ISNDArray(data,shape)

  override def create(data: DataBuffer, shape: Array[Int], stride: Array[Int], offset: Int): INDArray =
    return new ISNDArray(data,shape,stride,offset)

  override def create(floats: Array[Array[Float]]): INDArray = {
    val arr2  = ArrayUtil.flatten(floats)
    val shape : Array[Int] = Array(floats.length,floats(0).length)
    val ret =  new ISNDArray(Nd4j.createBuffer(arr2),shape)
    for(i <- 0 until ret.slices()) {
       val slice = ret.slice(i)
      for(j <- 0 until slice.length()) {
          slice.putScalar(j,floats(i)(j))
      }
    }

    return ret
  }


  override def create(data: Array[Float], shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char): INDArray =
    return new ISNDArray(data,shape,stride,offset,ordering)


  override def createComplex(data: Array[Float], shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char): IComplexNDArray =
    return new ISComplexNDArray(Nd4j.createBuffer(data),shape,stride,offset)

  override def createComplex(data: Array[Float], shape: Array[Int], offset: Int): IComplexNDArray =
    return new ISComplexNDArray(Nd4j.createBuffer(data),shape,offset)

  override def createComplex(data: DataBuffer, shape: Array[Int], stride: Array[Int]): IComplexNDArray =
    return new ISComplexNDArray(data,shape,stride)

  override def createComplex(data: DataBuffer, shape: Array[Int]): IComplexNDArray =
    return new ISComplexNDArray(data,shape)


  override def createComplex(data: Array[IComplexNumber], shape: Array[Int], offset: Int, ordering: Char): IComplexNDArray =
    return new ISComplexNDArray(data,shape,offset,ordering)

  override def createComplex(data: Array[Float], order: Character): IComplexNDArray =
    return new ISComplexNDArray(data,order)

  override def createComplex(data: DataBuffer, newDims: Array[Int], newStrides: Array[Int], offset: Int, ordering: Char): IComplexNDArray =
    return new ISComplexNDArray(data,newDims,newStrides,offset,ordering)

  override def createComplex(data: Array[Float], shape: Array[Int], offset: Int, ordering: Char): IComplexNDArray =
    return new ISComplexNDArray(Nd4j.createBuffer(data),shape,offset,ordering)

  override def createComplex(buffer: DataBuffer, shape: Array[Int], offset: Int): IComplexNDArray =
    return new ISComplexNDArray(buffer,shape,offset)

  override def createComplex(buffer: DataBuffer, shape: Array[Int], offset: Int, ordering: Char): IComplexNDArray =
    return new ISComplexNDArray(buffer,shape,offset,ordering)

  override def createComplex(data: Array[IComplexNumber], shape: Array[Int], stride: Array[Int], ordering: Char): IComplexNDArray =
    return new ISComplexNDArray(data,shape,stride,ordering)

  override def createComplex(data: Array[IComplexNumber], shape: Array[Int], stride: Array[Int], offset: Int, ordering: Char): IComplexNDArray =
    return new ISComplexNDArray(data,shape,stride,offset,ordering)

  override def createComplex(data: Array[IComplexNumber], shape: Array[Int], stride: Array[Int], offset: Int): IComplexNDArray =
    return new ISComplexNDArray(data,shape,stride,offset)

  override def createComplex(data: DataBuffer, shape: Array[Int], stride: Array[Int], offset: Int): IComplexNDArray =
    return new ISComplexNDArray(data,shape,stride,offset)

  override def createComplex(data: DataBuffer, rows: Int, columns: Int, stride: Array[Int], offset: Int): IComplexNDArray =
    return new ISComplexNDArray(data,Array[Int](rows,columns),stride,offset)

  override def createFloat(real: Float, imag: Float): IComplexFloat = return new SComplexFloat(real,imag)

  override def create(list: util.List[INDArray], shape: Array[Int]): INDArray = return new ISNDArray(list,shape)

  override def createComplex(arrs: util.List[IComplexNDArray], shape: Array[Int]): IComplexNDArray = return new ISComplexNDArray(arrs,shape,Nd4j.getComplexStrides(shape),Nd4j.order())
  override def create(list: util.List[INDArray], shape: Array[Int], ordering: Char): INDArray = return new ISNDArray(list,shape,ordering)

  override def createComplex(dim: Array[Float]): IComplexNDArray = return createComplex(Nd4j.createBuffer(dim))

  override def createComplex(data: DataBuffer): IComplexNDArray = return new ISComplexNDArray(data)

  override def createComplex(data: Array[IComplexNumber], shape: Array[Int], ordering: Char): IComplexNDArray = return new ISComplexNDArray(data,shape,Nd4j.getComplexStrides(shape),0,ordering)
}
