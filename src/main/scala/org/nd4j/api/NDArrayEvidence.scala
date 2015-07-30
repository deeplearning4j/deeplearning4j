package org.nd4j.api

import org.nd4j.linalg.api.complex.{IComplexNumber, IComplexNDArray}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex


object Evidences {
  implicit val double = DoubleNDArrayEvidence
  implicit val float = FloatNDArrayEvidence
  implicit val complex = ComplexNDArrayEvidence
}

object NDArrayEvidence {
  implicit val doubleNDArrayEvidence = DoubleNDArrayEvidence

  implicit val complexNDArrayEvidence = ComplexNDArrayEvidence
}

trait NDArrayEvidence[A] {
  type Value
  type NDArray = A

  def sum(ndarray: A): Value

  def mean(ndarray: A): Value

  def normMax(ndarray: A): Value

  def norm1(ndarray: A): Value

  def norm2(ndarray: A): Value

  def max(ndarray: A): Value

  def min(ndarray: A): Value

  def standardDeviation(ndarray: A): Value

  def product(ndarray: A): Value

  def variance(ndarray: A): Value

  def add(a: A, that: INDArray): A

  def sub(a: A, that: INDArray): A

  def mul(a: A, that: INDArray): A

  def mmul(a: A, that: INDArray): A

  def div(a: A, that: INDArray): A

  def rdiv(a: A, that: INDArray): A

  def addi(a: A, that: INDArray): A

  def subi(a: A, that: INDArray): A

  def muli(a: A, that: INDArray): A

  def mmuli(a: A, that: INDArray): A

  def divi(a: A, that: INDArray): A

  def rdivi(a: A, that: INDArray): A

  def add(a: A, that: Number): A

  def sub(a: A, that: Number): A

  def mul(a: A, that: Number): A

  def div(a: A, that: Number): A

  def rdiv(a: A, that: Number): A

  def addi(a: A, that: Number): A

  def subi(a: A, that: Number): A

  def muli(a: A, that: Number): A

  def divi(a: A, that: Number): A

  def rdivi(a: A, that: Number): A

  def put(a: A, i: Int, element: INDArray): A

  def put(a: A, i: Array[Int], element: INDArray): A

  def get(a: A, i: Int): Value

  def get(a: A, i: Int, j: Int): Value

  def get(a: A, i: Int*): Value
}

trait RealNDArrayEvidence extends NDArrayEvidence[INDArray] {
  override def add(a: INDArray, that: INDArray): INDArray = a.add(that)

  override def div(a: INDArray, that: INDArray): INDArray = a.div(that)

  override def mul(a: INDArray, that: INDArray): INDArray = a.mul(that)

  override def rdiv(a: INDArray, that: INDArray): INDArray = a.rdiv(that)

  override def sub(a: INDArray, that: INDArray): INDArray = a.sub(that)

  override def mmul(a: INDArray, that: INDArray): INDArray = a.mmul(that)

  override def addi(a: INDArray, that: INDArray): INDArray = a.addi(that)

  override def subi(a: INDArray, that: INDArray): INDArray = a.subi(that)

  override def muli(a: INDArray, that: INDArray): INDArray = a.muli(that)

  override def mmuli(a: INDArray, that: INDArray): INDArray = a.mmuli(that)

  override def divi(a: INDArray, that: INDArray): INDArray = a.divi(that)

  override def rdivi(a: INDArray, that: INDArray): INDArray = a.rdivi(that)

  override def add(a: INDArray, that: Number): INDArray = a.add(that)

  override def sub(a: INDArray, that: Number): INDArray = a.sub(that)

  override def mul(a: INDArray, that: Number): INDArray = a.mul(that)

  override def div(a: INDArray, that: Number): INDArray = a.div(that)

  override def rdiv(a: INDArray, that: Number): INDArray = a.rdiv(that)

  override def addi(a: INDArray, that: Number): INDArray = a.addi(that)

  override def subi(a: INDArray, that: Number): INDArray = a.subi(that)

  override def muli(a: INDArray, that: Number): INDArray = a.muli(that)

  override def divi(a: INDArray, that: Number): INDArray = a.divi(that)

  override def rdivi(a: INDArray, that: Number): INDArray = a.rdivi(that)

  override def put(a: INDArray, i:Array[Int], element: INDArray): INDArray = a.put(i,element)

  override def put(a: INDArray, i: Int, element: INDArray): INDArray = a.put(i,element)
}

case object DoubleNDArrayEvidence extends RealNDArrayEvidence {
  override type Value = Double

  override def sum(ndarray: INDArray): Value = ndarray.sumNumber().doubleValue()

  override def mean(ndarray: INDArray): Value = ndarray.meanNumber().doubleValue()

  override def variance(ndarray: INDArray): Value = ndarray.varNumber().doubleValue()

  override def norm2(ndarray: INDArray): Value = ndarray.norm2Number().doubleValue()

  override def max(ndarray: INDArray): Value = ndarray.maxNumber().doubleValue()

  override def product(ndarray: INDArray): Value = ndarray.prodNumber().doubleValue()

  override def standardDeviation(ndarray: INDArray): Value = ndarray.stdNumber().doubleValue()

  override def normMax(ndarray: INDArray): Value = ndarray.normmaxNumber().doubleValue()

  override def min(ndarray: INDArray): Value = ndarray.minNumber().doubleValue()

  override def norm1(ndarray: INDArray): Value = ndarray.norm1Number().doubleValue()

  override def get(a: INDArray, i: Int): Value = a.getDouble(i)

  override def get(a: INDArray, i: Int, j: Int): Value = a.getDouble(i,j)

  override def get(a: INDArray, i: Int*): Value = a.getDouble(i:_*)
}

case object FloatNDArrayEvidence extends RealNDArrayEvidence {
  override type Value = Float

  override def sum(ndarray: INDArray): Value = ndarray.sumNumber().floatValue()

  override def mean(ndarray: INDArray): Value = ndarray.meanNumber().floatValue()

  override def variance(ndarray: INDArray): Value = ndarray.varNumber().floatValue()

  override def norm2(ndarray: INDArray): Value = ndarray.norm2Number().floatValue()

  override def max(ndarray: INDArray): Value = ndarray.maxNumber().floatValue()

  override def product(ndarray: INDArray): Value = ndarray.prodNumber().floatValue()

  override def standardDeviation(ndarray: INDArray): Value = ndarray.stdNumber().floatValue()

  override def normMax(ndarray: INDArray): Value = ndarray.normmaxNumber().floatValue()

  override def min(ndarray: INDArray): Value = ndarray.minNumber().floatValue()

  override def norm1(ndarray: INDArray): Value = ndarray.norm1Number().floatValue()

  override def get(a: INDArray, i: Int): Value = a.getFloat(i)

  override def get(a: INDArray, i: Int, j: Int): Value = a.getFloat(i,j)

  override def get(a: INDArray, i: Int*): Value = a.getFloat(i.toArray)
}

case object ComplexNDArrayEvidence extends NDArrayEvidence[IComplexNDArray] {
  override type Value = IComplexNumber

  override def sum(ndarray: IComplexNDArray): Value = ndarray.sumComplex()

  override def mean(ndarray: IComplexNDArray): Value = ndarray.meanComplex()

  override def variance(ndarray: IComplexNDArray): Value = ndarray.varComplex()

  override def norm2(ndarray: IComplexNDArray): Value = ndarray.norm2Complex()

  override def max(ndarray: IComplexNDArray): Value = ndarray.maxComplex()

  override def product(ndarray: IComplexNDArray): Value = ndarray.prodComplex()

  override def standardDeviation(ndarray: IComplexNDArray): Value = ndarray.stdComplex()

  override def normMax(ndarray: IComplexNDArray): Value = ndarray.normmaxComplex()

  override def min(ndarray: IComplexNDArray): Value = ndarray.minComplex()

  override def norm1(ndarray: IComplexNDArray): Value = ndarray.norm1Complex()

  override def add(a: IComplexNDArray, that: INDArray): IComplexNDArray = a.add(that)

  override def div(a: IComplexNDArray, that: INDArray): IComplexNDArray = a.div(that)

  override def mul(a: IComplexNDArray, that: INDArray): IComplexNDArray = a.mul(that)

  override def rdiv(a: IComplexNDArray, that: INDArray): IComplexNDArray = a.rdiv(that)

  override def sub(a: IComplexNDArray, that: INDArray): IComplexNDArray = a.sub(that)

  override def mmul(a: IComplexNDArray, that: INDArray): IComplexNDArray = a.mmul(that)

  override def addi(a: IComplexNDArray, that: INDArray): IComplexNDArray = a.addi(that)

  override def div(a: IComplexNDArray, that: Number): IComplexNDArray = a.div(that)

  override def addi(a: IComplexNDArray, that: Number): IComplexNDArray = a.addi(that)

  override def mul(a: IComplexNDArray, that: Number): IComplexNDArray = a.mul(that)

  override def rdivi(a: IComplexNDArray, that: INDArray): IComplexNDArray = a.rdivi(that)

  override def rdivi(a: IComplexNDArray, that: Number): IComplexNDArray = a.rdivi(that)

  override def divi(a: IComplexNDArray, that: INDArray): IComplexNDArray = a.divi(that)

  override def divi(a: IComplexNDArray, that: Number): IComplexNDArray = a.divi(that)

  override def rdiv(a: IComplexNDArray, that: Number): IComplexNDArray = a.rdiv(that)

  override def muli(a: IComplexNDArray, that: INDArray): IComplexNDArray = a.muli(that)

  override def muli(a: IComplexNDArray, that: Number): IComplexNDArray = a.muli(that)

  override def sub(a: IComplexNDArray, that: Number): IComplexNDArray = a.sub(that)

  override def subi(a: IComplexNDArray, that: INDArray): IComplexNDArray = a.subi(that)

  override def subi(a: IComplexNDArray, that: Number): IComplexNDArray = a.subi(that)

  override def add(a: IComplexNDArray, that: Number): IComplexNDArray = a.add(that)

  override def mmuli(a: IComplexNDArray, that: INDArray): IComplexNDArray = a.mmuli(that)

  override def get(a: IComplexNDArray, i: Int): Value = a.getComplex(i)

  override def get(a: IComplexNDArray, i: Int, j: Int): Value = a.getComplex(i,j)

  override def get(a: IComplexNDArray, i: Int*): Value = a.getComplex(i:_*)

  override def put(a: IComplexNDArray, i: Int, element: INDArray): IComplexNDArray = a.put(i,element)

  override def put(a: IComplexNDArray, i: Array[Int], element: INDArray): IComplexNDArray = a.put(i,element)
}