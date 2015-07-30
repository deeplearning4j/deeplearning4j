package org.nd4j.api

import org.nd4j.linalg.api.complex.{IComplexNumber, IComplexNDArray}
import org.nd4j.linalg.api.ndarray.INDArray


object Evidences {
  implicit val double = DoubleNDArrayEvidence
  implicit val float = DoubleNDArrayEvidence
  implicit val complex = ComplexNDArrayEvidence
}
object NDArrayEvidence{
  implicit val doubleNDArrayEvidence = DoubleNDArrayEvidence

  implicit val complexNDArrayEvidence = ComplexNDArrayEvidence
}

trait NDArrayEvidence[A]{
  type T

  def sum(ndarray: A): T

  def mean(ndarray: A): T

  def normMax(ndarray: A): T

  def norm1(ndarray: A): T

  def norm2(ndarray: A): T

  def max(ndarray: A): T

  def min(ndarray: A): T

  def standardDeviation(ndarray: A): T

  def product(ndarray: A): T

  def variance(ndarray: A): T
}

case object DoubleNDArrayEvidence extends NDArrayEvidence[INDArray] {
  override type T = Double

  override def sum(ndarray: INDArray): T = ndarray.sumNumber().doubleValue()

  override def mean(ndarray: INDArray): T = ndarray.meanNumber().doubleValue()

  override def variance(ndarray: INDArray): T = ndarray.varNumber().doubleValue()

  override def norm2(ndarray: INDArray): T = ndarray.norm2Number().doubleValue()

  override def max(ndarray: INDArray): T = ndarray.maxNumber().doubleValue()

  override def product(ndarray: INDArray): T = ndarray.prodNumber().doubleValue()

  override def standardDeviation(ndarray: INDArray): T = ndarray.stdNumber().doubleValue()

  override def normMax(ndarray: INDArray): T = ndarray.normmaxNumber().doubleValue()

  override def min(ndarray: INDArray): T = ndarray.minNumber().doubleValue()

  override def norm1(ndarray: INDArray): T = ndarray.norm1Number().doubleValue()
}

case object FloatNDArrayEvidence extends NDArrayEvidence[INDArray] {
  override type T = Float

  override def sum(ndarray: INDArray): T = ndarray.sumNumber().floatValue()

  override def mean(ndarray: INDArray): T = ndarray.meanNumber().floatValue()

  override def variance(ndarray: INDArray): T = ndarray.varNumber().floatValue()

  override def norm2(ndarray: INDArray): T = ndarray.norm2Number().floatValue()

  override def max(ndarray: INDArray): T = ndarray.maxNumber().floatValue()

  override def product(ndarray: INDArray): T = ndarray.prodNumber().floatValue()

  override def standardDeviation(ndarray: INDArray): T = ndarray.stdNumber().floatValue()

  override def normMax(ndarray: INDArray): T = ndarray.normmaxNumber().floatValue()

  override def min(ndarray: INDArray): T = ndarray.minNumber().floatValue()

  override def norm1(ndarray: INDArray): T = ndarray.norm1Number().floatValue()
}

case object ComplexNDArrayEvidence extends NDArrayEvidence[IComplexNDArray] {
  override type T = IComplexNumber

  override def sum(ndarray: IComplexNDArray): T = ndarray.sumComplex()

  override def mean(ndarray: IComplexNDArray): T = ndarray.meanComplex()

  override def variance(ndarray: IComplexNDArray): T = ndarray.varComplex()

  override def norm2(ndarray: IComplexNDArray): T = ndarray.norm2Complex()

  override def max(ndarray: IComplexNDArray): T = ndarray.maxComplex()

  override def product(ndarray: IComplexNDArray): T = ndarray.prodComplex()

  override def standardDeviation(ndarray: IComplexNDArray): T = ndarray.stdComplex()

  override def normMax(ndarray: IComplexNDArray): T = ndarray.normmaxComplex()

  override def min(ndarray: IComplexNDArray): T = ndarray.minComplex()

  override def norm1(ndarray: IComplexNDArray): T = ndarray.norm1Complex()
}