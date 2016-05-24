package org.nd4s.ops

import org.nd4j.linalg.api.complex.IComplexNumber
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.{BaseScalarOp, Op}
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

trait LeftAssociativeBinaryOp {

  def op(origin: IComplexNumber, other: Double): IComplexNumber = op(origin)

  def op(origin: IComplexNumber, other: Float): IComplexNumber = op(origin)

  def op(origin: IComplexNumber, other: IComplexNumber): IComplexNumber = op(origin)

  def op(origin: Float, other: Float): Float = op(origin)

  def op(origin: Double, other: Double): Double = op(origin)

  def op(origin: Double): Double

  def op(origin: Float): Float

  def op(origin: IComplexNumber): IComplexNumber
}
