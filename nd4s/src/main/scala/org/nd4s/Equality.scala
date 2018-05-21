package org.nd4s

import org.nd4j.linalg.api.complex.IComplexNumber

/**
 * Created by taisukeoe on 16/02/12.
 */
trait Equality[A]{
   def equal(left:A,right:A):Boolean
}
object Equality{
  implicit lazy val doubleEquality = new Equality[Double] {
    lazy val tolerance = 0.01D
    override def equal(left: Double, right: Double): Boolean = math.abs(left - right) < tolerance
  }
  implicit lazy val floatEquality = new Equality[Float] {
    lazy val tolerance = 0.01F
    override def equal(left: Float, right: Float): Boolean = math.abs(left - right) < tolerance
  }
  implicit lazy val complexEquality = new Equality[IComplexNumber] {
    lazy val tolerance = 0.01D
    override def equal(left: IComplexNumber, right: IComplexNumber): Boolean = math.abs(left.absoluteValue().doubleValue() - right.absoluteValue().doubleValue()) < tolerance
  }
}
