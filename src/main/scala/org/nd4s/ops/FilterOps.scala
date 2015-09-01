package org.nd4s.ops

import org.nd4j.linalg.api.complex.IComplexNumber
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.{BaseScalarOp, Op}
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

object FilterOps{
  def apply(x:INDArray,f:Double=>Boolean):FilterOps = new FilterOps(x,x.length(),f)
}
class FilterOps(_x:INDArray,len:Int,f:Double => Boolean) extends BaseScalarOp(_x,null:INDArray,_x,len,0){
  def this(){
    this(0.toScalar,0,null)
  }

  x = _x
  override def name(): String = "filter_scalar"

  override def opForDimension(index: Int, dimension: Int): Op = FilterOps(x.tensorAlongDimension(index,dimension),f)

  override def opForDimension(index: Int, dimension: Int*): Op = FilterOps(x.tensorAlongDimension(index,dimension:_*),f)

  override def op(origin: IComplexNumber, other: Double): IComplexNumber = op(origin)

  override def op(origin: IComplexNumber, other: Float): IComplexNumber = op(origin)

  override def op(origin: IComplexNumber, other: IComplexNumber): IComplexNumber = op(origin)

  override def op(origin: Float, other: Float): Float = op(origin)

  override def op(origin: Double, other: Double): Double = op(origin)

  override def op(origin: Double): Double = if(f(origin)) origin else 0

  override def op(origin: Float): Float = if(f(origin)) origin else 0

  override def op(origin: IComplexNumber): IComplexNumber = if(f(origin.absoluteValue().doubleValue())) origin else Nd4j.createComplexNumber(0, 0)
}
