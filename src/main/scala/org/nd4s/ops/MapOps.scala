package org.nd4j.api.ops

import org.nd4j.linalg.api.complex.IComplexNumber
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.{BaseScalarOp, BaseOp, Op}
import org.nd4j.api.Implicits._

object MapOps{
  def apply(x:INDArray,f:Double=>Double,g:IComplexNumber =>IComplexNumber):MapOps = new MapOps(x,f,g)
}
class MapOps(_x:INDArray,f:Double => Double, g:IComplexNumber => IComplexNumber) extends BaseScalarOp(_x,null,_x,_x.length(),0){
  x = _x
  def this(){
    this(0.toScalar,null,null)
  }

  override def name(): String = "map_scalar"

  override def opForDimension(index: Int, dimension: Int): Op = MapOps(x.tensorAlongDimension(index,dimension),f,g)

  override def opForDimension(index: Int, dimension: Int*): Op = MapOps(x.tensorAlongDimension(index,dimension:_*),f,g)

  override def op(origin: IComplexNumber, other: Double): IComplexNumber = op(origin)

  override def op(origin: IComplexNumber, other: Float): IComplexNumber = op(origin)

  override def op(origin: IComplexNumber, other: IComplexNumber): IComplexNumber = op(origin)

  override def op(origin: Float, other: Float): Float = op(origin)

  override def op(origin: Double, other: Double): Double = op(origin)

  override def op(origin: Double): Double = f(origin)

  override def op(origin: Float): Float = f(origin).toFloat

  override def op(origin: IComplexNumber): IComplexNumber = g(origin)
}
