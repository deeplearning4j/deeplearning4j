package org.nd4s.ops

import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.linalg.api.complex.IComplexNumber
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.{BaseScalarOp, BaseOp, Op}
import org.nd4s.Implicits._

object MapOps{
  def apply(x:INDArray,f:Double=>Double,g:IComplexNumber =>IComplexNumber):MapOps = new MapOps(x,f,g)
}
class MapOps(_x:INDArray,f:Double => Double, g:IComplexNumber => IComplexNumber) extends BaseScalarOp(_x,null,_x,_x.length(),0) with LeftAssociativeBinaryOp {
  x = _x
  def this(){
    this(0.toScalar,null,null)
  }

  override def opNum(): Int = -1

  override def opName(): String = "map_scalar"

  override def onnxName(): String = throw new UnsupportedOperationException

  override def tensorflowName(): String = throw new UnsupportedOperationException

  override def doDiff(f1: java.util.List[SDVariable]): java.util.List[SDVariable] = throw new UnsupportedOperationException

//  override def opForDimension(index: Int, dimension: Int): Op = MapOps(x.tensorAlongDimension(index,dimension),f,g)
//
//  override def opForDimension(index: Int, dimension: Int*): Op = MapOps(x.tensorAlongDimension(index,dimension:_*),f,g)

  override def op(origin: Double): Double = f(origin)

  override def op(origin: Float): Float = f(origin).toFloat

  override def op(origin: IComplexNumber): IComplexNumber = g(origin)
}
