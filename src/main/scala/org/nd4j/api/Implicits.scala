package org.nd4j.api

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import _root_.scala.util.control.Breaks._

object Implicits {

  implicit class RichINDArray(val underlying: INDArray) extends SliceableNDArray with OperatableNDArray {
    def forall(f: Double => Boolean): Boolean = {
      var result = true
      breakable {
        for {
          i <- 0 until underlying.length()
        } if (!f(underlying.getDouble(i))) {
          result = false
          break()
        }
      }
      result
    }

    def >(d: Double): Boolean = forall(_ > d)

    def <(d: Double): Boolean = forall(_ < d)

    def >=(d: Double): Boolean = forall(_ >= d)

    def <=(d: Double): Boolean = forall(_ <= d)

    def apply(target: IndexRange*): INDArray = subMatrix(target: _*)
  }


  implicit class floatColl2INDArray(val underlying: Seq[Float]) extends AnyVal {
    def asNDArray(shape: Int*): INDArray = Nd4j.create(underlying.toArray, shape.toArray)
  }

  implicit class doubleColl2INDArray(val underlying: Seq[Double]) extends AnyVal {
    def asNDArray(shape: Int*): INDArray = Nd4j.create(underlying.toArray, shape.toArray)
  }

  implicit class int2Scalar(val underlying: Int) extends AnyVal {
    def asScalar: INDArray = Nd4j.scalar(underlying)
  }
  implicit class float2Scalar(val underlying: Float) extends AnyVal {
    def asScalar: INDArray = Nd4j.scalar(underlying)
  }
  implicit class double2Scalar(val underlying: Double) extends AnyVal {
    def asScalar: INDArray = Nd4j.scalar(underlying)
  }
  implicit class long2Scalar(val underlying: Long) extends AnyVal {
    def asScalar: INDArray = Nd4j.scalar(underlying)
  }

  case object -> extends IndexRange

  case object ---> extends IndexRange

  implicit class IntRange(val underlying: Int) extends IndexNumberRange {
    protected[api] override def asRange(max: => Int): DRange = DRange(underlying, underlying, 1, max)

    override def toString: String = s"$underlying"
  }

  implicit class TupleRange(val underlying: _root_.scala.Tuple2[Int, Int]) extends IndexNumberRange {
    protected[api] override def asRange(max: => Int): DRange = DRange(underlying._1, underlying._2, 1, max)

    override def toString: String = s"${underlying._1}->${underlying._2}"

    def by(i: Int) = new IndexRangeWrapper(underlying._1 to underlying._2 by i)
  }

  implicit class IntRangeFromGen(val underlying: Int) extends AnyVal {
    def -> = IntRangeFrom(underlying)
  }

  implicit class IndexRangeWrapper(val underlying: Range) extends IndexNumberRange {
    protected[api] override def asRange(max: => Int): DRange = DRange.from(underlying, max)
  }

}

sealed trait IndexNumberRange extends IndexRange {
  protected[api] def asRange(max: => Int): DRange
}

sealed trait IndexRange

case class IntRangeFrom(underlying: Int) extends IndexRange {
  def apply(i: Int): (Int, Int) = (underlying, i)

  override def toString: String = s"$underlying->"
}

private[api] case class DRange(startR: Int, endR: Int, step: Int, max: Int) {
  lazy val (start, end) = {
    val start = if (startR >= 0) startR else max + startR
    val end = if (endR >= 0) endR else max + endR
    (start, end)
  }
  lazy val length = (end - start) / step + 1

  def toList: List[Int] = List.iterate(start, length)(_ + step)

  override def toString: String = s"${getClass.getSimpleName}(start:$start,end:$end,step:$step,length:$length)"
}

private[api] object DRange extends {
  def from(r: Range, max: => Int): DRange = DRange(r.start, r.end, r.step, max)

  def apply(startR: Int, endR: Int, step: Int): DRange = DRange(startR, endR, step, Int.MinValue)
}
