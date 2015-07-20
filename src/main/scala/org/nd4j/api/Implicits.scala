package org.nd4j.api

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import _root_.scala.annotation.tailrec
import _root_.scala.util.control.Breaks._

object Implicits {
  implicit class RichINDArray(val underlying: INDArray) extends AnyVal {
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

    /*
      Extract subMatrix at given position.
    */
    def subMatrix(target: IndexRange*): INDArray = {
      require(target.size <= underlying.shape().length, "Matrix dimension must be equal or larger than shape's dimension to extract.")
      val originalShape = if (underlying.shape().head != 1 || underlying.shape().length != 2)
        underlying.shape()
      else
        underlying.shape().drop(1)

      @tailrec
      def modifyTargetIndices(input: List[IndexRange], i: Int, acc: List[DRange]): List[DRange] = input match {
        case -> :: t => modifyTargetIndices(t, i + 1, DRange(0,originalShape(i) - 1,1) :: acc)
        case ---> :: t =>
          val ellipsised = List.fill(originalShape.length - i - t.size)(->)
          modifyTargetIndices(ellipsised ::: t, i, acc)
        case IntRangeFrom(from: Int) :: t =>
          val max = originalShape(i)
          modifyTargetIndices(t, i + 1, DRange(from , max - 1,1, max) :: acc)
        case (inr: IndexNumberRange) :: t =>
          modifyTargetIndices(t, i + 1, inr.asRange(originalShape(i)) :: acc)

        case Nil => acc.reverse
      }

      val modifiedTarget = modifyTargetIndices(target.toList, 0, Nil)
      val targetShape = modifiedTarget.map(_.length)

      @tailrec
      def calcIndices(tgt: List[DRange], orgShape: List[Int], layerSize: Int, acc: List[Int]): List[Int] = {
        (tgt, orgShape) match {
          case (h :: t, hs :: ts) if acc.isEmpty =>
            calcIndices(t, ts, layerSize, h.toList)
          case (h :: t, hs :: ts) =>
            val thisLayer = layerSize * hs
            calcIndices(t, ts, thisLayer, h.toList.flatMap { i => acc.map(_ + thisLayer * i)}.toList)
          case _ => acc
        }
      }
      val indices = calcIndices(modifiedTarget.toList, originalShape.toList, 1, Nil)

      val filtered = indices.map { i => underlying.getDouble(i)}

      Nd4j.create(filtered.toArray, targetShape.toArray.filterNot(_ <= 1))
    }

    def apply(target: IndexRange*): INDArray = subMatrix(target: _*)
  }


  implicit class floatColl2INDArray(val underlying: Seq[Float]) {
    def asNDArray(shape: Int*): INDArray = Nd4j.create(underlying.toArray, shape.toArray)
  }

  implicit class doubleColl2INDArray(val underlying: Seq[Double]) {
    def asNDArray(shape: Int*): INDArray = Nd4j.create(underlying.toArray, shape.toArray)
  }

  case object -> extends IndexRange

  case object ---> extends IndexRange

  implicit class IntRange(val underlying: Int) extends IndexNumberRange {
    protected[api] override def asRange(max: => Int): DRange = DRange(underlying, underlying, 1, max)

    override def toString: String = s"$underlying"
  }

  case class IntRangeFrom(underlying: Int) extends IndexRange {
    def apply(i: Int): (Int, Int) = (underlying, i)

    override def toString: String = s"$underlying->"
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

private[api] case class DRange(startR: Int, endR: Int, step: Int, max: Int) {
  lazy val (start, end) = {
    val start = if (startR >= 0) startR else max + startR
    val end = if (endR >= 0) endR else max + endR
    (start,end)
  }
  lazy val length = (end - start) / step + 1

  def toList: List[Int] = List.iterate(start, length)(_ + step)

  override def toString: String = s"${getClass.getSimpleName}(start:$start,end:$end,step:$step,length:$length)"
}

private[api] object DRange extends {
  def from(r: Range, max: => Int): DRange = DRange(r.start, r.end, r.step, max)
  def apply(startR: Int, endR: Int, step: Int):DRange = DRange(startR,endR,step,Int.MinValue)
}
