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
      val originalShape = underlying.shape()
      @tailrec
      def modifyTargetIndices(input: List[IndexRange], i: Int, acc: List[(Int, Int)]): List[(Int, Int)] = input match {
        case -> :: t => modifyTargetIndices(t, i + 1, 0 -> (originalShape(i) - 1) :: acc)
        case ---> :: t =>
          val ellipsised = List.fill(originalShape.length - i - t.size)(->)
          modifyTargetIndices(ellipsised ::: t, i, acc)
        case (inr: IndexNumberRange) :: t =>
          modifyTargetIndices(t, i + 1, inr.asTuple :: acc)
        case Nil => acc.reverse
      }

      val modifiedTarget = modifyTargetIndices(target.toList, 0, Nil)
      val targetShape = modifiedTarget.collect {
        case (start, end) => end - start + 1
      }

      @tailrec
      def calcIndices(tgt: List[(Int, Int)], orgShape: List[Int], layerSize: Int, acc: List[Int]): List[Int] = {
        (tgt, orgShape) match {
          case (h :: t, shape) if acc.isEmpty =>
            calcIndices(t, shape, layerSize, (h._1 to h._2).toList)
          case (h :: t, hs :: ts) =>
            val thisLayer = layerSize * hs
            calcIndices(t, ts, thisLayer, acc ++ acc.map(_ + thisLayer))
          case _ => acc
        }
      }

      val indices = calcIndices(modifiedTarget.toList, underlying.shape().toList, 1, Nil)
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
    override def asTuple: (Int, Int) = (underlying, underlying)
  }

  implicit class TupleRange(val underlying: _root_.scala.Tuple2[Int, Int]) extends IndexNumberRange {
    override def asTuple: (Int, Int) = underlying
  }

}

sealed trait IndexNumberRange extends IndexRange {
  def asTuple: (Int, Int)
}

sealed trait IndexRange
