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
    def subMatrix(target: Seq[Int]*): INDArray = {
      require(target.size <= underlying.shape().length, "Matrix dimension must be equal or larger than shape's dimension to extract.")

      val targetShape = target.map(_.size)

      @tailrec
      def calcIndices(tgt: List[Seq[Int]], orgShape: List[Int], layerSize: Int, acc: List[Int]): List[Int] = {
        (tgt, orgShape) match {
          case (h :: t, shape) if acc.isEmpty =>
            calcIndices(t, shape, layerSize, h.toList)
          case (h :: t, hs :: ts) =>
            val thisLayer = layerSize * hs
            calcIndices(t, ts, thisLayer, acc ++ acc.map(_ + thisLayer))
          case _ => acc
        }
      }

      val indices = calcIndices(target.toList, underlying.shape().toList, 1, Nil)
      val filtered = indices.map { i => underlying.getDouble(i)}
      Nd4j.create(filtered.toArray, targetShape.toArray.filterNot(_ <= 1))
    }

    def apply(target: Seq[Int]*): INDArray = subMatrix(target: _*)
  }

  implicit class RangedInt(val underlying: Int) extends AnyVal {
    def ~(end: Int): Seq[Int] = underlying to end
  }

  implicit class floatColl2INDArray(val underlying: Seq[Float]) {
    def asNDArray(shape: Int*): INDArray = Nd4j.create(underlying.toArray, shape.toArray)
  }

  implicit class doubleColl2INDArray(val underlying: Seq[Double]) {
    def asNDArray(shape: Int*): INDArray = Nd4j.create(underlying.toArray, shape.toArray)
  }
}
