package org.nd4j.api

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import _root_.scala.util.control.Breaks._

object Implicits {

  implicit class RichINDArray(val underlying: INDArray) extends AnyVal {
    def forall(f: Double => Boolean): Boolean = {
      val row = underlying.rows()
      val column = underlying.columns()
      var result = true
      breakable {
        for {
          r <- 0 until row
          c <- 0 until column
        } if (!f(underlying.getDouble(r, c))) {
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

    def apply(row: Seq[Int], column: Seq[Int]): INDArray = {
      val targetRows = row.intersect(0 until underlying.rows())
      val targetColumns = column.intersect(0 until underlying.columns())
      val data = for {
        c <- targetColumns
        r <- targetRows
      } yield underlying.getDouble(r, c)
      val shape = Array(targetRows.size, targetColumns.size)
      Nd4j.create(data.toArray, shape)
    }
  }

  implicit class RangedInt(val underlying: Int) extends AnyVal {
    def ~(end: Int): Seq[Int] = underlying to end
  }
}
