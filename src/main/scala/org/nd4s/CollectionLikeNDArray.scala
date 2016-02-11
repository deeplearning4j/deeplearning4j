package org.nd4s

import org.nd4s.ops._
import org.nd4s.Implicits._
import org.nd4j.linalg.api.complex.IComplexNumber
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.util.control.Breaks._

trait CollectionLikeNDArray [A <: INDArray]{
  val underlying: A
  
  def filteri(f: Double => Boolean)(implicit ev:NDArrayEvidence[A,Double]): A = notCleanedUp { array =>
    val shape = underlying.shape()
    ev.reshape(Nd4j.getExecutioner.exec(FilterOps(ev.linearView(underlying), f)).z().asInstanceOf[A],shape:_*)
  }
  def filter(f: Double => Boolean)(implicit ev:NDArrayEvidence[A,Double]): A = ev.dup(underlying).filteri(f)(ev)

  def filterBiti(f: Double => Boolean)(implicit ev:NDArrayEvidence[A,_]): A = notCleanedUp { array =>
    val shape = underlying.shape()
    ev.reshape(Nd4j.getExecutioner.exec(BitFilterOps(underlying.linearView(), f)).z().asInstanceOf[A],shape:_*)
  }

  def filterBit(f: Double => Boolean)(implicit ev:NDArrayEvidence[A,_]): A = ev.dup(underlying).filterBiti(f)(ev)

  def mapRCi(f: Double => Double)(g:IComplexNumber => IComplexNumber)(implicit ev:NDArrayEvidence[A,_]): A = notCleanedUp { array =>
    val shape = underlying.shape()
    ev.reshape(Nd4j.getExecutioner.exec(MapOps(underlying.linearView(), f,g)).z().asInstanceOf[A],shape:_*)
  }
  def mapRC(f: Double => Double)(g:IComplexNumber => IComplexNumber)(implicit ev:NDArrayEvidence[A,_]): A = ev.dup(underlying).mapRCi(f)(g)(ev)

  def mapi(f: Double => Double)(implicit ev:NDArrayEvidence[A,_]): A = mapRCi(f)(g => g)(ev)
  def map(f: Double => Double)(implicit ev:NDArrayEvidence[A,_]): A = mapRC(f)(g => g)(ev)

  def mapCi(g:IComplexNumber => IComplexNumber)(implicit ev:NDArrayEvidence[A,_]):A = mapRCi(f => f)(g)(ev)
  def mapC(g:IComplexNumber => IComplexNumber)(implicit ev:NDArrayEvidence[A,_]):A = mapRC(f => f)(g)(ev)

  def notCleanedUp[B](f: INDArray => B): B = {
    if (underlying.isCleanedUp)
      throw new IllegalStateException("Invalid operation: already collected")
    f(underlying)
  }

  def forall(f: Double => Boolean): Boolean = {
    var result = true
    val lv = underlying.linearView()
    breakable {
      for {
        i <- 0 until lv.length()
      } if (!f(lv.getDouble(i))) {
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

  def columnP:ColumnProjectedNDArray = new ColumnProjectedNDArray(underlying)

  def rowP:RowProjectedNDArray = new RowProjectedNDArray(underlying)

  def sliceP:SliceProjectedNDArray = new SliceProjectedNDArray(underlying)
}
