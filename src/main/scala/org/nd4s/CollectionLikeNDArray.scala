package org.nd4s

import org.nd4s.ops._
import org.nd4s.Implicits._
import org.nd4j.linalg.api.complex.IComplexNumber
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.util.control.Breaks._

/*
  This provides Scala Collection like APIs such as map, filter, exist, forall.
 */
trait CollectionLikeNDArray[A <: INDArray] {
  val underlying: A

  def filterRCi(f: Double => Boolean)(g:IComplexNumber => Boolean)(implicit ev: NDArrayEvidence[A,_]): A = notCleanedUp { array =>
    val shape = underlying.shape()
    ev.reshape(Nd4j.getExecutioner.exec(FilterOps(ev.linearView(underlying), f,g)).z().asInstanceOf[A], shape: _*)
  }

  def filterRC(f: Double => Boolean)(g:IComplexNumber => Boolean)(implicit ev: NDArrayEvidence[A,_]): A = ev.dup(underlying).filterRCi(f)(g)

  def filteri(f: Double => Boolean)(implicit ev: NDArrayEvidence[A, Double]): A = filterRCi(f)(_ => false)

  def filter(f: Double => Boolean)(implicit ev: NDArrayEvidence[A, Double]): A = filterRC(f)(_ => false)

  def filterCi(f: IComplexNumber => Boolean)(implicit ev: NDArrayEvidence[A, IComplexNumber]): A = filterRCi(_ => false)(f)

  def filterC(f: IComplexNumber => Boolean)(implicit ev: NDArrayEvidence[A, IComplexNumber]): A = filterRC(_ => false)(f)

  def filterBitRCi(f: Double => Boolean)(g: IComplexNumber => Boolean)(implicit ev: NDArrayEvidence[A, _]): A = notCleanedUp { array =>
    val shape = underlying.shape()
    ev.reshape(Nd4j.getExecutioner.exec(BitFilterOps(ev.linearView(underlying), f, g)).z().asInstanceOf[A], shape: _*)
  }

  def filterBitRC(f: Double => Boolean)(g: IComplexNumber => Boolean)(implicit ev: NDArrayEvidence[A, _]): A = ev.dup(underlying).filterBitRCi(f)(g)

  def filterBitCi(f: IComplexNumber => Boolean)(implicit ev: NDArrayEvidence[A, IComplexNumber]): A = filterBitRCi(_ => false)(f)

  def filterBitC(f: IComplexNumber => Boolean)(implicit ev: NDArrayEvidence[A, IComplexNumber]): A = filterBitRC(_ => false)(f)

  def filterBiti(f: Double => Boolean)(implicit ev: NDArrayEvidence[A, Double]): A = filterBitRCi(f)(_ => false)

  def filterBit(f: Double => Boolean)(implicit ev: NDArrayEvidence[A, Double]): A = filterBitRC(f)(_ => false)

  def mapRCi(f: Double => Double)(g: IComplexNumber => IComplexNumber)(implicit ev: NDArrayEvidence[A, _]): A = notCleanedUp { array =>
    val shape = underlying.shape()
    ev.reshape(Nd4j.getExecutioner.exec(MapOps(ev.linearView(underlying), f, g)).z().asInstanceOf[A], shape: _*)
  }

  def mapRC(f: Double => Double)(g: IComplexNumber => IComplexNumber)(implicit ev: NDArrayEvidence[A, _]): A = ev.dup(underlying).mapRCi(f)(g)(ev)

  def mapi(f: Double => Double)(implicit ev: NDArrayEvidence[A, _]): A = mapRCi(f)(g => g)(ev)

  def map(f: Double => Double)(implicit ev: NDArrayEvidence[A, _]): A = mapRC(f)(g => g)(ev)

  def mapCi(g: IComplexNumber => IComplexNumber)(implicit ev: NDArrayEvidence[A, _]): A = mapRCi(f => f)(g)(ev)

  def mapC(g: IComplexNumber => IComplexNumber)(implicit ev: NDArrayEvidence[A, _]): A = mapRC(f => f)(g)(ev)

  def notCleanedUp[B](f: INDArray => B): B = {
    if (underlying.isCleanedUp)
      throw new IllegalStateException("Invalid operation: already collected")
    f(underlying)
  }

  def existsRC[B](f: B => Boolean)(implicit ev: NDArrayEvidence[A, B]): Boolean = {
    var result = false
    val lv = ev.linearView(underlying)
    breakable {
      for {
        i <- 0 until lv.length()
      } if (!f(ev.get(lv, i))) {
        result = true
        break()
      }
    }
    result
  }

  def exists(f: Double => Boolean)(implicit ev: NDArrayEvidence[A, Double]): Boolean = existsRC[Double](f)

  def existsC(f: IComplexNumber => Boolean)(implicit ev: NDArrayEvidence[A, IComplexNumber]): Boolean = existsRC[IComplexNumber](f)

  def forallRC[B](f: B => Boolean)(implicit ev: NDArrayEvidence[A, B]): Boolean = {
    var result = true
    val lv = ev.linearView(underlying)
    breakable {
      for {
        i <- 0 until lv.length()
      } if (!f(ev.get(lv, i))) {
        result = false
        break()
      }
    }
    result
  }

  def forall(f: Double => Boolean)(implicit ev: NDArrayEvidence[A, Double]): Boolean = forallRC[Double](f)

  def forallC(f: IComplexNumber => Boolean)(implicit ev: NDArrayEvidence[A, IComplexNumber]): Boolean = forallRC[IComplexNumber](f)

  def >[B, C](d: C)(implicit ev: NDArrayEvidence[A, B], ev2: C => B): Boolean = forallRC { i: B => ev.greaterThan(i, d) }

  def <[B, C](d: C)(implicit ev: NDArrayEvidence[A, B], ev2: C => B): Boolean = forallRC { i: B => ev.lessThan(i, d) }

  def >=[B, C](d: C)(implicit ev: NDArrayEvidence[A, B], ev2: Equality[B], ev3: C => B): Boolean = forallRC { i: B => ev.greaterThan(i, d) || ev2.equal(i, d) }

  def <=[B, C](d: C)(implicit ev: NDArrayEvidence[A, B], ev2: Equality[B], ev3: C => B): Boolean = forallRC { i: B => ev.lessThan(i, d) || ev2.equal(i, d) }

  def columnP: ColumnProjectedNDArray = new ColumnProjectedNDArray(underlying)

  def rowP: RowProjectedNDArray = new RowProjectedNDArray(underlying)

  def sliceP: SliceProjectedNDArray = new SliceProjectedNDArray(underlying)
}
