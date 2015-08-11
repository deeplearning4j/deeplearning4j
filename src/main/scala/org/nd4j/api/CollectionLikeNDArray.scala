package org.nd4j.api

import org.nd4j.api.ops._
import org.nd4j.api.Implicits._
import org.nd4j.linalg.api.complex.IComplexNumber
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

trait CollectionLikeNDArray {
  val underlying: INDArray
  
  def filteri(f: Double => Boolean): INDArray = notCleanedUp { array =>
    val shape = underlying.shape()
    Nd4j.getExecutioner.exec(FilterOps(underlying.linearView(), f)).z().reshape(shape:_*)
  }
  def filter(f: Double => Boolean): INDArray = underlying.dup().filteri(f)

  def filterBiti(f: Double => Boolean): INDArray = notCleanedUp { array =>
    val shape = underlying.shape()
    Nd4j.getExecutioner.exec(BitFilterOps(underlying.linearView(), f)).z().reshape(shape:_*)
  }

  def filterBit(f: Double => Boolean): INDArray = underlying.dup().filterBiti(f)

  def mapRCi(f: Double => Double)(g:IComplexNumber => IComplexNumber): INDArray = notCleanedUp { array =>
    Nd4j.getExecutioner.exec(MapOps(underlying.linearView(), f,g)).z().reshape(underlying.shape():_*)
  }
  def mapRC(f: Double => Double)(g:IComplexNumber => IComplexNumber): INDArray = underlying.dup().mapRCi(f)(g)

  def mapi(f: Double => Double): INDArray = mapRCi(f)(g => g)
  def map(f: Double => Double): INDArray = mapRC(f)(g => g)

  def mapCi(g:IComplexNumber => IComplexNumber): INDArray = mapRCi(f => f)(g)
  def mapC(g:IComplexNumber => IComplexNumber): INDArray = mapRC(f => f)(g)

  def notCleanedUp[A](f: INDArray => A): A = {
    if (underlying.isCleanedUp)
      throw new IllegalStateException("Invalid operation: already collected")
    f(underlying)
  }
}
