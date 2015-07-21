package org.nd4j.api

import org.nd4j.api.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import _root_.scala.annotation.tailrec

trait SliceableNDArray {
  val underlying:INDArray
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
}
