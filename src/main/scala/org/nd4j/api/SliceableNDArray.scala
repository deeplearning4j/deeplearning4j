package org.nd4j.api

import org.nd4j.api.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.{NDArrayIndex, INDArrayIndex}
import org.slf4j.LoggerFactory

import _root_.scala.annotation.tailrec

trait SliceableNDArray {
  lazy val log = LoggerFactory.getLogger(classOf[SliceableNDArray])
  val underlying: INDArray

  /*
    Extract subMatrix at given position.
  */
  def subMatrix(target: IndexRange*): INDArray = {
    require(target.size <= underlying.shape().length, "Matrix dimension must be equal or larger than shape's dimension to extract.")

    if(underlying.isRowVector || target.exists(_.hasNegative)) {
      val originalShape = if (underlying.isRowVector && underlying.shape().length == 1)
        1 +: underlying.shape()
      else
        underlying.shape()

      val originalTarget = if (underlying.isRowVector && target.size == 1)
        IntRange(0) +: target
      else
        target

      @tailrec
      def modifyTargetIndices(input: List[IndexRange], i: Int, acc: List[DRange]): List[DRange] = input match {
        case -> :: t => modifyTargetIndices(t, i + 1, DRange(0, originalShape(i), 1) :: acc)
        case ---> :: t =>
          val ellipsised = List.fill(originalShape.length - i - t.size)(->)
          modifyTargetIndices(ellipsised ::: t, i, acc)
        case IntRangeFrom(from: Int) :: t =>
          val max = originalShape(i)
          modifyTargetIndices(t, i + 1, DRange(from, max, false, 1, max) :: acc)
        case (inr: IndexNumberRange) :: t =>
          modifyTargetIndices(t, i + 1, inr.asRange(originalShape(i)) :: acc)

        case Nil =>
          acc.reverse
      }

      val modifiedTarget = modifyTargetIndices(originalTarget.toList, 0, Nil)

      val targetShape = modifiedTarget.map(_.length).toArray

      def calcIndices(tgt: List[DRange], stride: List[Int]): List[Int] = {
        val indicesOnAxis = (tgt zip stride).collect {
          case (range, st) => range.toList.map(_ * st)
        }
        indicesOnAxis.reduceLeft[List[Int]] { case (l, r) =>
          if (underlying.ordering() == NDOrdering.C.value)
            l.flatMap { i => r.map(_ + i)}
          else
            r.flatMap { i => l.map(_ + i)}
        }
      }

      val indices = calcIndices(modifiedTarget.toList, underlying.stride().toList)

      log.trace(s"${target.mkString("[", ",", "]")} means $modifiedTarget at ${originalShape.mkString("[", "x", s"]${underlying.ordering}")} matrix with stride:${underlying.stride.mkString(",")}. Target shape:${targetShape.mkString("[", "x", s"]${underlying.ordering}")} indices:$indices")

      val lv = underlying.linearView()
      val filtered = indices.map { i => lv.getDouble(i)}

      filtered.mkNDArray(targetShape, NDOrdering(underlying.ordering()), 0)

    }else{
      underlying.get(getINDArrayIndexfrom(target:_*):_*)
    }
  }

  def getINDArrayIndexfrom(target: IndexRange*):List[INDArrayIndex] ={
    val originalShape = if (underlying.isRowVector && underlying.shape().length == 1)
      1 +: underlying.shape()
    else
      underlying.shape()

    val originalTarget = if (underlying.isRowVector && target.size == 1)
      IntRange(0) +: target
    else
      target

    @tailrec
    def modifyTargetIndices(input: List[IndexRange], i: Int, acc: List[INDArrayIndex]): List[INDArrayIndex] = input match {
      case -> :: t => modifyTargetIndices(t, i + 1, NDArrayIndex.all() :: acc)
      case ---> :: t =>
        val ellipsised = List.fill(originalShape.length - i - t.size)(->)
        modifyTargetIndices(ellipsised ::: t, i, acc)
      case IntRangeFrom(from: Int) :: t =>
        val max = originalShape(i)
        modifyTargetIndices(t, i + 1, IndexNumberRange.toNDArrayIndex(from, max, false, 1, max) :: acc)
      case (inr: IndexNumberRange) :: t =>
        modifyTargetIndices(t, i + 1, inr.asNDArrayIndex(originalShape(i)) :: acc)

      case Nil =>
        acc.reverse
    }

    modifyTargetIndices(originalTarget.toList, 0, Nil)
  }
}