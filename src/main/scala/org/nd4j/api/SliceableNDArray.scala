package org.nd4j.api

import org.nd4j.api.Implicits._
import org.nd4j.api.NDOrdering.Fortran
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import _root_.scala.annotation.tailrec

trait SliceableNDArray {
  val underlying: INDArray

  /*
 Extract subMatrix at given position.
  */
  def subMatrix(target: IndexRange*): INDArray = {
    require(target.size <= underlying.shape().length, "Matrix dimension must be equal or larger than shape's dimension to extract.")
    val originalShape = if (underlying.isRowVector)
      underlying.shape().drop(1)
    else
      underlying.shape()

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
        if(underlying.ordering() == NDOrdering.Fortran.value)
          acc
        else
          acc.reverse
    }

    val modifiedTarget = modifyTargetIndices(target.toList, 0, Nil)

    val targetShape =
      if (underlying.isRowVector)
        Array(1, modifiedTarget.head.length)
      else
        modifiedTarget.map(_.length).toArray

    def calcIndices(tgt: List[DRange], stride: List[Int]): List[Int] =
      (tgt zip stride).collect {
        case (range, st) => range.toList.map(_ * st)
      }.reduceLeft[List[Int]] { case (l, r) => l.flatMap { i => r.map(_ + i)}}

    val indices = calcIndices(modifiedTarget.toList, underlying.stride().toList)

    val lv = underlying.linearView()
    val filtered = indices.map { i => lv.getDouble(i)}

    filtered.mkNDArray(targetShape, NDOrdering(underlying.ordering()),0)
  }
}
