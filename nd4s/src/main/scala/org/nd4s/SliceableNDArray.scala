/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4s

import org.nd4s.Implicits._
import org.nd4j.linalg.api.complex.{IComplexNDArray, IComplexNumber}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{NDArrayIndex, INDArrayIndex}
import org.slf4j.LoggerFactory

import _root_.scala.annotation.tailrec

trait SliceableNDArray [A <: INDArray] {
  lazy val log = LoggerFactory.getLogger(classOf[SliceableNDArray[A]])
  val underlying:A

  def apply[B](target: IndexRange*)(implicit ev:NDArrayEvidence[A,B],ev2:Manifest[B]):A = subMatrix(target: _*)(ev,ev2)

  /*
    Extract subMatrix at given position.
  */
  def subMatrix[B](target: IndexRange*)(implicit ev:NDArrayEvidence[A,B],ev2:Manifest[B]): A = {
    require(target.size <= underlying.shape().length, "Matrix dimension must be equal or larger than shape's dimension to extract.")

    if(target.exists(_.hasNegative)) {
      val SubMatrixIndices(indices,targetShape) = indicesFrom(target:_*)

      val lv = ev.linearView(underlying)
      val filtered = indices.map { i => ev.get(lv,i)}.toArray

      ev.create(filtered, targetShape, NDOrdering(underlying.ordering()),0)

    }else{
      ev.get(underlying,getINDArrayIndexfrom(target:_*):_*)
    }
  }

  def indicesFrom(target:IndexRange*):SubMatrixIndices = {
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
      indicesOnAxis.reduceLeftOption[List[Int]] { case (l, r) =>
        if (underlying.ordering() == NDOrdering.C.value)
          l.flatMap { i => r.map(_ + i)}
        else
          r.flatMap { i => l.map(_ + i)}
      }.getOrElse(List.empty)
    }

    val indices = calcIndices(modifiedTarget.toList, Nd4j.getStrides(originalShape,underlying.ordering()).toList)
    log.trace(s"${target.mkString("[", ",", "]")} means $modifiedTarget at ${originalShape.mkString("[", "x", s"]${underlying.ordering}")} matrix with stride:${underlying.stride.mkString(",")}. Target shape:${targetShape.mkString("[", "x", s"]${underlying.ordering}")} indices:$indices")
    SubMatrixIndices(indices,targetShape)
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
      case -> :: t =>
        val all =  NDArrayIndex.all()
        all.init(underlying,i)
        modifyTargetIndices(t, i + 1, all :: acc)
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

    val modifiedTarget = modifyTargetIndices(originalTarget.toList, 0, Nil)
    log.trace(s"${target.mkString("[", ",", "]")} means $modifiedTarget at ${originalShape.mkString("[", "x", s"]${underlying.ordering}")} matrix with stride:${underlying.stride.mkString(",")}.")
    modifiedTarget
  }

  def update[T,T1](indices:Array[IndexRange],num:T)(implicit ev:NDArrayEvidence[A,T1],ev2:T => T1):INDArray = ev.update(underlying,indices,num)
  def update[T,T1](i1:IndexRange,num:T)(implicit ev:NDArrayEvidence[A,T1],ev2:T => T1):INDArray = ev.update(underlying,Array(i1),num)
  def update[T,T1](i1:IndexRange,i2:IndexRange,num:T)(implicit ev:NDArrayEvidence[A,T1], ev2:T => T1):INDArray = ev.update(underlying,Array(i1,i2),num)
  def update[T,T1](i1:IndexRange,i2:IndexRange,i3:IndexRange,num:T)(implicit ev:NDArrayEvidence[A,T1], ev2:T => T1):INDArray = ev.update(underlying,Array(i1,i2,i3),num)
  def update[T,T1](i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,num:T)(implicit ev:NDArrayEvidence[A,T1], ev2:T => T1):INDArray =  ev.update(underlying,Array(i1,i2,i3,i4),num)
  def update[T,T1](i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,num:T)(implicit ev:NDArrayEvidence[A,T1], ev2:T => T1):INDArray =  ev.update(underlying,Array(i1,i2,i3,i4,i5),num)
  def update[T,T1](i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,i6:IndexRange,num:T)(implicit ev:NDArrayEvidence[A,T1], ev2:T => T1):INDArray = ev.update(underlying,Array(i1,i2,i3,i4,i5,i6),num)
  def update[T,T1](i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,i6:IndexRange,i7:IndexRange,num:T)(implicit ev:NDArrayEvidence[A,T1], ev2:T => T1):INDArray = ev.update(underlying,Array(i1,i2,i3,i4,i5,i6,i7),num)
  def update[T,T1](i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,i6:IndexRange,i7:IndexRange,i8:IndexRange,num:T)(implicit ev:NDArrayEvidence[A,T1], ev2:T => T1):INDArray = ev.update(underlying,Array(i1,i2,i3,i4,i5,i6,i7,i8),num)
  def update[T,T1](i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,i6:IndexRange,i7:IndexRange,i8:IndexRange,i9:IndexRange,num:T)(implicit ev:NDArrayEvidence[A,T1], ev2:T => T1):INDArray = ev.update(underlying,Array(i1,i2,i3,i4,i5,i6,i7,i8,i9),num)
  def update[T,T1](i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,i6:IndexRange,i7:IndexRange,i8:IndexRange,i9:IndexRange,i10:IndexRange,num:T)(implicit ev:NDArrayEvidence[A,T1], ev2:T => T1):INDArray  = ev.update(underlying,Array(i1,i2,i3,i4,i5,i6,i7,i8,i9,i10),num)
  def update[T,T1](i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,i6:IndexRange,i7:IndexRange,i8:IndexRange,i9:IndexRange,i10:IndexRange,i11:IndexRange,num:T)(implicit ev:NDArrayEvidence[A,T1], ev2:T => T1):INDArray = ev.update(underlying,Array(i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11),num)
  def update[T,T1](i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,i6:IndexRange,i7:IndexRange,i8:IndexRange,i9:IndexRange,i10:IndexRange,i11:IndexRange,i12:IndexRange,num:T)(implicit ev:NDArrayEvidence[A,T1], ev2:T => T1):INDArray = ev.update(underlying,Array(i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12),num)

  def update(indices:Array[IndexRange],num:A)(implicit ev:NDArrayEvidence[A,_]):INDArray = ev.update(underlying,indices,num)
  def update(i1:IndexRange,num:A)(implicit ev:NDArrayEvidence[A,_]):A = ev.update(underlying,Array(i1),num)
  def update(i1:IndexRange,i2:IndexRange,num:A)(implicit ev:NDArrayEvidence[A,_]):A = ev.update(underlying,Array(i1,i2),num)
  def update(i1:IndexRange,i2:IndexRange,i3:IndexRange,num:A)(implicit ev:NDArrayEvidence[A,_]):A = ev.update(underlying,Array(i1,i2,i3),num)
  def update(i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,num:A)(implicit ev:NDArrayEvidence[A,_]):A =  ev.update(underlying,Array(i1,i2,i3,i4),num)
  def update(i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,num:A)(implicit ev:NDArrayEvidence[A,_]):A =  ev.update(underlying,Array(i1,i2,i3,i4,i5),num)
  def update(i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,i6:IndexRange,num:A)(implicit ev:NDArrayEvidence[A,_]):A = ev.update(underlying,Array(i1,i2,i3,i4,i5,i6),num)
  def update(i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,i6:IndexRange,i7:IndexRange,num:A)(implicit ev:NDArrayEvidence[A,_]):A = ev.update(underlying,Array(i1,i2,i3,i4,i5,i6,i7),num)
  def update(i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,i6:IndexRange,i7:IndexRange,i8:IndexRange,num:A)(implicit ev:NDArrayEvidence[A,_]):A = ev.update(underlying,Array(i1,i2,i3,i4,i5,i6,i7,i8),num)
  def update(i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,i6:IndexRange,i7:IndexRange,i8:IndexRange,i9:IndexRange,num:A)(implicit ev:NDArrayEvidence[A,_]):A = ev.update(underlying,Array(i1,i2,i3,i4,i5,i6,i7,i8,i9),num)
  def update(i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,i6:IndexRange,i7:IndexRange,i8:IndexRange,i9:IndexRange,i10:IndexRange,num:A)(implicit ev:NDArrayEvidence[A,_]):A  = ev.update(underlying,Array(i1,i2,i3,i4,i5,i6,i7,i8,i9,i10),num)
  def update(i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,i6:IndexRange,i7:IndexRange,i8:IndexRange,i9:IndexRange,i10:IndexRange,i11:IndexRange,num:A)(implicit ev:NDArrayEvidence[A,_]):A = ev.update(underlying,Array(i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11),num)
  def update(i1:IndexRange,i2:IndexRange,i3:IndexRange,i4:IndexRange,i5:IndexRange,i6:IndexRange,i7:IndexRange,i8:IndexRange,i9:IndexRange,i10:IndexRange,i11:IndexRange,i12:IndexRange,num:A)(implicit ev:NDArrayEvidence[A,_]):A = ev.update(underlying,Array(i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12),num)
}
case class SubMatrixIndices(indices:List[Int],targetShape:Array[Int])
