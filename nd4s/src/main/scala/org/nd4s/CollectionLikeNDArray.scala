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
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.Op
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.ops.{ BitFilterOps, FilterOps, MapOps }

import scalaxy.loops._
import scala.language.postfixOps
import scala.util.control.Breaks._

/*
  This provides Scala Collection like APIs such as map, filter, exist, forall.
 */
trait CollectionLikeNDArray[A <: INDArray] {
  val underlying: A

  def filter(f: Double => Boolean)(implicit ev: NDArrayEvidence[A, _]): A = notCleanedUp { _ =>
    val shape = underlying.shape()
    ev.reshape(Nd4j.getExecutioner
                 .exec(FilterOps(ev.linearView(underlying), f): Op)
                 .z()
                 .asInstanceOf[A],
               shape.map(_.toInt): _*)
  }

  def filterBit(f: Double => Boolean)(implicit ev: NDArrayEvidence[A, _]): A = notCleanedUp { _ =>
    val shape = underlying.shape()
    ev.reshape(Nd4j.getExecutioner
                 .exec(BitFilterOps(ev.linearView(underlying), f): Op)
                 .z()
                 .asInstanceOf[A],
               shape.map(_.toInt): _*)
  }

  def map(f: Double => Double)(implicit ev: NDArrayEvidence[A, _]): A = notCleanedUp { _ =>
    val shape = underlying.shape()
    ev.reshape(Nd4j.getExecutioner
                 .exec(MapOps(ev.linearView(underlying), f): Op)
                 .z()
                 .asInstanceOf[A],
               shape.map(_.toInt): _*)
  }

  def notCleanedUp[B](f: INDArray => B): B =
    f(underlying)

  def exists(f: Double => Boolean)(implicit ev: NDArrayEvidence[A, Double]): Boolean = existsTyped[Double](f)

  def existsTyped[B](f: B => Boolean)(implicit ev: NDArrayEvidence[A, B]): Boolean = {
    var result = false
    val lv = ev.linearView(underlying)
    breakable {
      for {
        i <- 0 until lv.length().toInt optimized
      } if (!f(ev.get(lv, i))) {
        result = true
        break()
      }
    }
    result
  }

  def forall(f: Double => Boolean)(implicit ev: NDArrayEvidence[A, Double]): Boolean = forallTyped[Double](f)

  def forallTyped[B](f: B => Boolean)(implicit ev: NDArrayEvidence[A, B]): Boolean = {
    var result = true
    val lv = ev.linearView(underlying)
    breakable {
      for {
        i <- 0 until lv.length().toInt optimized
      } if (!f(ev.get(lv, i))) {
        result = false
        break()
      }
    }
    result
  }

  def >[B, C](d: C)(implicit ev: NDArrayEvidence[A, B], ev2: C => B): Boolean =
    forallTyped { i: B =>
      ev.greaterThan(i, d)
    }

  def <[B, C](d: C)(implicit ev: NDArrayEvidence[A, B], ev2: C => B): Boolean =
    forallTyped { i: B =>
      ev.lessThan(i, d)
    }

  def >=[B, C](d: C)(implicit ev: NDArrayEvidence[A, B], ev2: Equality[B], ev3: C => B): Boolean = forallTyped { i: B =>
    ev.greaterThan(i, d) || ev2.equal(i, d)
  }

  def <=[B, C](d: C)(implicit ev: NDArrayEvidence[A, B], ev2: Equality[B], ev3: C => B): Boolean = forallTyped { i: B =>
    ev.lessThan(i, d) || ev2.equal(i, d)
  }

  def columnP: ColumnProjectedNDArray = new ColumnProjectedNDArray(underlying)

  def rowP: RowProjectedNDArray = new RowProjectedNDArray(underlying)

  def sliceP: SliceProjectedNDArray = new SliceProjectedNDArray(underlying)
}
