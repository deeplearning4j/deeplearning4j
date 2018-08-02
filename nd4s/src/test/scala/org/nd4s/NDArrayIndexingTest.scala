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
import org.nd4j.linalg.indexing.{NDArrayIndexAll, PointIndex, IntervalIndex}
import org.scalatest.FlatSpec


class NDArrayIndexingTest extends FlatSpec{
  "IndexRange" should "convert -> DSL to indices" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).mkNDArray(NDOrdering.C)

    val indices = ndArray.indicesFrom(0->2,->)
    assert(indices.indices == List(0,1,2,3,4,5))
    assert(indices.targetShape.toList == List(2,3))
  }
  
  it should "convert -> DSL to NDArrayIndex interval with stride 1 or 2" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).mkNDArray(NDOrdering.C)

    val indices = ndArray.getINDArrayIndexfrom(0->2,0->3 by 2)
    val rowI = indices(0)
    assert(rowI.isInstanceOf[IntervalIndex])
    assert(rowI.hasNext)
    assert(rowI.next() == 0)
    assert(rowI.next() == 1)
    assert(!rowI.hasNext)

    val columnI = indices(1)
    assert(columnI.isInstanceOf[IntervalIndex])
    assert(columnI.hasNext)
    assert(columnI.next() == 0)
    assert(columnI.next() == 2)
    assert(!columnI.hasNext)
  }
  it should "convert -> DSL to NDArrayIndex point,all" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).mkNDArray(NDOrdering.C)

    val indices = ndArray.getINDArrayIndexfrom(0,->)
    val rowI = indices(0)
    assert(rowI.isInstanceOf[PointIndex])
    assert(rowI.hasNext)
    assert(rowI.next() == 0)
    assert(!rowI.hasNext)

    val columnI = indices(1)
    assert(columnI.isInstanceOf[NDArrayIndexAll])
    assert(columnI.hasNext)
    assert(columnI.next() == 0)
    assert(columnI.next() == 1)
    assert(columnI.next() == 2)
    assert(!columnI.hasNext)
  }
}
