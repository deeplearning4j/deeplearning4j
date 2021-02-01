/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4s

import org.nd4s.Implicits._
import org.scalatest.{ FlatSpec, Matchers }

class NDArrayProjectionAPITest extends FlatSpec {
  "ColumnProjectedNDArray" should "map column correctly" in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    val result = for {
      c <- ndArray.columnP
      if c.get(0) % 2 == 0
    } yield c * c

    assert(
      result == Array(
        Array(4d),
        Array(25d),
        Array(64d)
      ).toNDArray
    )
  }

  "ColumnProjectedNDArray" should "map column correctly 2" in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    val result = ndArray.columnP map (input => input + 1)
    assert(
      result == Array(
        Array(2d, 3d, 4d),
        Array(5d, 6d, 7d),
        Array(8d, 9d, 10d)
      ).toNDArray
    )
  }

  "ColumnProjectedNDArray" should "map column correctly 3" in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    val result = ndArray.columnP flatMap (input => input + 1)
    assert(
      result == Array(
        Array(2d, 3d, 4d),
        Array(5d, 6d, 7d),
        Array(8d, 9d, 10d)
      ).toNDArray
    )
  }

  "ColumnProjectedNDArray" should "map column correctly in place " in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    ndArray.columnP flatMapi (input => input + 1)
    assert(
      ndArray == Array(
        Array(2d, 3d, 4d),
        Array(5d, 6d, 7d),
        Array(8d, 9d, 10d)
      ).toNDArray
    )
  }

  "ColumnProjectedNDArray" should "map column correctly 4" in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    val result = ndArray.columnP map (input => input + 1)
    assert(
      result == Array(
        Array(2d, 3d, 4d),
        Array(5d, 6d, 7d),
        Array(8d, 9d, 10d)
      ).toNDArray
    )
  }

  "ColumnProjectedNDArray" should "map column correctly 5" in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    ndArray.columnP mapi (input => input + 1)
    assert(
      ndArray == Array(
        Array(2d, 3d, 4d),
        Array(5d, 6d, 7d),
        Array(8d, 9d, 10d)
      ).toNDArray
    )
  }

  "ColumnProjectedNDArray" should "flatmap column correctly" in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    val result = ndArray.columnP withFilter (input => false)
    assert(result.filtered.isEmpty)
  }

  "RowProjectedNDArray" should "map row correctly in for loop " in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    val result = for {
      c <- ndArray.rowP
      if c.get(0) % 2 == 0
    } yield c * c

    assert(
      result ==
        Array(Array(16d, 25d, 36d)).toNDArray
    )
  }

  "RowProjectedNDArray" should "map row correctly  " in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    val result = ndArray.rowP map (input => input / 2)

    assert(
      result ==
        Array[Double](0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 4.5000).toNDArray.reshape(3, 3)
    )
  }

  "RowProjectedNDArray" should "filter rows correctly  " in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    val result = ndArray.rowP withFilter (input => false)
    assert(result.filtered.isEmpty)
  }

  "RowProjectedNDArray" should "flatMap rows correctly  " in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    val result = ndArray.rowP flatMap (input => input + 1)
    val expected =
      Array(
        Array(2d, 3d, 4d),
        Array(5d, 6d, 7d),
        Array(8d, 9d, 10d)
      ).toNDArray

    assert(result == expected)
  }

  "RowProjectedNDArray" should "map row correctly 2 " in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    val result = ndArray.rowP map (input => input / 2)

    assert(
      result ==
        Array[Double](0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 4.5000).toNDArray.reshape(3, 3)
    )
  }

  "RowProjectedNDArray" should "flatMap in place rows correctly  " in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    ndArray.rowP flatMapi (input => input + 1)
    val expected =
      Array(
        Array(2d, 3d, 4d),
        Array(5d, 6d, 7d),
        Array(8d, 9d, 10d)
      ).toNDArray

    assert(ndArray == expected)
  }

  "RowProjectedNDArray" should "map in place rows correctly " in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    ndArray.rowP mapi (input => input / 2)

    assert(
      ndArray ==
        Array[Double](0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 4.5000).toNDArray.reshape(3, 3)
    )
  }

  "SliceProjectedNDArray" should "map slice correctly" in {
    val ndArray =
      (1d to 8d by 1).asNDArray(2, 2, 2)

    val result = for {
      slice <- ndArray.sliceP
      if slice.get(0) > 1
    } yield slice * slice

    assert(result == List(25d, 36d, 49d, 64d).asNDArray(1, 2, 2))
  }

  "SliceProjectedNDArray" should "flatmap slice correctly" in {
    val ndArray =
      (1d to 8d by 1).asNDArray(2, 2, 2)

    val result = ndArray.sliceP flatMap (input => input * 2)
    val expected =
      (2d to 16d by 2).asNDArray(2, 2, 2)
    assert(result == expected)
  }

  "SliceProjectedNDArray" should "flatmap slice correctly in place" in {
    val ndArray =
      (1d to 8d by 1).asNDArray(2, 2, 2)

    ndArray.sliceP flatMapi (input => input * 2)
    val expected =
      (2d to 16d by 2).asNDArray(2, 2, 2)
    assert(ndArray == expected)
  }

  "SliceProjectedNDArray" should "map slice correctly in place" in {
    val ndArray =
      (1d to 8d by 1).asNDArray(2, 2, 2)

    ndArray.sliceP mapi (input => input * 2)
    val expected =
      (2d to 16d by 2).asNDArray(2, 2, 2)
    assert(ndArray == expected)
  }

  "SliceProjectedNDArray" should "filter slice correctly" in {
    val ndArray = (1d until 9d by 1).asNDArray(2, 2, 2)
    val result = ndArray.sliceP withFilter (input => false)
    assert(result.filtered.isEmpty)
  }
}
