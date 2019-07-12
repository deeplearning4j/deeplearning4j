/*******************************************************************************
  * Copyright (c) 2015-2019 Skymind, Inc.
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

import org.nd4j.linalg.api.buffer.DataType
import org.nd4s.Implicits._
import org.scalatest.FlatSpec

class NDArrayConstructionTest extends FlatSpec with COrderingForTest {
  self: OrderingForTest =>

  it should "be able to create 2d matrix filled with integers" in {
    val ndArray =
      Array(
        Array(1, 2),
        Array(4, 5),
        Array(7, 9)
      ).mkNDArray(ordering)

    assert(DataType.INT == ndArray.dataType())
    assert(3 == ndArray.rows())
    assert(2 == ndArray.columns())
  }

  it should "be able to create 2d matrix filled with long integers" in {
    val ndArray =
      Array(
        Array(1L, 2L, 3L),
        Array(4L, 5L, 6L),
        Array(7L, 8L, 9L)
      ).mkNDArray(ordering)

    assert(DataType.LONG == ndArray.dataType())
    assert(3 == ndArray.rows())
    assert(3 == ndArray.columns())
  }

  it should "be able to create 2d matrix filled with float numbers" in {
    val ndArray =
      Array(
        Array(1f, 2f, 3f),
        Array(4f, 5f, 6f),
        Array(7f, 8f, 9f)
      ).mkNDArray(ordering)

    assert(DataType.FLOAT == ndArray.dataType())
    assert(3 == ndArray.rows())
    assert(3 == ndArray.columns())
  }

  it should "be able to create 2d matrix filled with double numbers" in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).mkNDArray(ordering)

    assert(DataType.DOUBLE == ndArray.dataType())
    assert(3 == ndArray.rows())
    assert(3 == ndArray.columns())
  }

  it should "be able to create vector filled with short integers" in {
    val ndArray = Array[Short](1, 2, 3).toNDArray

    assert(DataType.SHORT == ndArray.dataType())
    assert(1 == ndArray.rows())
    assert(3 == ndArray.columns())
  }

  it should "be able to create vector filled with byte values" in {
    val ndArray = Array[Byte](1, 2, 3).toNDArray

    assert(DataType.BYTE == ndArray.dataType())
    assert(1 == ndArray.rows())
    assert(3 == ndArray.columns())
  }

  it should "be able to create vector filled with boolean values" in {
    val ndArray = Array(true, false, true).toNDArray

    assert(DataType.BOOL == ndArray.dataType())
    assert(1 == ndArray.rows())
    assert(3 == ndArray.columns())
  }

  it should "be able to create vector from integer range" in {
    val list = (0 to 9).toNDArray
    assert(DataType.INT == list.dataType())

    val stepped = list(1 -> 7 by 2)
    assert(Array(1, 3, 5).toNDArray == stepped)
    assert(DataType.INT == list.dataType())
  }

  it should "be able to create vector from strings" in {
    val oneString = "testme".toScalar
    assert("testme" == oneString.getString(0))
    assert(DataType.UTF8 == oneString.dataType())

    val someStrings = Array[String]("one", "two", "three").toNDArray
    assert("one" == someStrings.getString(0))
    assert("two" == someStrings.getString(1))
    assert("three" == someStrings.getString(2))
    assert(DataType.UTF8 == someStrings.dataType())
  }
}
