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
import org.scalatest.FlatSpec


class NDArrayExtractionInCOrderingTest extends NDArrayExtractionTestBase with COrderingForTest
class NDArrayExtractionInFortranOrderingTest extends NDArrayExtractionTestBase with FortranOrderingForTest

trait NDArrayExtractionTestBase extends FlatSpec{self:OrderingForTest =>

  "org.nd4j.api.Implicits.RichNDArray" should "be able to extract a value in specified indices" in {
    val ndArray = Array(
      Array(1, 2),
      Array(3, 4)
    ).mkNDArray(ordering)
  }

  it should "be able to extract a part of 2d matrix" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).mkNDArray(ordering)

    val extracted = ndArray(1 -> 3, 0 -> 2)

    val expected =
      Array(
        Array(4, 5),
        Array(7, 8)
      ).mkNDArray(ordering)
    assert(extracted == expected)
  }

  it should "be able to extract a part of 2d matrix with offset" in {
    val ndArray = (1 to 9).mkNDArray(Array(2, 2), NDOrdering.C, offset = 4)

    val expectedArray = Array(
      Array(5, 6),
      Array(7, 8)
    ).mkNDArray(ordering)
    assert(ndArray == expectedArray)

    val expectedSlice = Array(
      Array(5),
      Array(7)
    ).toNDArray
    assert(ndArray(->, 0) == expectedSlice)
  }

  it should "be able to extract a part of vertically long matrix in" in {
    val ndArray =
      Array(
        Array(1, 2),
        Array(3, 4),
        Array(5, 6),
        Array(7, 8)
      ).mkNDArray(ordering)

    assert(ndArray(0 -> 2, ->) ==
      Array(
        Array(1, 2),
        Array(3, 4)
      ).mkNDArray(ordering))

    assert(ndArray(2 -> 4, ->) ==
      Array(
        Array(5, 6),
        Array(7, 8)
      ).mkNDArray(ordering))
  }

  it should "be able to extract a part of horizontally long matrix" in {
    val ndArray =
      Array(
        Array(1, 2, 3, 4),
        Array(5, 6, 7, 8)
      ).mkNDArray(ordering)

    assert(ndArray(->, 0 -> 2) ==
      Array(
        Array(1, 2),
        Array(5, 6)
      ).mkNDArray(ordering))

    assert(ndArray(->, 2 -> 4) ==
      Array(
        Array(3, 4),
        Array(7, 8)
      ).mkNDArray(ordering))
  }

  it should "be able to extract a part of 3d matrix" in {
    val ndArray = (1 to 8).mkNDArray(Array(2, 2, 2),ordering)

    val extracted = ndArray(0, ->, ->)
    val expected = ndArray.slice(0)
    assert(extracted == expected)
  }

  it should "return original NDArray if indexRange is all in 2d matrix" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).mkNDArray(ordering)
    val extracted = ndArray(->, ->)
    assert(ndArray == extracted)

    val ellipsised = ndArray(--->)
    assert(ellipsised == ndArray)
  }

  it should "return original NDArray if indexRange is all in 3d matrix" in {
    val ndArray = (1f to 8f by 1).mkNDArray(Array(2, 2, 2),ordering)
    val extracted = ndArray(->, ->, ->)
    assert(ndArray == extracted)

    val ellipsised = ndArray(--->)
    assert(ellipsised == ndArray)
  }

  it should "accept partially ellipsis indices" in {
    val ndArray = (1f to 8f by 1).mkNDArray(Array(2, 2, 2),ordering)

    val ellipsised = ndArray(--->, 0)
    val notEllipsised = ndArray(->, ->, 0)
    assert(ellipsised == notEllipsised)

    val ellipsisedAtEnd = ndArray(0, --->)
    val notEllipsisedAtEnd = ndArray(0, ->, ->)
    assert(ellipsisedAtEnd == notEllipsisedAtEnd)

    val ellipsisedOneHand = ndArray(0 ->, ->, ->)
    val notEllipsisedOneHand = ndArray(->, ->, ->)
    assert(ellipsisedOneHand == notEllipsisedOneHand)
  }

  it should "be able to extract submatrix with index range by step" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).mkNDArray(ordering)

    val extracted = ndArray(0 -> 3 by 2, ->)
    val extractedWithRange = ndArray(0 until 3 by 2, ->)
    val extractedWithInclusiveRange = ndArray(0 to 2 by 2, ->)

    val expected =
      Array(
        Array(1, 2, 3),
        Array(7, 8, 9)
      ).mkNDArray(ordering)

    assert(extracted == expected)
    assert(extractedWithRange == expected)
    assert(extractedWithInclusiveRange == expected)

    /*
     Equivalent with NumPy document examples.
     @see http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#basic-slicing-and-indexing
    */
    val list = (0 to 9).toNDArray
    val step = list(1 -> 7 by 2).linearView()
    assert(step.length() == 3)
    assert(step.getFloat(0) == 1)
    assert(step(0) == 1)
    assert(step(0,0) == 1)
    assert(step.getFloat(1) == 3)
    assert(step.getFloat(2) == 5)

    val filtered = list(-2 -> 10).linearView()
    assert(filtered.length() == 2)
    assert(filtered.getFloat(0) == 8)
    assert(filtered.getFloat(1) == 9)

    val nStep = list(-3 -> 3 by -1).linearView()
    assert(nStep.length() == 4)
    assert(nStep.getFloat(0) == 7)
    assert(nStep.getFloat(1) == 6)
    assert(nStep.getFloat(2) == 5)
    assert(nStep.getFloat(3) == 4)
  }

  it should "be able to update value with specified indices" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).mkNDArray(ordering)

    ndArray(0 -> 3 by 2, ->) = 0

    assert(ndArray == Array(
      Array(0, 0, 0),
      Array(4, 5, 6),
      Array(0, 0, 0)
    ).mkNDArray(ordering))
  }

  it should "be able to update INDArray with specified indices" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).mkNDArray(ordering)

    ndArray(0 -> 2, 0 -> 2) = Array(Array(0,1),Array(2,3)).mkNDArray(ordering)

    assert(ndArray == Array(
      Array(0, 1, 3),
      Array(2, 3, 6),
      Array(7, 8, 9)
    ).mkNDArray(ordering))
  }

  "num2Scalar" should "convert number to Scalar INDArray" in {
    assert(1.toScalar == List(1).toNDArray)
    assert(2f.toScalar == List(2).toNDArray)
    assert(3d.toScalar == List(3).toNDArray)
  }
}
