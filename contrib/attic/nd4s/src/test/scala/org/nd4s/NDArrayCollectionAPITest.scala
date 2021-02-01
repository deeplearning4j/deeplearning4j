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

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4s.ops.FunctionalOpExecutioner
import org.scalatest.{ FlatSpec, Matchers }

class NDArrayCollectionAPITest extends FlatSpec with Matchers {
  "CollectionLikeNDArray" should "provides filter API" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    val filtered = ndArray.filter(_ > 3)

    assert(
      filtered ==
        Array(
          Array(0, 0, 0),
          Array(4, 5, 6),
          Array(7, 8, 9)
        ).toNDArray
    )
  }

  "CollectionLikeNDArray from Floats" should "provides filter API" in {
    val ndArray =
      Array(
        Array(1f, 2f, 3f),
        Array(4f, 5f, 6f),
        Array(7f, 8f, 9f)
      ).toNDArray

    val filtered = ndArray.filter(_ > 3)

    assert(
      filtered ==
        Array(
          Array(0f, 0f, 0f),
          Array(4f, 5f, 6f),
          Array(7f, 8f, 9f)
        ).toNDArray
    )
  }

  "CollectionLikeNDArray from Long " should "provides filter API" in {
    val ndArray =
      Array(
        Array(1L, 2L, 3L),
        Array(4L, 5L, 6L),
        Array(7L, 8L, 9L)
      ).toNDArray

    val filtered = ndArray.filter(_ > 3)

    assert(
      filtered ==
        Array(
          Array(0L, 0L, 0L),
          Array(4L, 5L, 6L),
          Array(7L, 8L, 9L)
        ).toNDArray
    )
  }

  it should "provides filter bitmask API" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    val filterMasked = ndArray.filterBit(_ % 2 == 0)

    assert(
      filterMasked ==
        Array(
          Array(0, 1, 0),
          Array(1, 0, 1),
          Array(0, 1, 0)
        ).toNDArray
    )
  }
  it should "provides map API" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    val mapped = ndArray.map(_ * 2 + 1)

    assert(
      mapped ==
        Array(
          Array(3, 5, 7),
          Array(9, 11, 13),
          Array(15, 17, 19)
        ).toNDArray
    )
  }

  it should "provides forall checker" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    //check if all elements in nd meet the criteria.
    assert(ndArray > 0)
    assert(ndArray.forall(_ > 0))
    "ndArray.forallC(_.absoluteValue().doubleValue() > 0)" shouldNot typeCheck
    assert(ndArray < 10)
    assert(!(ndArray >= 5))
  }

  it should "provides exist API" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    //check if any element in nd meet the criteria.
    assert(ndArray.exists(_ > 8))
  }

  it should "provides existTyped API" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    //check if any element in nd meet the criteria.
    assert(ndArray.existsTyped[Int](_ > 8)(IntNDArrayEvidence))
  }

  "CollectionLikeNDArray" should "provides forAll API" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    val resultFalse = ndArray.forall(_ > 3)
    assert(false == resultFalse)

    val resultTrue = ndArray.forall(_ < 10)
    assert(true == resultTrue)
  }

  "CollectionLikeNDArray" should "provides forAllTyped API" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    val results = ndArray.forallTyped[Int](_ > 3)(IntNDArrayEvidence)
    assert(false == results)
  }

  "FunctionalOpExecutioner" should "allow debug and verbose" in {
    val executioner = new FunctionalOpExecutioner
    executioner.enableDebugMode(true)
    executioner.enableVerboseMode(true)

    assert(executioner.isDebug)
    assert(executioner.isVerbose)
  }

  "FunctionalOpExecutioner" should "provide access to environment information" in {
    FunctionalOpExecutioner.apply.printEnvironmentInformation()
    val environment = FunctionalOpExecutioner.apply.getEnvironmentInformation
    assert(environment != null)
  }
}
