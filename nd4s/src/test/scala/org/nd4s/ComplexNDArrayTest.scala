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

import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.scalatest.{BeforeAndAfter, FlatSpec}

class ComplexNDArrayInCOrderTest extends ComplexNDArrayTestBase with COrderingForTest
class ComplexNDArrayInFortranOrderTest extends ComplexNDArrayTestBase with FortranOrderingForTest

trait ComplexNDArrayTestBase extends FlatSpec with BeforeAndAfter{self:OrderingForTest =>
  val current = Nd4j.ORDER

  it should "work with ComplexNDArray correctly" ignore {

    val complexNDArray =
      Array(
        Array(1 + i, 1 + i),
        Array(1 + 3 * i, 1 + 3 * i)
      ).toNDArray

    val result = complexNDArray(0,0)

    assert(result == 1 + i)

    val result2 = complexNDArray(->,0)

    assert(result2 == Array(Array(1 + i),Array(1 + 3*i)).toNDArray)

    complexNDArray.forallC(_.realComponent() == 1)
  }


  override protected def before(fun: => Any): Unit = {
    Nd4j.ORDER = ordering.value
  }

  override protected def after(fun: => Any): Unit = {
    Nd4j.ORDER = current
  }
}
