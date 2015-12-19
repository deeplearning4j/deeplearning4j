/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4s

import org.junit.runner.RunWith
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}
import org.nd4s.Implicits._

@RunWith(classOf[JUnitRunner])
class DSLSpec extends FlatSpec with Matchers {

  "DSL" should "wrap and extend an INDArray" in {

    // This test just verifies that an INDArray gets wrapped with an implicit conversion

    val nd = Nd4j.create(Array[Float](1, 2), Array(2, 1))
    val nd1 = nd + 10L // + creates new array, += modifies in place

    nd.get(0) should equal(1)
    nd1.get(0) should equal(11)

    val nd2 = nd += 100
    nd2 should equal(nd)
    nd2.get(0) should equal(101)

    // Verify that we are working with regular old INDArray objects
    nd2 match {
      case i: INDArray => // do nothing
      case _ => fail("Expect our object to be an INDArray")
    }

  }

  "DSL" should "not prevent Map[Int,T] creation" in {
    Map(0->"hello") shouldBe a [Map[_,_]]
  }
}