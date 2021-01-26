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

import org.junit.runner.RunWith
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.junit.JUnitRunner
import org.scalatest.{ FlatSpec, Matchers }

@RunWith(classOf[JUnitRunner])
class OperatableNDArrayTest extends FlatSpec with Matchers {
  "RichNDArray" should "use the apply method to access values" in {
    // -- 2D array
    val nd2 = Nd4j.create(Array[Double](1, 2, 3, 4), Array[Int](1, 4))

    nd2.get(0) should be(1)
    nd2.get(0, 3) should be(4)

    // -- 3D array
    val nd3 = Nd4j.create(Array[Double](1, 2, 3, 4, 5, 6, 7, 8), Array[Int](2, 2, 2))
    nd3.get(0, 0, 0) should be(1)
    nd3.get(1, 1, 1) should be(8)

  }

  it should "use transpose abbreviation" in {
    val nd1 = Nd4j.create(Array[Double](1, 2, 3), Array(3, 1))
    nd1.shape should equal(Array(3, 1))
    val nd1t = nd1.T
    nd1t.shape should equal(Array(1, 3))
  }

  it should "add correctly" in {
    val a = Nd4j.create(Array[Double](1, 2, 3, 4, 5, 6, 7, 8), Array(2, 2, 2))
    val b = a + 100
    a.get(0, 0, 0) should be(1)
    b.get(0, 0, 0) should be(101)
    a += 1
    a.get(0, 0, 0) should be(2)
  }

  it should "subtract correctly" in {
    val a = Nd4j.create(Array[Double](1, 2, 3, 4, 5, 6, 7, 8), Array(2, 2, 2))
    val b = a - 100
    a.get(0, 0, 0) should be(1)
    b.get(0, 0, 0) should be(-99)
    a -= 1
    a.get(0, 0, 0) should be(0)

    val c = Nd4j.create(Array[Double](1, 2))
    val d = c - c
    d.get(0) should be(0)
    d.get(1) should be(0)
  }

  it should "divide correctly" in {
    val a = Nd4j.create(Array[Double](1, 2, 3, 4, 5, 6, 7, 8), Array(2, 2, 2))
    val b = a / a
    a.get(1, 1, 1) should be(8)
    b.get(1, 1, 1) should be(1)
    a /= a
    a.get(1, 1, 1) should be(1)
  }

  it should "element-by-element multiply correctly" in {
    val a = Nd4j.create(Array[Double](1, 2, 3, 4), Array(4, 1))
    val b = a * a
    a.get(3) should be(4) // [1.0, 2.0, 3.0, 4.0
    b.get(3) should be(16) // [1.0 ,4.0 ,9.0 ,16.0]
    a *= 5 // [5.0 ,10.0 ,15.0 ,20.0]
    a.get(0) should be(5)
  }

  it should "use the update method to mutate values" in {
    val nd3 = Nd4j.create(Array[Double](1, 2, 3, 4, 5, 6, 7, 8), Array(2, 2, 2))
    nd3(0) = 11
    nd3.get(0) should be(11)

    val idx = Array(1, 1, 1)
    nd3(idx) = 100
    nd3.get(idx) should be(100)
  }

  it should "use === for equality comparisons" in {
    val a = Nd4j.create(Array[Double](1, 2))

    val b = Nd4j.create(Array[Double](1, 2))
    val c = a === b
    c.get(0) should be(1)
    c.get(1) should be(1)

    val d = Nd4j.create(Array[Double](10, 20))
    val e = a === d
    e.get(0) should be(0)
    e.get(1) should be(0)

    val f = a === 1 // === from our DSL
    f.get(0) should be(1)
    f.get(1) should be(0)
  }

  it should "use - prefix for negation" in {
    val a = Nd4j.create(Array[Float](1, 3))
    val b = -a
    b.get(0) should be(-1)
    b.get(1) should be(-3)
  }

  it should "not prevent any2stringadd syntax" in {
    val s: String = Nd4j.create(2, 2) + ""
  }

  "Sum function" should "choose return value depending on INDArray type" in {
    val ndArray =
      Array(
        Array(1, 2),
        Array(4, 5)
      ).toNDArray

    //return Double in real NDArray at default
    ndArray.get(0) shouldBe a[java.lang.Double]
    val sumValue = ndArray.sumT
    sumValue shouldBe a[java.lang.Double]

    //switch return value with passing corresponding evidence explicitly
    val sumValueInFloatExplicit = ndArray.sumT(FloatNDArrayEvidence)
    sumValueInFloatExplicit shouldBe a[java.lang.Float]

    //switch return value with declaring implicit value but explicit one would be more readable.
    import org.nd4s.Evidences.float
    val sumValueInFloatImplicit = ndArray.sumT
    sumValueInFloatImplicit shouldBe a[java.lang.Float]
  }

  it should "provide matrix multiplicaton operations " in {
    val a = Nd4j.create(Array[Float](4, 6, 5, 7)).reshape(2, 2)
    val b = Nd4j.create(Array[Float](1, 3, 4, 8)).reshape(2, 2)
    a **= b
    val expected = Array[Float](28.0000f, 60.0000f, 33.0000f, 71.0000f).toNDArray.reshape(2, 2)
    a shouldBe expected
  }

  it should "provide matrix division operations " in {
    val a = Nd4j.create(Array[Float](4, 6, 5, 7)).reshape(2, 2)
    a /= 12
    a.get(0) shouldBe (0.3333 +- 0.0001)
    a.get(1) shouldBe (0.5 +- 0.0001)
    a.get(2) shouldBe (0.4167 +- 0.0001)
    a.get(3) shouldBe (0.5833 +- 0.0001)

    val b = Nd4j.create(Array[Float](4, 6, 5, 7)).reshape(2, 2)
    b %= 12
    b.get(0) shouldBe (4.0)
    b.get(1) shouldBe (6.0)
    b.get(2) shouldBe (5.0)
    b.get(3) shouldBe (-5.0)

    val c = Nd4j.create(Array[Float](4, 6, 5, 7)).reshape(2, 2)
    c \= 12
    c.get(0) shouldBe (3.0)
    c.get(1) shouldBe (2.0)
    c.get(2) shouldBe (2.4000 +- 0.0001)
    c.get(3) shouldBe (1.7143 +- 0.0001)
  }

  it should "provide math operations for vectors " in {
    val a = Nd4j.create(Array[Float](4, 6))
    val b = Nd4j.create(Array[Float](1, 3))
    a /= b
    val expected1 = Nd4j.create(Array[Float](4, 2))
    assert(a == expected1)

    a *= b
    val expected2 = Nd4j.create(Array[Float](4, 6))
    assert(a == expected2)

    a += b
    val expected3 = Nd4j.create(Array[Float](5, 9))
    assert(a == expected3)

    a -= b
    val expected4 = Nd4j.create(Array[Float](4, 6))
    assert(a == expected4)

    a \= b
    val expected5 = Array[Float](0.25f, 0.5f).toNDArray
    assert(a == expected5)

    val c = a * b
    val expected6 = Array[Float](0.25f, 1.5f).toNDArray
    assert(c == expected6)

    val d = a + b
    val expected7 = Array[Float](1.25f, 3.5f).toNDArray
    assert(d == expected7)

    val e = a / b
    e.get(0) should be(0.2500 +- 0.0001)
    e.get(1) should be(0.1667 +- 0.0001)

    val f = a \ b
    f.get(0) should be(4.0 +- 0.0001)
    f.get(1) should be(6.0 +- 0.0001)

    val g = a ** b
    g.get(0) shouldBe 1.7500

    val h = a dot b
    g.get(0) shouldBe 1.7500

    d.sumT shouldBe 4.75

    d.meanT shouldBe 2.375

    d.norm1T shouldBe 4.75

    d.maxT shouldBe 3.5

    d.minT shouldBe 1.25

    d.prodT shouldBe 4.375

    d.varT shouldBe 2.53125

    d.norm2T should be(3.7165 +- 0.0001)

    d.stdT should be(1.5909 +- 0.0001)
  }

  it should "provide arithmetic ops calls on integers " in {
    val ndArray = Array(1, 2).toNDArray
    val c = ndArray + 5
    c shouldBe Array(6, 7).toNDArray

    val d = 5 + ndArray
    c shouldBe Array(6, 7).toNDArray
  }

  it should "broadcast add ops calls on vectors with different length " in {
    val x = Array(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f).mkNDArray(Array(3, 5))
    val y = Array[Float](1f, 1f, 1f, 1f, 1f).toNDArray
    val e = x + 1f.toScalar
    assert((x + y) == e)

    val x1 = Array(1f, 1f, 1f, 1f, 1f, 1f).mkNDArray(Array(3, 1, 2))
    val y1 = Array[Float](1f, 1f, 1f, 1f).toNDArray.reshape(2, 2)
    val t1 = Array(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f).mkNDArray(Array(3, 2, 2))
    val e1 = t1 + 1f
    assert((x1 + y1) == e1)

    val e2 = 1f + t1
    assert(e1 == e2)
  }

  it should "broadcast multiplication ops " in {

    val x1 = Array(1f, 1f, 1f, 1f, 1f, 1f).mkNDArray(Array(3, 1, 2))
    val y1 = Array[Float](1f, 1f, 1f, 1f).toNDArray.reshape(2, 2)
    val t1 = Array(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f).mkNDArray(Array(3, 2, 2))
    val e1 = t1 * 1f
    assert((x1 * y1) == e1)

    val e2 = 1f * t1
    assert(e1 == e2)
  }
}
