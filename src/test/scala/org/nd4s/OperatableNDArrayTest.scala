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
import org.nd4s.Implicits._
import org.nd4j.linalg.api.complex.{IComplexNDArray, IComplexNumber}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class OperatableNDArrayTest extends FlatSpec with Matchers {
  "RichNDArray" should "use the apply method to access values" in {
    // -- 2D array
    val nd2 = Nd4j.create(Array[Double](1, 2, 3, 4), Array(4, 1))

    nd2.get(0) should be(1)
    nd2.get(3, 0) should be(4)

    // -- 3D array
    val nd3 = Nd4j.create(Array[Double](1, 2, 3, 4, 5, 6, 7, 8), Array(2, 2, 2))
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
    val a = Nd4j.create(Array[Double](1, 3))
    val b = -a
    b.get(0) should be(-1)
    b.get(1) should be(-3)
  }

  it should "worked with ComplexNDArray correctly" in {
    val complexNDArray =
      Array(
        Array(1 + i, 1 + i),
        Array(1 + 3 * i, 1 + 3 * i)
      ).toNDArray

    val result = complexNDArray + 2
    result shouldBe a[IComplexNDArray]

    result shouldBe Nd4j.createComplex(
      Array(
        Array(Nd4j.createComplexNumber(3, 1), Nd4j.createComplexNumber(3, 1)),
        Array(Nd4j.createComplexNumber(3, 3), Nd4j.createComplexNumber(3, 3)))
    )
  }

  it should "not prevent any2stringadd syntax" in {
    val s:String = Nd4j.create(2,2) + ""
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

    val complexNDArray = Nd4j.createComplex(ndArray)

    //return ComplexNumber in ComplexNDArray in IComplexNumber
    complexNDArray.get(0) shouldBe a[IComplexNumber]
    val sumComplexValue = complexNDArray.sumT
    sumComplexValue shouldBe a[IComplexNumber]

    //switch return value with passing corresponding evidence explicitly
    val sumValueInFloatExplicit = ndArray.sumT(FloatNDArrayEvidence)
    sumValueInFloatExplicit shouldBe a[java.lang.Float]

    //switch return value with declaring implicit value but explicit one would be more readable.
    import org.nd4s.Evidences.float
    val sumValueInFloatImplicit = ndArray.sumT
    sumValueInFloatImplicit shouldBe a[java.lang.Float]
  }
}