package org.nd4j.api.linalg

import org.junit.runner.RunWith
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, FlatSpec}

import Implicits._

@RunWith(classOf[JUnitRunner])
class INDArrayExtSpec extends FlatSpec with Matchers {

  "INDArrayEx" should "use the apply method to access values" in {

    // -- 2D array
    val nd2 = Nd4j.create(Array[Double](1, 2, 3, 4), Array(4, 1))

    nd2(0) should equal(1)
    nd2(3, 0) should equal(4)

    // -- 3D array
    val nd3 = Nd4j.create(Array[Double](1, 2, 3, 4, 5, 6, 7, 8), Array(2, 2, 2))
    nd3(0, 0, 0) should equal(1)
    nd3(1, 1, 1) should equal(8)

  }

  it should "use transpose abbreviation" in {
    val nd1 = Nd4j.create(Array[Double](1, 2, 3), Array(3, 1))
    nd1.shape should equal(Array(3, 1))
    val nd1t = nd1.t
    nd1t.shape should equal(Array(3))
  }

  it should "add correctly" in {
    val nd3 = Nd4j.create(Array[Double](1, 2, 3, 4, 5, 6, 7, 8), Array(2, 2, 2))
    nd3 + 1
    nd3(0, 0, 0) should equal(2)
  }

  it should "subtract correctly" in {
    val nd3 = Nd4j.create(Array[Double](1, 2, 3, 4, 5, 6, 7, 8), Array(2, 2, 2))
    nd3 - 1
    nd3(0, 0, 0) should equal(0)
  }

  it should "divide correctly" in {
    val nd2 = Nd4j.create(Array[Double](1, 2, 3, 4), Array(4, 1))
    nd2 / nd2
    nd2(3) should equal(1)
  }

  it should "multiply correctly" in {
    val nd2 = Nd4j.create(Array[Double](1, 2, 3, 4), Array(4, 1))
    nd2 x nd2
    nd2(3) should equal(16) // [1.0 ,4.0 ,9.0 ,16.0]
    nd2 x 4 // [4.0 ,16.0 ,36.0 ,64.0]
    nd2(0) should equal(4)
  }

  it should "use the update method to mutate values" in {
    val nd3 = Nd4j.create(Array[Double](1, 2, 3, 4, 5, 6, 7, 8), Array(2, 2, 2))
    nd3(0) = 11
    nd3(0) should equal(11)

    val idx = Array(1, 1, 1)
    nd3(idx) = 100
    nd3(idx) should equal(100)

  }

}