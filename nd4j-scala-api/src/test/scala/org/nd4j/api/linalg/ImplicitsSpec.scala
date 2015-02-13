package org.nd4j.api.linalg

import org.junit.runner.RunWith
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, FlatSpec}

import Implicits._

@RunWith(classOf[JUnitRunner])
class ImplicitsSpec extends FlatSpec with Matchers {


  "Implicits" should "extend an INDArray" in {

    // This test just verifies that an INDArray gets wrapped with an implicit conversion

    val nd = Nd4j.create(Array[Float](1, 2), Array(2, 1))
    val nd1 = nd + 10L // nd1 and nd refer to same array

    nd1 should equal(nd)
    nd1(0) should equal(11)

  }

}