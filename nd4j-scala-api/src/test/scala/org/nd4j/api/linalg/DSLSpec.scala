package org.nd4j.api.linalg

import org.junit.runner.RunWith
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, FlatSpec}

import DSL._

@RunWith(classOf[JUnitRunner])
class DSLSpec extends FlatSpec with Matchers {


  "DSL" should "wrap and extend an INDArray" in {

    // This test just verifies that an INDArray gets wrapped with an implicit conversion

    val nd = Nd4j.create(Array[Float](1, 2), Array(2, 1))
    val nd1 = nd + 10L // + creates new array, += modifies in place

    nd(0) should equal(1)
    nd1(0) should equal(11)

    val nd2 = nd += 100
    nd2 should  equal(nd)
    nd2(0) should equal(101)

    nd2 match {
      case i: INDArray =>  // do nothing
      case _ => fail("Expect our object to be an INDArray")
    }

  }

}