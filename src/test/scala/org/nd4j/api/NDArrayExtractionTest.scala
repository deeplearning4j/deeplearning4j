package org.nd4j.api

import org.scalatest.FlatSpec

import org.nd4j.api.Implicits._
import org.nd4j.linalg.factory.Nd4j

class RichNDArrayTest extends FlatSpec {
  "org.nd4j.api.Implicits.RichNDArray" should "provides forall checker" in {
    val nd = Nd4j.create(Array(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f), Array(3, 3))

    //check if all elements in nd meet the criteria.
    assert(nd > 0)
    assert(nd < 10)
    assert(!(nd > 5))
  }

  it should "be able to extract a part of 2d matrix" in {
    val nd = List(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f).asNDArray(3, 3)

    val extracted = nd(1->2, 0->1)
    assert(extracted.rows() == 2)
    assert(extracted.columns() == 2)
    assert(extracted.getFloat(0) == 2)
    assert(extracted.getFloat(1) == 3)
    assert(extracted.getFloat(2) == 5)
    assert(extracted.getFloat(3) == 6)
  }

  it should "be able to extract a part of 3d matrix" in {
    val nd = (1f to 8f by 1).asNDArray(2, 2, 2)

    val extracted = nd(0, 0->1, ->)
    assert(extracted.getFloat(0) == 1)
    assert(extracted.getFloat(1) == 3)
    assert(extracted.getFloat(2) == 5)
    assert(extracted.getFloat(3) == 7)
  }
}
