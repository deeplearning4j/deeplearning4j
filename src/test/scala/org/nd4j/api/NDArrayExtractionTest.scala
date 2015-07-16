package org.nd4j.api

import org.scalatest.FlatSpec

import org.nd4j.api.Implicits._
import org.nd4j.linalg.factory.Nd4j

class RichNDArrayTest extends FlatSpec {
  "org.nd4j.api.Implicits.RichNDArray" should "provides forall checker" in {
    val ndArray = Nd4j.create(Array(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f), Array(3, 3))

    //check if all elements in nd meet the criteria.
    assert(ndArray > 0)
    assert(ndArray < 10)
    assert(!(ndArray > 5))
  }

  it should "be able to extract a part of 2d matrix" in {
    val ndArray = List(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f).asNDArray(3, 3)

    val extracted = ndArray(1 -> 2, 0 -> 1)
    assert(extracted.rows() == 2)
    assert(extracted.columns() == 2)
    assert(extracted.getFloat(0) == 2)
    assert(extracted.getFloat(1) == 3)
    assert(extracted.getFloat(2) == 5)
    assert(extracted.getFloat(3) == 6)
  }

  it should "be able to extract a part of 3d matrix" in {
    val ndArray = (1f to 8f by 1).asNDArray(2, 2, 2)

    val extracted = ndArray(0, 0 -> 1, ->)
    assert(extracted.getFloat(0) == 1)
    assert(extracted.getFloat(1) == 3)
    assert(extracted.getFloat(2) == 5)
    assert(extracted.getFloat(3) == 7)
  }

  it should "return original NDArray if indexRange is all in 2d matrix" in {
    val ndArray = (1f to 9f by 1).asNDArray(3, 3)
    val extracted = ndArray(->, ->)
    assert(ndArray == extracted)

    val ellipsised = ndArray(--->)
    assert(ellipsised == ndArray)
  }

  it should "return original NDArray if indexRange is all in 3d matrix" in {
    val ndArray = (1f to 8f by 1).asNDArray(2, 2, 2)
    val extracted = ndArray(->, ->, ->)
    assert(ndArray == extracted)

    val ellipsised = ndArray(--->)
    assert(ellipsised == ndArray)
  }

  it should "accept partially ellipsis indices" in {
    val ndArray = (1f to 8f by 1).asNDArray(2, 2, 2)

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

  it should "be able to extract submatrix with index range by step" in{
    val ndArray = (1f to 9f by 1).asNDArray(3,3)

    val extracted = ndArray(0->3 by 2,->)
    val extractedWithRange = ndArray(0 to 3 by 2,->)

    assert(extracted == extractedWithRange)
    assert(extracted.getFloat(0) == 1)
    assert(extracted.getFloat(1) == 3)
    assert(extracted.getFloat(2) == 4)
    assert(extracted.getFloat(3) == 6)
    assert(extracted.getFloat(4) == 7)
    assert(extracted.getFloat(5) == 9)
  }
}
