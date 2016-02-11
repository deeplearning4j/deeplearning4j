package org.nd4s

import org.nd4s.Implicits._
import org.scalatest.FlatSpec

class NDArrayCollectionAPITest extends FlatSpec {
  "CollectionLikeNDArray" should "provides filter API" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    val filtered = ndArray.filter(_ > 3)

    assert(filtered ==
      Array(
        Array(0, 0, 0),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray)
  }
  it should "provides filter bitmask API" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    val filterMasked = ndArray.filterBit(_ % 2 == 0)

    assert(filterMasked ==
      Array(
        Array(0, 1, 0),
        Array(1, 0, 1),
        Array(0, 1, 0)
      ).toNDArray)
  }
  it should "provides map API" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    val mapped = ndArray.map(_ * 2 + 1)

    assert(mapped ==
      Array(
        Array(3, 5, 7),
        Array(9, 11, 13),
        Array(15, 17, 19)
      ).toNDArray)
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
    assert(ndArray.exist(_ > 8))
  }
}
