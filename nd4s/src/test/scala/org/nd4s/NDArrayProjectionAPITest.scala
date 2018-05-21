package org.nd4s

import org.scalatest.FlatSpec
import org.nd4s.Implicits._

class NDArrayProjectionAPITest extends FlatSpec {
  "ColumnProjectedNDArray" should "map column correctly" in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    val result = for {
      c <- ndArray.columnP
      if c.get(0) % 2 == 0
    } yield c * c

    assert(result == Array(
      Array(4d),
      Array(25d),
      Array(64d)
    ).toNDArray)
  }

  "RowProjectedNDArray" should "map row correctly" in {
    val ndArray =
      Array(
        Array(1d, 2d, 3d),
        Array(4d, 5d, 6d),
        Array(7d, 8d, 9d)
      ).toNDArray

    val result = for {
      c <- ndArray.rowP
      if c.get(0) % 2 == 0
    } yield c * c

    assert(result ==
      Array(Array(16d, 25d, 36d)).toNDArray)
  }

  "SliceProjectedNDArray" should "map slice correctly" in {
    val ndArray =
      (1d to 8d by 1).asNDArray(2,2,2)

    val result = for {
      slice <- ndArray.sliceP
      if slice.get(0) > 1
    } yield slice * slice

    assert(result == List(25d,36d,49d,64d).asNDArray(1,2,2))
  }
}
