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
      Array(1d, 4d, 3d),
      Array(4d, 25d, 6d),
      Array(7d, 64d, 9d)
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

    assert(result == Array(
      Array(1d, 2d, 3d),
      Array(16d, 25d, 36d),
      Array(7d, 8d, 9d)
    ).toNDArray)
  }

  "SliceProjectedNDArray" should "map slice correctly" in {
    val ndArray =
      (1d to 8d by 1).asNDArray(2,2,2)

    val result = for {
      slice <- ndArray.sliceP
      r <- slice.rowP
      if r.get(0) > 1
    } yield r * r

    assert(result == List(1d,2d,9d,16d,25d,36d,49d,64d).asNDArray(2,2,2))
  }
}
