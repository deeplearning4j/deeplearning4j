package org.nd4s

import org.nd4s.Implicits._
import org.nd4j.linalg.api.complex.IComplexNDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.FlatSpec


class NDArrayExtractionTest extends FlatSpec{
  "org.nd4j.api.Implicits.RichNDArray" should "provides forall checker" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    //check if all elements in nd meet the criteria.
    assert(ndArray > 0)
    assert(ndArray < 10)
    assert(!(ndArray > 5))
  }

  it should "be able to extract a part of 2d matrix in C ordering" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).mkNDArray(NDOrdering.C)

    val extracted = ndArray(1 -> 3, 0 -> 2)

    val expected =
      Array(
        Array(4, 5),
        Array(7, 8)
      ).toNDArray
    assert(extracted == expected)
  }

  it should "be able to extract a part of 2d matrix in C ordering with offset" in {
    val ndArray = (1 to 9).mkNDArray(Array(2, 2), NDOrdering.C, offset = 4)

    val expectedArray = Array(
      Array(5, 6),
      Array(7, 8)
    ).toNDArray
    assert(ndArray == expectedArray)

    val expectedSlice = Array(
      Array(5),
      Array(7)
    ).toNDArray
    assert(ndArray(->, 0) == expectedSlice)
  }

  it should "be able to extract a part of 2d matrix in F ordering with offset" in {
    val ndArray = (1 to 9).mkNDArray(Array(2, 2), NDOrdering.Fortran, offset = 4)

    val expectedArray = Array(
      Array(5, 7),
      Array(6, 8)
    ).toNDArray
    assert(ndArray == expectedArray)

    val expectedSlice = Array(
      Array(5),
      Array(6)
    ).toNDArray
    assert(ndArray(->, 0) == expectedSlice)
  }

  it should "be able to extract a part of vertically long matrix in C ordering" in {
    val ndArray =
      Array(
        Array(1, 2),
        Array(3, 4),
        Array(5, 6),
        Array(7, 8)
      ).mkNDArray(NDOrdering.C)

    assert(ndArray(0 -> 2, ->) ==
      Array(
        Array(1, 2),
        Array(3, 4)
      ).toNDArray)

    assert(ndArray(2 -> 4, ->) ==
      Array(
        Array(5, 6),
        Array(7, 8)
      ).toNDArray)
  }

  it should "be able to extract a part of horizontally long matrix in C ordering" in {
    val ndArray =
      Array(
        Array(1, 2, 3, 4),
        Array(5, 6, 7, 8)
      ).mkNDArray(NDOrdering.C)

    assert(ndArray(->, 0 -> 2) ==
      Array(
        Array(1, 2),
        Array(5, 6)
      ).toNDArray)

    assert(ndArray(->, 2 -> 4) ==
      Array(
        Array(3, 4),
        Array(7, 8)
      ).toNDArray)
  }

  it should "be able to extract a part of 2d matrix in Fortran ordering" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).mkNDArray(NDOrdering.Fortran)

    val extracted = ndArray(1 -> 3, 0 -> 2)

    val expected =
      Array(
        Array(4, 5),
        Array(7, 8)
      ).toNDArray
    assert(extracted == expected)
  }

  it should "be able to extract a part of 3d matrix in C ordering" in {
    val ndArray = (1 to 8).mkNDArray(Array(2, 2, 2),NDOrdering.C)

    val extracted = ndArray(0, ->, ->)
    val lv = extracted.linearView()
    assert(lv.getFloat(0) == 1)
    assert(lv.getFloat(1) == 2)
    assert(lv.getFloat(2) == 3)
    assert(lv.getFloat(3) == 4)
  }

  it should "return original NDArray if indexRange is all in 2d matrix" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray
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

  it should "accept partially ellipsis indices in C ordering" in {
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

  it should "be able to extract submatrix with index range by step in C ordering" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    val extracted = ndArray(0 -> 3 by 2, ->)
    val extractedWithRange = ndArray(0 until 3 by 2, ->)
    val extractedWithInclusiveRange = ndArray(0 to 2 by 2, ->)

    val expected =
      Array(
        Array(1, 2, 3),
        Array(7, 8, 9)
      ).toNDArray

    assert(extracted == expected)
    assert(extractedWithRange == expected)
    assert(extractedWithInclusiveRange == expected)

    /*
     Equivalent with NumPy document examples.
     @see http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#basic-slicing-and-indexing
    */
    val list = (0 to 9).toNDArray
    val step = list(1 -> 7 by 2).linearView()
    assert(step.length() == 3)
    assert(step.getFloat(0) == 1)
    assert(step(0) == 1.toScalar)
    assert(step(0,0) == 1.toScalar)
    assert(step.getFloat(1) == 3)
    assert(step.getFloat(2) == 5)

    val filtered = list(-2 -> 10).linearView()
    assert(filtered.length() == 2)
    assert(filtered.getFloat(0) == 8)
    assert(filtered.getFloat(1) == 9)

    val nStep = list(-3 -> 3 by -1).linearView()
    assert(nStep.length() == 4)
    assert(nStep.getFloat(0) == 7)
    assert(nStep.getFloat(1) == 6)
    assert(nStep.getFloat(2) == 5)
    assert(nStep.getFloat(3) == 4)
  }

  it should "work with ComplexNDArray correctly" in {
    val complexNDArray =
      Array(
        Array(1 + i, 1 + i),
        Array(1 + 3 * i, 1 + 3 * i)
      ).toNDArray

    val result = complexNDArray(0,0)

    assert(result == (1 + i).toScalar)

    val result2 = complexNDArray(->,0)

    assert(result2 == Array(Array(1 + i),Array(1 + 3*i)).toNDArray)
  }

  it should "be able to update value with specified indices" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    ndArray(0 -> 3 by 2, ->) = 0

    assert(ndArray == Array(
      Array(0, 0, 0),
      Array(4, 5, 6),
      Array(0, 0, 0)
    ).toNDArray)
  }

  it should "be able to update INDArray with specified indices" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    ndArray(0 -> 2, 0 -> 2) = Array(Array(0,1),Array(2,3)).toNDArray

    assert(ndArray == Array(
      Array(0, 1, 3),
      Array(2, 3, 6),
      Array(7, 8, 9)
    ).toNDArray)
  }

  "num2Scalar" should "convert number to Scalar INDArray" in {
    assert(1.toScalar == List(1).toNDArray)
    assert(2f.toScalar == List(2).toNDArray)
    assert(3d.toScalar == List(3).toNDArray)
  }
}
