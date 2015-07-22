package org.nd4j.api

import org.scalatest.FlatSpec

import org.nd4j.api.Implicits._
import org.nd4j.linalg.factory.Nd4j

class NDArrayExtractionTest extends FlatSpec {
  "org.nd4j.api.Implicits.RichNDArray" should "provides forall checker" in {
    val ndArray = Nd4j.create(Array(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f), Array(3, 3))

    //check if all elements in nd meet the criteria.
    assert(ndArray > 0)
    assert(ndArray < 10)
    assert(!(ndArray > 5))
  }

  it should "be able to extract a part of 2d matrix in C ordering" in {
    Nd4j.factory().setOrder(NDOrdering.C.value)
    val ndArray = (1 to 9).asNDArray(3, 3)

    val extracted = ndArray(1 -> 3, 0 -> 2)
    assert(extracted.rows() == 2)
    assert(extracted.columns() == 2)

    val lv = extracted.linearView()
    assert(lv.getFloat(0) == 2)
    assert(lv.getFloat(1) == 3)
    assert(lv.getFloat(2) == 5)
    assert(lv.getFloat(3) == 6)
  }

  it should "be able to extract a part of 2d matrix in Fortran ordering" in {
    Nd4j.factory().setOrder(NDOrdering.Fortran.value)
    val ndArray = (1 to 9).asNDArray(3, 3)

    val extracted = ndArray(1 -> 3, 0 -> 2)
    assert(extracted.rows() == 2)
    assert(extracted.columns() == 2)

    val lv = extracted.linearView()
    assert(lv.getFloat(0) == 2)
    assert(lv.getFloat(1) == 5)
    assert(lv.getFloat(2) == 3)
    assert(lv.getFloat(3) == 6)
  }

  it should "be able to extract a part of 3d matrix in C ordering" in {
    Nd4j.factory().setOrder(NDOrdering.C.value)
    val ndArray = (1 to 8).asNDArray(2, 2, 2)

    val extracted = ndArray(0, 0 -> 2, ->)
    val lv = extracted.linearView()
    assert(lv.getFloat(0) == 1)
    assert(lv.getFloat(1) == 3)
    assert(lv.getFloat(2) == 5)
    assert(lv.getFloat(3) == 7)
  }

  it should "be able to extract a part of 3d matrix in Fortran ordering" in {
    Nd4j.factory().setOrder(NDOrdering.Fortran.value)
    val ndArray = (1 to 8).asNDArray(2, 2, 2)

    val extracted = ndArray(0, 0 -> 2, ->)
    val lv = extracted.linearView()
    assert(lv.getFloat(0) == 1)
    assert(lv.getFloat(1) == 5)
    assert(lv.getFloat(2) == 3)
    assert(lv.getFloat(3) == 7)
  }

  it should "return original NDArray if indexRange is all in 2d matrix in C ordering" in {
    Nd4j.factory().setOrder(NDOrdering.C.value)
    val ndArray = (1f to 9f by 1).asNDArray(3, 3)
    val extracted = ndArray(->, ->)
    assert(ndArray == extracted)

    val ellipsised = ndArray(--->)
    assert(ellipsised == ndArray)
  }

  it should "return original NDArray if indexRange is all in 3d matrix in C ordering" in {
    Nd4j.factory().setOrder(NDOrdering.C.value)
    val ndArray = (1f to 8f by 1).asNDArray(2, 2, 2)
    val extracted = ndArray(->, ->, ->)
    assert(ndArray == extracted)

    val ellipsised = ndArray(--->)
    assert(ellipsised == ndArray)
  }

  it should "accept partially ellipsis indices in C ordering" in {
    Nd4j.factory().setOrder(NDOrdering.C.value)
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
    Nd4j.factory().setOrder(NDOrdering.C.value)
    val ndArray = (1f to 9f by 1).asNDArray(3, 3)

    val extracted = ndArray(0 -> 3 by 2, ->)
    val extractedWithRange = ndArray(0 until 3 by 2, ->)
    val extractedWithInclusiveRange = ndArray(0 to 2 by 2, ->)

    val lv = extracted.linearView()
    assert(extracted == extractedWithRange)
    assert(extracted == extractedWithInclusiveRange)
    assert(lv.getFloat(0) == 1)
    assert(lv.getFloat(1) == 3)
    assert(lv.getFloat(2) == 4)
    assert(lv.getFloat(3) == 6)
    assert(lv.getFloat(4) == 7)
    assert(lv.getFloat(5) == 9)

    /*
     Equivalent with NumPy document examples.
     @see http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#basic-slicing-and-indexing
    */
    val list = (0 to 9).toNDArray
    val step = list(1 -> 7 by 2).linearView()
    assert(step.length() == 3)
    assert(step.getFloat(0)== 1)
    assert(step.getFloat(1)== 3)
    assert(step.getFloat(2)== 5)

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

  "num2Scalar" should "convert number to Scalar INDArray" in {
    assert(1.toScalar == List(1).toNDArray)
    assert(2f.toScalar == List(2).toNDArray)
    assert(3d.toScalar == List(3).toNDArray)
  }
}
