package org.nd4s

import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.scalatest.{BeforeAndAfter, FlatSpec}

class ComplexNDArrayInCOrderTest extends ComplexNDArrayTestBase with COrderingForTest
class ComplexNDArrayInFortranOrderTest extends ComplexNDArrayTestBase with FortranOrderingForTest

trait ComplexNDArrayTestBase extends FlatSpec with BeforeAndAfter{self:OrderingForTest =>
  val current = Nd4j.ORDER

  it should "work with ComplexNDArray correctly" ignore {

    val complexNDArray =
      Array(
        Array(1 + i, 1 + i),
        Array(1 + 3 * i, 1 + 3 * i)
      ).toNDArray

    val result = complexNDArray(0,0)

    assert(result == 1 + i)

    val result2 = complexNDArray(->,0)

    assert(result2 == Array(Array(1 + i),Array(1 + 3*i)).toNDArray)

    complexNDArray.forallC(_.realComponent() == 1)
  }


  override protected def before(fun: => Any): Unit = {
    Nd4j.ORDER = ordering.value
  }

  override protected def after(fun: => Any): Unit = {
    Nd4j.ORDER = current
  }
}
