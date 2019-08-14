package org.nd4s.samediff

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.scalatest.{ FlatSpec, Matchers }

class ConstructionTest extends FlatSpec {

  "SameDiff" should "create simple arithmetic operations " in {
    val graph = new SameDiff[INDArray](_ + _)

    graph.bind(1.0.toScalar)
    graph.bind(2.0.toScalar)

    val out = graph.exec

    assert(out.length() == 1)
    assert(out.getDouble(0:Long,0:Long) == 3.0)
  }
}