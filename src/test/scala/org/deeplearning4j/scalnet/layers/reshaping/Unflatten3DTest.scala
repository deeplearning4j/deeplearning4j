package org.deeplearning4j.scalnet.layers.reshaping

import org.deeplearning4j.nn.conf.InputPreProcessor
import org.scalatest.FunSpec

/**
  * Created by maxpumperla on 19/07/17.
  */
class Unflatten3DTest extends FunSpec {

  describe("A 3D unflatten layer with output dimensions (10, 20, 30)") {
    val outShape = List(10, 20, 30)
    val unflatten = Unflatten3D(outShape)
    it("should have inputShape List(0)") {
      assert(unflatten.inputShape == List(0))
    }
    it("should have outputShape as provided") {
      assert(unflatten.outputShape == outShape)
    }
    it("should accept a new input shape when provided") {
      val reshapedUnflatten = unflatten.reshapeInput(List(10 * 20 * 30))
      assert(reshapedUnflatten.inputShape == List(10 * 20 * 30))
    }
    it("should not compile if input shape is not set properly") {
      assertThrows[java.lang.IllegalArgumentException] {
        unflatten.compile
      }
    }
    it("should become a DL4J InputPreProcessor when compiled correctly") {
      val reshapedUnflatten = unflatten.reshapeInput(List(10 * 20 * 30))
      val compiledUnflatten = reshapedUnflatten.compile
      assert(compiledUnflatten.isInstanceOf[InputPreProcessor])
    }
  }
}
