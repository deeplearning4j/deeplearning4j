package org.deeplearning4j.scalnet.layers.reshaping

import org.deeplearning4j.nn.conf.InputPreProcessor
import org.scalatest.FunSpec

/**
  * Created by maxpumperla on 19/07/17.
  */
class ReshapeTest extends FunSpec {

  describe("A reshape layer with in and out shapes") {
    val inShape = List(20, 10)
    val outShape = List(10, 20)
    val reshape = Reshape(outShape, inShape)
    it("should have inputShape as provided") {
      assert(reshape.inputShape == inShape)
    }
    it("should have outputShape as provided") {
      assert(reshape.outputShape == outShape)
    }
    it("should accept a new input shape when provided") {
      val reshaped = reshape.reshapeInput(List(10, 2, 10))
      assert(reshaped.inputShape == List(10, 2, 10))
    }
    it("should become a DL4J InputPreProcessor when compiled") {
      val compiledReshape = reshape.compile
      assert(compiledReshape.isInstanceOf[InputPreProcessor])
    }
  }
}
