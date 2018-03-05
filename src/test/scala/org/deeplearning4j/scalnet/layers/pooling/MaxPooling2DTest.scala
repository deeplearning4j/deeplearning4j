package org.deeplearning4j.scalnet.layers.pooling

import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.scalatest.FunSpec

/**
  * Created by maxpumperla on 19/07/17.
  */
class MaxPooling2DTest extends FunSpec {

  describe("A 2D max pooling layer with kernel size (5,5)") {
    val kernelSize = List(5, 5)
    val maxPool = MaxPooling2D(kernelSize)
    it("should have inputShape List(0)") {
      assert(maxPool.inputShape == List(0))
    }
    it("should have empty outputShape") {
      assert(maxPool.outputShape == List())
    }
    it("should accept a new input shape when provided") {
      val reshapedPool = maxPool.reshapeInput(List(1, 2, 3))
      assert(reshapedPool.inputShape == List(1, 2, 3))
    }
    it("should become a DL4J pooling layer when compiled") {
      val compiledPool = maxPool.compile
      assert(compiledPool.isInstanceOf[SubsamplingLayer])
    }
  }
}
