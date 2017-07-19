package org.deeplearning4j.scalnet.models

import org.scalatest.{BeforeAndAfter, FunSpec}

/**
  * Created by maxpumperla on 19/07/17.
  */
class NeuralNetTest extends FunSpec with BeforeAndAfter {

  var model: NeuralNet = NeuralNet()

  before {
    model = NeuralNet()
  }

  describe("A NeuralNet network") {

    it("without layers should produce a MatchError when compiled") {
      assertThrows[scala.MatchError] {
        model.compile(null)
      }
    }
  }
}