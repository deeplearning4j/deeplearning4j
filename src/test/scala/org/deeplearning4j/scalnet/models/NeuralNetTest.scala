package org.deeplearning4j.scalnet.models

import org.deeplearning4j.scalnet.layers.{ Dense, OutputLayer }
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.scalatest.{ BeforeAndAfter, FunSpec }

/**
  * Created by maxpumperla on 19/07/17.
  */
class NeuralNetTest extends FunSpec with BeforeAndAfter {

  var model: NeuralNet = NeuralNet()
  val shape = 100

  before {
    model = NeuralNet()
  }

  describe("A NeuralNet network") {

    it("without layers should produce an IllegalArgumentException when compiled") {
      assertThrows[java.lang.IllegalArgumentException] {
        model.compile(null)
      }
    }
    it("without buildOutput called should not have an output layer") {
      model.add(Dense(shape, shape))
      assert(!model.getLayers.last.asInstanceOf[OutputLayer].output.isOutput)
    }

    it("with buildOutput called should have an output layer") {
      model.add(Dense(shape, shape))
      model.buildOutput(LossFunction.NEGATIVELOGLIKELIHOOD)
      assert(model.getLayers.last.asInstanceOf[OutputLayer].output.isOutput)
    }

  }
}
