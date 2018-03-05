package org.deeplearning4j.scalnet.models

import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.scalnet.layers.convolutional.Convolution2D
import org.scalatest._
import org.deeplearning4j.scalnet.layers.{ Dense, OutputLayer }
import org.deeplearning4j.scalnet.layers.pooling.MaxPooling2D
import org.deeplearning4j.scalnet.layers.reshaping.{ Flatten3D, Unflatten3D }
import org.deeplearning4j.scalnet.regularizers.L2
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
  * Created by maxpumperla on 29/06/17.
  */
class SequentialTest extends FunSpec with BeforeAndAfter {

  var model: Sequential = Sequential()
  val shape = 100
  val wrongInputShape = 10
  val nbRows: Int = 28
  val nbColumns: Int = 28
  val nbChannels: Int = 1
  val nbOutput: Int = 10
  val weightDecay: Double = 0.0005
  val momentum: Double = 0.9
  val learningRate: Double = 0.01

  before {
    model = Sequential()
  }

  describe("A Sequential network") {

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

    it("should infer the correct shape of an incorrectly initialized layer") {
      model.add(Dense(shape, shape))
      model.add(Dense(shape, wrongInputShape))
      assert(model.getLayers.last.inputShape == List(shape))
    }

    it("should propagate the correct shape of all layers and preprocessors") {
      model.add(Unflatten3D(List(nbRows, nbColumns, nbChannels), nIn = nbRows * nbColumns))
      model.add(
        Convolution2D(nFilter = 20,
                      kernelSize = List(5, 5),
                      stride = List(1, 1),
                      weightInit = WeightInit.XAVIER,
                      regularizer = L2(weightDecay))
      )
      model.add(MaxPooling2D(kernelSize = List(2, 2), stride = List(2, 2)))
      model.add(
        Convolution2D(nFilter = 50,
                      kernelSize = List(5, 5),
                      stride = List(1, 1),
                      weightInit = WeightInit.XAVIER,
                      regularizer = L2(weightDecay))
      )
      model.add(MaxPooling2D(kernelSize = List(2, 2), stride = List(2, 2)))
      model.add(Flatten3D())

      val preprocessorOutShapes = model.getPreprocessors.values.map(_.outputShape)
      assert(preprocessorOutShapes == List(List(nbRows, nbColumns, nbChannels), List(4 * 4 * 50)))

      val layerOutShapes = model.getLayers.map(_.outputShape)
      assert(
        layerOutShapes == List(List(24, 24, 20), List(12, 12, 20), List(8, 8, 50), List(4, 4, 50))
      )

    }
  }
}
