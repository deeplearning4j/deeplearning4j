package org.deeplearning4j.scalnet.layers.recurrent

import org.deeplearning4j.scalnet.layers.GravesLSTM
import org.scalatest.{ Matchers, WordSpec }

class GravesLSTMTest extends WordSpec with Matchers {

  "A Graves LSTM layer" should {

    "have an input layer of shape (10, 100)" in {
      val gravesLSTMLayer = GravesLSTM(10, 100)
      gravesLSTMLayer.inputShape shouldBe List(10, 100)
    }

    "have an ouput layer of shape (10, 100)" in {
      val gravesLSTMLayer = GravesLSTM(10, 100)
      gravesLSTMLayer.outputShape shouldBe List(100, 10)
    }

    "compile to a DL4J GravesLSTM" in {
      val gravesLSTMLayer = GravesLSTM(10, 100)
      val compiledLayer = gravesLSTMLayer.compile
      compiledLayer.isInstanceOf[org.deeplearning4j.nn.conf.layers.GravesLSTM] shouldBe true
    }

  }
}
