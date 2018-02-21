package org.deeplearning4j.scalnet.layers

import org.scalatest.{ Matchers, WordSpec }

class EmbeddingLayerTest extends WordSpec with Matchers {

  "An embedding layer" should {

    "have an input layer of size 100" in {
      val embeddingLayer = EmbeddingLayer(100, 100)
      embeddingLayer.inputShape shouldBe List(100, 100)
    }

    "have an ouput layer of size 100" in {
      val embeddingLayer = EmbeddingLayer(100, 100)
      embeddingLayer.outputShape shouldBe List(100, 100)
    }

    "compile to a DL4J EmbeddingLayer" in {
      val embeddingLayer = EmbeddingLayer(100, 100)
      val compiledLayer = embeddingLayer.compile
      compiledLayer.isInstanceOf[org.deeplearning4j.nn.conf.layers.EmbeddingLayer] shouldBe true
    }

  }

}
