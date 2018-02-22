package org.deeplearning4j.scalnet.layers

import org.scalatest.{ Matchers, WordSpec }

class EmbeddingLayerTest extends WordSpec with Matchers {

  "An embedding layer" should {

    "have an input layer of shape (10, 100)" in {
      val embeddingLayer = EmbeddingLayer(10, 100)
      embeddingLayer.inputShape shouldBe List(10, 100)
    }

    "have an ouput layer of shape (10, 100)" in {
      val embeddingLayer = EmbeddingLayer(10, 100)
      embeddingLayer.outputShape shouldBe List(100, 10)
    }

    "compile to a DL4J EmbeddingLayer" in {
      val embeddingLayer = EmbeddingLayer(10, 100)
      val compiledLayer = embeddingLayer.compile
      compiledLayer.isInstanceOf[org.deeplearning4j.nn.conf.layers.EmbeddingLayer] shouldBe true
    }

  }

}
