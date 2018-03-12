package org.deeplearning4j.scalnet.it

import org.deeplearning4j.scalnet.examples.dl4j.feedforward.IrisCSVExample
import org.deeplearning4j.scalnet.examples.dl4j.recurrent.{BasicRNNExample, RNNEmbeddingExample}
import org.scalatest.{Matchers, WordSpec}

import scala.util.Try

/**
  * A suite of basic, short and non cpu-heavy integration tests which only test if example is run without errors
  */
class DL4Test extends WordSpec with Matchers {

  "DL4J integration tests" should {

    "ensure that Iris example run without errors" in {
      val runExample = Try(IrisCSVExample.main(Array("")))
      assert(runExample.isSuccess)
    }

    "ensure that basic RNN example run without errors" in {
      val runExample = Try(BasicRNNExample.main(Array("")))
      assert(runExample.isSuccess)
    }

    "ensure that RNN embedding example run without errors" in {
      val runExample = Try(RNNEmbeddingExample.main(Array("")))
      assert(runExample.isSuccess)
    }

  }

}
