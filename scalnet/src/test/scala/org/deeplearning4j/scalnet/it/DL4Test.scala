/*
 * Copyright 2016 Skymind
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.deeplearning4j.scalnet.it

import org.deeplearning4j.scalnet.examples.dl4j.feedforward.IrisCSVExample
import org.deeplearning4j.scalnet.examples.dl4j.recurrent.{ BasicRNNExample, RNNEmbeddingExample }
import org.scalatest.{ Matchers, WordSpec }

import scala.util.Try

/**
  * A suite of basic, short and non cpu-heavy integration tests. Only test if example is run without errors.
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
