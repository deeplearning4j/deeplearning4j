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

package org.deeplearning4j.scalnet.layers.recurrent

import org.scalatest.{ Matchers, WordSpec }

class BidirectionalTest extends WordSpec with Matchers {

  "A Bidirectional wrapper layer" should {

    "compile to a DL4J Bidirectional wrapper layer with a LSTM" in {
      val bidirectionalLSTM = Bidirectional(LSTM(10, 100))
      val compiledLayer = bidirectionalLSTM.compile
      compiledLayer.isInstanceOf[org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional] shouldBe true
    }

    "compile to a DL4J Bidirectional wrapper layer with a GravesLSTM" in {
      val bidirectionalLSTM = Bidirectional(GravesLSTM(10, 100))
      val compiledLayer = bidirectionalLSTM.compile
      compiledLayer.isInstanceOf[org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional] shouldBe true
    }

  }
}
