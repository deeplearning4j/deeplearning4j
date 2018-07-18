/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.scalnet.layers.recurrent

import org.scalatest.{ Matchers, WordSpec }

class LSTMTest extends WordSpec with Matchers {

  "A LSTM layer" should {

    "have an input layer of shape (10, 100)" in {
      val LSTMLayer = LSTM(10, 100)
      LSTMLayer.inputShape shouldBe List(10, 100)
    }

    "have an ouput layer of shape (10, 100)" in {
      val LSTMLayer = LSTM(10, 100)
      LSTMLayer.outputShape shouldBe List(100, 10)
    }

    "compile to a DL4J LSTM" in {
      val LSTMLayer = LSTM(10, 100)
      val compiledLayer = LSTMLayer.compile
      compiledLayer.isInstanceOf[org.deeplearning4j.nn.conf.layers.LSTM] shouldBe true
    }

  }
}
