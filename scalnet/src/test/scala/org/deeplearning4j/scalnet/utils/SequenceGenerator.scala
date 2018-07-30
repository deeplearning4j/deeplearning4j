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

package org.deeplearning4j.scalnet.utils

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

/**
  * A sequence generator that output toy examples for sequence classification
  * Features are ramdom values between 0 and 1 and class is based on whether
  * cumulative sum crossed 'timesteps * threshold'.
  * ie. for 10 timesteps and 0.25 threshold: [0.6 0.2 0.9 0.9 0.3 0.6 0.8 0.1 0.8 0.2] [0 0 0 1 1 1 1 1 1 1]
  */
object SequenceGenerator {

  def generate(timesteps: Int, threshold: Double = 0.25): DataSet = {
    val x = Nd4j.rand(1, timesteps)
    val y = Nd4j.create(1, timesteps)
    for (i <- 0 until timesteps) {
      val cumulativeSum = Nd4j.cumsum(x.getRow(0), 1)
      val limit = timesteps * threshold
      y.putScalar(0, i, if (cumulativeSum.getDouble(0l, i) > limit) 1 else 0)
    }
    new DataSet(x.reshape(1, timesteps, 1), y.reshape(1, timesteps, 1))
  }

}
