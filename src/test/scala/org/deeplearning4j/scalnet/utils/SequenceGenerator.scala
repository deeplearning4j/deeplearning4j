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

package org.deeplearning4j.scalnet.utils

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

object SequenceGenerator {

  def generate(rows: Int, timesteps: Int, threshold: Double = 0.5, seed: Long = 1234): DataSet = {
    Nd4j.getRandom.setSeed(seed)
    val x = Nd4j.rand(rows, timesteps)
    val y = Nd4j.create(rows, timesteps)
    for (i <- 0 until rows; j <- 0 until timesteps) {
      val cumulativeSum = Nd4j.cumsum(x.getRow(i), 1)
      val limit = cumulativeSum.max(1).getDouble(0) * threshold
      y.putScalar(i, j, if (cumulativeSum.getDouble(i, j) > limit) 1 else 0)
    }
    new DataSet(
      x.reshape(rows, timesteps, 1),
      y.reshape(rows, timesteps, 1)
    )
  }

}
