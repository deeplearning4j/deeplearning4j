/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.linalg.factory.ops;

import static org.nd4j.linalg.factory.NDValidation.isSameType;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.GRUWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.SRUWeights;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;

public class NDRNN {
  public NDRNN() {
  }

  /**
   * The GRU cell.  Does a single time step operation<br>
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param hLast Output of the previous cell/time step, with shape [batchSize, numUnits] (NUMERIC type)
   * @param GRUWeights Configuration Object
   * @return output The cell's outputs. (NUMERIC type)
   */
  public INDArray gru(INDArray x, INDArray hLast, GRUWeights GRUWeights) {
    NDValidation.validateNumerical("gru", "x", x);
    NDValidation.validateNumerical("gru", "hLast", hLast);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell(x, hLast, GRUWeights))[0];
  }

  /**
   * The LSTM cell.  Does a single time step operation.<br>
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param cLast Previous cell state, with shape [batchSize, numUnits] (NUMERIC type)
   * @param yLast revious cell output, with shape [batchSize, numUnits] (NUMERIC type)
   * @param LSTMWeights Configuration Object
   * @param LSTMConfiguration Configuration Object
   * @return output The cell's outputs (NUMERIC type)
   */
  public INDArray lstmCell(INDArray x, INDArray cLast, INDArray yLast, LSTMWeights LSTMWeights,
      LSTMConfiguration LSTMConfiguration) {
    NDValidation.validateNumerical("lstmCell", "x", x);
    NDValidation.validateNumerical("lstmCell", "cLast", cLast);
    NDValidation.validateNumerical("lstmCell", "yLast", yLast);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlockCell(x, cLast, yLast, LSTMWeights, LSTMConfiguration))[0];
  }

  /**
   * The LSTM layer.  Does multiple time steps.<br>
   *
   * @param maxTSLength  (NUMERIC type)
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param cLast Previous/initial cell state, with shape [batchSize, numUnits] (NUMERIC type)
   * @param yLast Previous/initial cell output, with shape [batchSize, numUnits] (NUMERIC type)
   * @param LSTMWeights Configuration Object
   * @param LSTMConfiguration Configuration Object
   * @return output The layer's outputs. (NUMERIC type)
   */
  public INDArray lstmLayer(INDArray maxTSLength, INDArray x, INDArray cLast, INDArray yLast,
      LSTMWeights LSTMWeights, LSTMConfiguration LSTMConfiguration) {
    NDValidation.validateNumerical("lstmLayer", "maxTSLength", maxTSLength);
    NDValidation.validateNumerical("lstmLayer", "x", x);
    NDValidation.validateNumerical("lstmLayer", "cLast", cLast);
    NDValidation.validateNumerical("lstmLayer", "yLast", yLast);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer(maxTSLength, x, cLast, yLast, LSTMWeights, LSTMConfiguration))[0];
  }

  /**
   * The SRU layer.  Does a single time step operation.<br>
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param cLast Previous cell state, with shape [batchSize, inSize] (NUMERIC type)
   * @param SRUWeights Configuration Object
   * @return output The cell's outputs. (NUMERIC type)
   */

//  TODO figure out with overloading of methods below
  public INDArray sruCell(INDArray x, INDArray cLast, SRUWeights SRUWeights) {
    NDValidation.validateNumerical("sruCell", "x", x);
    NDValidation.validateNumerical("sruCell", "cLast", cLast);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.SRUCell(x, cLast, SRUWeights))[0];
  }

  /**
   * The SRU layer.  Does a single time step operation.<br>
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param initialC Initial cell state, with shape [batchSize, inSize] (NUMERIC type)
   * @param SRUWeights Configuration Object
   * @return output The cell's outputs.. (NUMERIC type)
   */
  public INDArray sruCell(INDArray x, INDArray initialC, SRUWeights SRUWeights) {
    NDValidation.validateNumerical("sruCell", "x", x);
    NDValidation.validateNumerical("sruCell", "initialC", initialC);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU(x, initialC, SRUWeights))[0];
  }

  /**
   * The SRU layer.  Does a single time step operation.<br>
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param initialC Initial cell state, with shape [batchSize, inSize] (NUMERIC type)
   * @param mask An optional dropout mask, with shape [batchSize, inSize] (NUMERIC type)
   * @param SRUWeights Configuration Object
   * @return output The cell's outputs.. (NUMERIC type)
   */
  public INDArray sruCell(INDArray x, INDArray initialC, INDArray mask, SRUWeights SRUWeights) {
    NDValidation.validateNumerical("sruCell", "x", x);
    NDValidation.validateNumerical("sruCell", "initialC", initialC);
    NDValidation.validateNumerical("sruCell", "mask", mask);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU(x, initialC, mask, SRUWeights))[0];
  }
}
