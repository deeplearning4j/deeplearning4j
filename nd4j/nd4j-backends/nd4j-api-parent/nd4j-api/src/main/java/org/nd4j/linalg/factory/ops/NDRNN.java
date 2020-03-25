/*******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
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
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.SRUWeights;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.enums.CellAct;
import org.nd4j.linalg.factory.enums.GateAct;
import org.nd4j.linalg.factory.enums.LSTMDataFormat;
import org.nd4j.linalg.factory.enums.LSTMDirectionMode;
import org.nd4j.linalg.factory.enums.OutAct;

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
   * The LSTM block<br>
   *
   * @param maxTSLength  (NUMERIC type)
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param cLast Previous/initial cell state, with shape [batchSize, numUnits] (NUMERIC type)
   * @param yLast Previous/initial cell output, with shape [batchSize, numUnits] (NUMERIC type)
   * @param LSTMWeights Configuration Object
   * @param LSTMConfiguration Configuration Object
   * @return output The layer's outputs. (NUMERIC type)
   */
  public INDArray lstmblock(INDArray maxTSLength, INDArray x, INDArray cLast, INDArray yLast,
      LSTMWeights LSTMWeights, LSTMConfiguration LSTMConfiguration) {
    NDValidation.validateNumerical("lstmblock", "maxTSLength", maxTSLength);
    NDValidation.validateNumerical("lstmblock", "x", x);
    NDValidation.validateNumerical("lstmblock", "cLast", cLast);
    NDValidation.validateNumerical("lstmblock", "yLast", yLast);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlock(maxTSLength, x, cLast, yLast, LSTMWeights, LSTMConfiguration))[0];
  }

  /**
   * The LSTM layer<br>
   *
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param cLast Previous/initial cell state, with shape [batchSize, numUnits] (NUMERIC type)
   * @param yLast Previous/initial cell output, with shape [batchSize, numUnits] (NUMERIC type)
   * @param maxTSLength  (NUMERIC type)
   * @param LSTMDataFormat for unidirectional:
   *   TNS: shape [timeLength, numExamples, inOutSize] - sometimes referred to as "time major"<br>
   *   NST: shape [numExamples, inOutSize, timeLength]<br>
   *   NTS: shape [numExamples, timeLength, inOutSize] - TF "time_major=false" layout<br>
   *  for bidirectional:
   *    T2NS: 3 = [timeLength, 2, numExamples, inOutSize] (for ONNX)
   * @param LSTMDirectionMode direction <br>
   *  FWD: 0 = fwd
   *  BWD: 1 = bwd
   *  BS: 2 = bidirectional sum
   *  BC: 3 = bidirectional concat
   *  BE: 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3)
   * @param gateAct Activations
   * @param cellAct Activations
   * @param outAct Activations
   * @param retFullSequence indicates whether to return whole time sequence h {h_0, h_1, ... , h_sL-1}
   * @param retLastH indicates whether to return output at last time step only,
   *  in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
   * @param retLastC indicates whether to return cells state at last time step only,
   *  in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
   * @param cellClip Cell clipping value, if it = 0 then do not apply clipping
   * @param LSTMLayerWeights Configuration Object
   * @return output The layer's outputs. (NUMERIC type)
   */
  public INDArray lstmlayer(INDArray x, INDArray cLast, INDArray yLast, INDArray maxTSLength,
      LSTMDataFormat LSTMDataFormat, LSTMDirectionMode LSTMDirectionMode, GateAct gateAct,
      CellAct cellAct, OutAct outAct, boolean retFullSequence, boolean retLastH, boolean retLastC,
      double cellClip, LSTMLayerWeights LSTMLayerWeights) {
    NDValidation.validateNumerical("lstmlayer", "x", x);
    NDValidation.validateNumerical("lstmlayer", "cLast", cLast);
    NDValidation.validateNumerical("lstmlayer", "yLast", yLast);
    NDValidation.validateNumerical("lstmlayer", "maxTSLength", maxTSLength);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer(x, cLast, yLast, maxTSLength, LSTMDataFormat, LSTMDirectionMode, gateAct, cellAct, outAct, retFullSequence, retLastH, retLastC, cellClip, LSTMLayerWeights))[0];
  }

  /**
   * The LSTM layer<br>
   *
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param cLast Previous/initial cell state, with shape [batchSize, numUnits] (NUMERIC type)
   * @param yLast Previous/initial cell output, with shape [batchSize, numUnits] (NUMERIC type)
   * @param maxTSLength  (NUMERIC type)
   * @param LSTMDataFormat for unidirectional:
   *   TNS: shape [timeLength, numExamples, inOutSize] - sometimes referred to as "time major"<br>
   *   NST: shape [numExamples, inOutSize, timeLength]<br>
   *   NTS: shape [numExamples, timeLength, inOutSize] - TF "time_major=false" layout<br>
   *  for bidirectional:
   *    T2NS: 3 = [timeLength, 2, numExamples, inOutSize] (for ONNX)
   * @param LSTMDirectionMode direction <br>
   *  FWD: 0 = fwd
   *  BWD: 1 = bwd
   *  BS: 2 = bidirectional sum
   *  BC: 3 = bidirectional concat
   *  BE: 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3)
   * @param gateAct Activations
   * @param cellAct Activations
   * @param outAct Activations
   * @param retLastH indicates whether to return output at last time step only,
   *  in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
   * @param retLastC indicates whether to return cells state at last time step only,
   *  in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
   * @param cellClip Cell clipping value, if it = 0 then do not apply clipping
   * @param LSTMLayerWeights Configuration Object
   * @return output The layer's outputs. (NUMERIC type)
   */
  public INDArray lstmlayer(INDArray x, INDArray cLast, INDArray yLast, INDArray maxTSLength,
      LSTMDataFormat LSTMDataFormat, LSTMDirectionMode LSTMDirectionMode, GateAct gateAct,
      CellAct cellAct, OutAct outAct, boolean retLastH, boolean retLastC, double cellClip,
      LSTMLayerWeights LSTMLayerWeights) {
    NDValidation.validateNumerical("lstmlayer", "x", x);
    NDValidation.validateNumerical("lstmlayer", "cLast", cLast);
    NDValidation.validateNumerical("lstmlayer", "yLast", yLast);
    NDValidation.validateNumerical("lstmlayer", "maxTSLength", maxTSLength);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer(x, cLast, yLast, maxTSLength, LSTMDataFormat, LSTMDirectionMode, gateAct, cellAct, outAct, true, retLastH, retLastC, cellClip, LSTMLayerWeights))[0];
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
  public INDArray sru(INDArray x, INDArray initialC, INDArray mask, SRUWeights SRUWeights) {
    NDValidation.validateNumerical("sru", "x", x);
    NDValidation.validateNumerical("sru", "initialC", initialC);
    NDValidation.validateNumerical("sru", "mask", mask);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU(x, initialC, mask, SRUWeights))[0];
  }

  /**
   * The SRU layer.  Does a single time step operation.<br>
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param initialC Initial cell state, with shape [batchSize, inSize] (NUMERIC type)
   * @param SRUWeights Configuration Object
   * @return output The cell's outputs.. (NUMERIC type)
   */
  public INDArray sru(INDArray x, INDArray initialC, SRUWeights SRUWeights) {
    NDValidation.validateNumerical("sru", "x", x);
    NDValidation.validateNumerical("sru", "initialC", initialC);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU(x, initialC, null, SRUWeights))[0];
  }

  /**
   * The SRU layer.  Does a single time step operation.<br>
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param cLast Previous cell state, with shape [batchSize, inSize] (NUMERIC type)
   * @param SRUWeights Configuration Object
   * @return output The cell's outputs. (NUMERIC type)
   */
  public INDArray sruCell(INDArray x, INDArray cLast, SRUWeights SRUWeights) {
    NDValidation.validateNumerical("sruCell", "x", x);
    NDValidation.validateNumerical("sruCell", "cLast", cLast);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.SRUCell(x, cLast, SRUWeights))[0];
  }
}
