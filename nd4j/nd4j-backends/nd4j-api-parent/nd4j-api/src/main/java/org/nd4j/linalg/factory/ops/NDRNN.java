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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.GRUWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.SRUWeights;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;

public class NDRNN {
  public NDRNN() {
  }

  /**
   * The GRU operation. Gated Recurrent Unit - Cho et al. 2014.<br>
   *
   * @param x input [time, bS, nIn] (NUMERIC type)
   * @param hLast initial cell output (at time step = 0) [bS, nOut] (NUMERIC type)
   * @param Wx input-to-hidden  weights, [nIn, 3*nOut] (NUMERIC type)
   * @param Wh hidden-to-hidden weights, [nOut, 3*nOut] (NUMERIC type)
   * @param biases biases, [3*nOut] (NUMERIC type)
   * @return h cell outputs [time, bS, nOut], that is per each time step (NUMERIC type)
   */
  public INDArray gru(INDArray x, INDArray hLast, INDArray Wx, INDArray Wh, INDArray biases) {
    NDValidation.validateNumerical("gru", "x", x);
    NDValidation.validateNumerical("gru", "hLast", hLast);
    NDValidation.validateNumerical("gru", "Wx", Wx);
    NDValidation.validateNumerical("gru", "Wh", Wh);
    NDValidation.validateNumerical("gru", "biases", biases);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.GRU(x, hLast, Wx, Wh, biases))[0];
  }

  /**
   * The GRU cell.  Does a single time step operation<br>
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param hLast Output of the previous cell/time step, with shape [batchSize, numUnits] (NUMERIC type)
   * @param GRUWeights Configuration Object
   */
  public INDArray[] gruCell(INDArray x, INDArray hLast, GRUWeights GRUWeights) {
    NDValidation.validateNumerical("gruCell", "x", x);
    NDValidation.validateNumerical("gruCell", "hLast", hLast);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell(x, hLast, GRUWeights));
  }

  /**
   * The LSTM cell.  Does a single time step operation.<br>
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param cLast Previous cell state, with shape [batchSize, numUnits] (NUMERIC type)
   * @param yLast revious cell output, with shape [batchSize, numUnits] (NUMERIC type)
   * @param LSTMWeights Configuration Object
   * @param LSTMConfiguration Configuration Object
   */
  public INDArray[] lstmCell(INDArray x, INDArray cLast, INDArray yLast, LSTMWeights LSTMWeights,
      LSTMConfiguration LSTMConfiguration) {
    NDValidation.validateNumerical("lstmCell", "x", x);
    NDValidation.validateNumerical("lstmCell", "cLast", cLast);
    NDValidation.validateNumerical("lstmCell", "yLast", yLast);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlockCell(x, cLast, yLast, LSTMWeights, LSTMConfiguration));
  }

  /**
   * Long Short-Term Memory layer - Hochreiter 1997.<br>
   * SUPPORTS following data formats:\n<br>
   * for unidirectional: \n" +<br>
   * TNS: shapes [timeLength, numExamples, inOutSize]\n<br>
   * NST: shapes [numExamples, inOutSize, timeLength]\n<br>
   * NTS: shapes [numExamples, timeLength, inOutSize]<br>
   * for bidirectional:\n<br>
   * T2NS: shapes [timeLength, 2, numExamples, inOutSize] (for ONNX)\n<br>
   * SUPPORTS following direction modes:\n<br>
   * FWD: forward<br>
   * BWD: backward<br>
   * BIDIR_SUM: bidirectional sum\n<br>
   * BIDIR_CONCAT: bidirectional concat\n" +<br>
   * BIDIR_EXTRA_DIM: bidirectional extra output dim (in conjunction with format dataFormat - T2NS)"<br>
   * You may use different gate configurations:<br>
   * specify gate/cell/out aplha/beta and numbers of activations for gate/cell/out described in activations enum\n<br>
   * ("RELU","SIGMOID","AFFINE","LEAKY_RELU","THRESHHOLD_RELU","SCALED_TAHN","HARD_SIGMOID","ELU","SOFTSIGN","SOFTPLUS")\n<br>
   * Also this layer supports MKLDNN (DNNL) and cuDNN acceleration<br>
   *
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param cLast Previous/initial cell state, with shape [batchSize, numUnits] (NUMERIC type)
   * @param yLast Previous/initial cell output, with shape [batchSize, numUnits] (NUMERIC type)
   * @param maxTSLength maxTSLength with shape [batchSize] (NUMERIC type)
   * @param LSTMLayerWeights Configuration Object
   * @param LSTMLayerConfig Configuration Object
   */
  public INDArray[] lstmLayer(INDArray x, INDArray cLast, INDArray yLast, INDArray maxTSLength,
      LSTMLayerWeights LSTMLayerWeights, LSTMLayerConfig LSTMLayerConfig) {
    NDValidation.validateNumerical("lstmLayer", "x", x);
    NDValidation.validateNumerical("lstmLayer", "cLast", cLast);
    NDValidation.validateNumerical("lstmLayer", "yLast", yLast);
    NDValidation.validateNumerical("lstmLayer", "maxTSLength", maxTSLength);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer(x, cLast, yLast, maxTSLength, LSTMLayerWeights, LSTMLayerConfig));
  }

  /**
   * Long Short-Term Memory layer - Hochreiter 1997.<br>
   * SUPPORTS following data formats:\n<br>
   * for unidirectional: \n" +<br>
   * TNS: shapes [timeLength, numExamples, inOutSize]\n<br>
   * NST: shapes [numExamples, inOutSize, timeLength]\n<br>
   * NTS: shapes [numExamples, timeLength, inOutSize]<br>
   * for bidirectional:\n<br>
   * T2NS: shapes [timeLength, 2, numExamples, inOutSize] (for ONNX)\n<br>
   * SUPPORTS following direction modes:\n<br>
   * FWD: forward<br>
   * BWD: backward<br>
   * BIDIR_SUM: bidirectional sum\n<br>
   * BIDIR_CONCAT: bidirectional concat\n" +<br>
   * BIDIR_EXTRA_DIM: bidirectional extra output dim (in conjunction with format dataFormat - T2NS)"<br>
   * You may use different gate configurations:<br>
   * specify gate/cell/out aplha/beta and numbers of activations for gate/cell/out described in activations enum\n<br>
   * ("RELU","SIGMOID","AFFINE","LEAKY_RELU","THRESHHOLD_RELU","SCALED_TAHN","HARD_SIGMOID","ELU","SOFTSIGN","SOFTPLUS")\n<br>
   * Also this layer supports MKLDNN (DNNL) and cuDNN acceleration<br>
   *
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param LSTMLayerWeights Configuration Object
   * @param LSTMLayerConfig Configuration Object
   */
  public INDArray[] lstmLayer(INDArray x, LSTMLayerWeights LSTMLayerWeights,
      LSTMLayerConfig LSTMLayerConfig) {
    NDValidation.validateNumerical("lstmLayer", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer(x, null, null, null, LSTMLayerWeights, LSTMLayerConfig));
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
   * The LSTM block<br>
   *
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param LSTMWeights Configuration Object
   * @param LSTMConfiguration Configuration Object
   * @return output The layer's outputs. (NUMERIC type)
   */
  public INDArray lstmblock(INDArray x, LSTMWeights LSTMWeights,
      LSTMConfiguration LSTMConfiguration) {
    NDValidation.validateNumerical("lstmblock", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlock(null, x, null, null, LSTMWeights, LSTMConfiguration))[0];
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
