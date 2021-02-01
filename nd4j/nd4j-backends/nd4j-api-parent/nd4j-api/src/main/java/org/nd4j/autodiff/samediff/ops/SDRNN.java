/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.autodiff.samediff.ops;

import java.lang.String;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.GRUWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.SRUWeights;

public class SDRNN extends SDOps {
  public SDRNN(SameDiff sameDiff) {
    super(sameDiff);
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
  public SDVariable gru(SDVariable x, SDVariable hLast, SDVariable Wx, SDVariable Wh,
      SDVariable biases) {
    SDValidation.validateNumerical("gru", "x", x);
    SDValidation.validateNumerical("gru", "hLast", hLast);
    SDValidation.validateNumerical("gru", "Wx", Wx);
    SDValidation.validateNumerical("gru", "Wh", Wh);
    SDValidation.validateNumerical("gru", "biases", biases);
    return new org.nd4j.linalg.api.ops.impl.layers.recurrent.GRU(sd,x, hLast, Wx, Wh, biases).outputVariable();
  }

  /**
   * The GRU operation. Gated Recurrent Unit - Cho et al. 2014.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x input [time, bS, nIn] (NUMERIC type)
   * @param hLast initial cell output (at time step = 0) [bS, nOut] (NUMERIC type)
   * @param Wx input-to-hidden  weights, [nIn, 3*nOut] (NUMERIC type)
   * @param Wh hidden-to-hidden weights, [nOut, 3*nOut] (NUMERIC type)
   * @param biases biases, [3*nOut] (NUMERIC type)
   * @return h cell outputs [time, bS, nOut], that is per each time step (NUMERIC type)
   */
  public SDVariable gru(String name, SDVariable x, SDVariable hLast, SDVariable Wx, SDVariable Wh,
      SDVariable biases) {
    SDValidation.validateNumerical("gru", "x", x);
    SDValidation.validateNumerical("gru", "hLast", hLast);
    SDValidation.validateNumerical("gru", "Wx", Wx);
    SDValidation.validateNumerical("gru", "Wh", Wh);
    SDValidation.validateNumerical("gru", "biases", biases);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.recurrent.GRU(sd,x, hLast, Wx, Wh, biases).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * The GRU cell.  Does a single time step operation<br>
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param hLast Output of the previous cell/time step, with shape [batchSize, numUnits] (NUMERIC type)
   * @param GRUWeights Configuration Object
   */
  public SDVariable[] gruCell(SDVariable x, SDVariable hLast, GRUWeights GRUWeights) {
    SDValidation.validateNumerical("gruCell", "x", x);
    SDValidation.validateNumerical("gruCell", "hLast", hLast);
    return new org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell(sd,x, hLast, GRUWeights).outputVariables();
  }

  /**
   * The GRU cell.  Does a single time step operation<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param hLast Output of the previous cell/time step, with shape [batchSize, numUnits] (NUMERIC type)
   * @param GRUWeights Configuration Object
   */
  public SDVariable[] gruCell(String[] names, SDVariable x, SDVariable hLast,
      GRUWeights GRUWeights) {
    SDValidation.validateNumerical("gruCell", "x", x);
    SDValidation.validateNumerical("gruCell", "hLast", hLast);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell(sd,x, hLast, GRUWeights).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] lstmCell(SDVariable x, SDVariable cLast, SDVariable yLast,
      LSTMWeights LSTMWeights, LSTMConfiguration LSTMConfiguration) {
    SDValidation.validateNumerical("lstmCell", "x", x);
    SDValidation.validateNumerical("lstmCell", "cLast", cLast);
    SDValidation.validateNumerical("lstmCell", "yLast", yLast);
    return new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlockCell(sd,x, cLast, yLast, LSTMWeights, LSTMConfiguration).outputVariables();
  }

  /**
   * The LSTM cell.  Does a single time step operation.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param cLast Previous cell state, with shape [batchSize, numUnits] (NUMERIC type)
   * @param yLast revious cell output, with shape [batchSize, numUnits] (NUMERIC type)
   * @param LSTMWeights Configuration Object
   * @param LSTMConfiguration Configuration Object
   */
  public SDVariable[] lstmCell(String[] names, SDVariable x, SDVariable cLast, SDVariable yLast,
      LSTMWeights LSTMWeights, LSTMConfiguration LSTMConfiguration) {
    SDValidation.validateNumerical("lstmCell", "x", x);
    SDValidation.validateNumerical("lstmCell", "cLast", cLast);
    SDValidation.validateNumerical("lstmCell", "yLast", yLast);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlockCell(sd,x, cLast, yLast, LSTMWeights, LSTMConfiguration).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] lstmLayer(SDVariable x, SDVariable cLast, SDVariable yLast,
      SDVariable maxTSLength, LSTMLayerWeights LSTMLayerWeights, LSTMLayerConfig LSTMLayerConfig) {
    SDValidation.validateNumerical("lstmLayer", "x", x);
    SDValidation.validateNumerical("lstmLayer", "cLast", cLast);
    SDValidation.validateNumerical("lstmLayer", "yLast", yLast);
    SDValidation.validateNumerical("lstmLayer", "maxTSLength", maxTSLength);
    return new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer(sd,x, cLast, yLast, maxTSLength, LSTMLayerWeights, LSTMLayerConfig).outputVariables();
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
   * @param names names May be null. Arrays of names for the output variables.
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param cLast Previous/initial cell state, with shape [batchSize, numUnits] (NUMERIC type)
   * @param yLast Previous/initial cell output, with shape [batchSize, numUnits] (NUMERIC type)
   * @param maxTSLength maxTSLength with shape [batchSize] (NUMERIC type)
   * @param LSTMLayerWeights Configuration Object
   * @param LSTMLayerConfig Configuration Object
   */
  public SDVariable[] lstmLayer(String[] names, SDVariable x, SDVariable cLast, SDVariable yLast,
      SDVariable maxTSLength, LSTMLayerWeights LSTMLayerWeights, LSTMLayerConfig LSTMLayerConfig) {
    SDValidation.validateNumerical("lstmLayer", "x", x);
    SDValidation.validateNumerical("lstmLayer", "cLast", cLast);
    SDValidation.validateNumerical("lstmLayer", "yLast", yLast);
    SDValidation.validateNumerical("lstmLayer", "maxTSLength", maxTSLength);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer(sd,x, cLast, yLast, maxTSLength, LSTMLayerWeights, LSTMLayerConfig).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] lstmLayer(SDVariable x, LSTMLayerWeights LSTMLayerWeights,
      LSTMLayerConfig LSTMLayerConfig) {
    SDValidation.validateNumerical("lstmLayer", "x", x);
    return new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer(sd,x, null, null, null, LSTMLayerWeights, LSTMLayerConfig).outputVariables();
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
   * @param names names May be null. Arrays of names for the output variables.
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param LSTMLayerWeights Configuration Object
   * @param LSTMLayerConfig Configuration Object
   */
  public SDVariable[] lstmLayer(String[] names, SDVariable x, LSTMLayerWeights LSTMLayerWeights,
      LSTMLayerConfig LSTMLayerConfig) {
    SDValidation.validateNumerical("lstmLayer", "x", x);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer(sd,x, null, null, null, LSTMLayerWeights, LSTMLayerConfig).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable lstmblock(SDVariable maxTSLength, SDVariable x, SDVariable cLast,
      SDVariable yLast, LSTMWeights LSTMWeights, LSTMConfiguration LSTMConfiguration) {
    SDValidation.validateNumerical("lstmblock", "maxTSLength", maxTSLength);
    SDValidation.validateNumerical("lstmblock", "x", x);
    SDValidation.validateNumerical("lstmblock", "cLast", cLast);
    SDValidation.validateNumerical("lstmblock", "yLast", yLast);
    return new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlock(sd,maxTSLength, x, cLast, yLast, LSTMWeights, LSTMConfiguration).outputVariable();
  }

  /**
   * The LSTM block<br>
   *
   * @param name name May be null. Name for the output variable
   * @param maxTSLength  (NUMERIC type)
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param cLast Previous/initial cell state, with shape [batchSize, numUnits] (NUMERIC type)
   * @param yLast Previous/initial cell output, with shape [batchSize, numUnits] (NUMERIC type)
   * @param LSTMWeights Configuration Object
   * @param LSTMConfiguration Configuration Object
   * @return output The layer's outputs. (NUMERIC type)
   */
  public SDVariable lstmblock(String name, SDVariable maxTSLength, SDVariable x, SDVariable cLast,
      SDVariable yLast, LSTMWeights LSTMWeights, LSTMConfiguration LSTMConfiguration) {
    SDValidation.validateNumerical("lstmblock", "maxTSLength", maxTSLength);
    SDValidation.validateNumerical("lstmblock", "x", x);
    SDValidation.validateNumerical("lstmblock", "cLast", cLast);
    SDValidation.validateNumerical("lstmblock", "yLast", yLast);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlock(sd,maxTSLength, x, cLast, yLast, LSTMWeights, LSTMConfiguration).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * The LSTM block<br>
   *
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param LSTMWeights Configuration Object
   * @param LSTMConfiguration Configuration Object
   * @return output The layer's outputs. (NUMERIC type)
   */
  public SDVariable lstmblock(SDVariable x, LSTMWeights LSTMWeights,
      LSTMConfiguration LSTMConfiguration) {
    SDValidation.validateNumerical("lstmblock", "x", x);
    return new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlock(sd,null, x, null, null, LSTMWeights, LSTMConfiguration).outputVariable();
  }

  /**
   * The LSTM block<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param LSTMWeights Configuration Object
   * @param LSTMConfiguration Configuration Object
   * @return output The layer's outputs. (NUMERIC type)
   */
  public SDVariable lstmblock(String name, SDVariable x, LSTMWeights LSTMWeights,
      LSTMConfiguration LSTMConfiguration) {
    SDValidation.validateNumerical("lstmblock", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlock(sd,null, x, null, null, LSTMWeights, LSTMConfiguration).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable sru(SDVariable x, SDVariable initialC, SDVariable mask, SRUWeights SRUWeights) {
    SDValidation.validateNumerical("sru", "x", x);
    SDValidation.validateNumerical("sru", "initialC", initialC);
    SDValidation.validateNumerical("sru", "mask", mask);
    return new org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU(sd,x, initialC, mask, SRUWeights).outputVariable();
  }

  /**
   * The SRU layer.  Does a single time step operation.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param initialC Initial cell state, with shape [batchSize, inSize] (NUMERIC type)
   * @param mask An optional dropout mask, with shape [batchSize, inSize] (NUMERIC type)
   * @param SRUWeights Configuration Object
   * @return output The cell's outputs.. (NUMERIC type)
   */
  public SDVariable sru(String name, SDVariable x, SDVariable initialC, SDVariable mask,
      SRUWeights SRUWeights) {
    SDValidation.validateNumerical("sru", "x", x);
    SDValidation.validateNumerical("sru", "initialC", initialC);
    SDValidation.validateNumerical("sru", "mask", mask);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU(sd,x, initialC, mask, SRUWeights).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * The SRU layer.  Does a single time step operation.<br>
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param initialC Initial cell state, with shape [batchSize, inSize] (NUMERIC type)
   * @param SRUWeights Configuration Object
   * @return output The cell's outputs.. (NUMERIC type)
   */
  public SDVariable sru(SDVariable x, SDVariable initialC, SRUWeights SRUWeights) {
    SDValidation.validateNumerical("sru", "x", x);
    SDValidation.validateNumerical("sru", "initialC", initialC);
    return new org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU(sd,x, initialC, null, SRUWeights).outputVariable();
  }

  /**
   * The SRU layer.  Does a single time step operation.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param initialC Initial cell state, with shape [batchSize, inSize] (NUMERIC type)
   * @param SRUWeights Configuration Object
   * @return output The cell's outputs.. (NUMERIC type)
   */
  public SDVariable sru(String name, SDVariable x, SDVariable initialC, SRUWeights SRUWeights) {
    SDValidation.validateNumerical("sru", "x", x);
    SDValidation.validateNumerical("sru", "initialC", initialC);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU(sd,x, initialC, null, SRUWeights).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * The SRU layer.  Does a single time step operation.<br>
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param cLast Previous cell state, with shape [batchSize, inSize] (NUMERIC type)
   * @param SRUWeights Configuration Object
   * @return output The cell's outputs. (NUMERIC type)
   */
  public SDVariable sruCell(SDVariable x, SDVariable cLast, SRUWeights SRUWeights) {
    SDValidation.validateNumerical("sruCell", "x", x);
    SDValidation.validateNumerical("sruCell", "cLast", cLast);
    return new org.nd4j.linalg.api.ops.impl.layers.recurrent.SRUCell(sd,x, cLast, SRUWeights).outputVariable();
  }

  /**
   * The SRU layer.  Does a single time step operation.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param cLast Previous cell state, with shape [batchSize, inSize] (NUMERIC type)
   * @param SRUWeights Configuration Object
   * @return output The cell's outputs. (NUMERIC type)
   */
  public SDVariable sruCell(String name, SDVariable x, SDVariable cLast, SRUWeights SRUWeights) {
    SDValidation.validateNumerical("sruCell", "x", x);
    SDValidation.validateNumerical("sruCell", "cLast", cLast);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.recurrent.SRUCell(sd,x, cLast, SRUWeights).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }
}
