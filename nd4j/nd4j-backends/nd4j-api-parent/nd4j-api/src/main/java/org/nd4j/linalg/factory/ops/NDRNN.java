/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
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

package org.nd4j.linalg.factory.ops;

import static org.nd4j.linalg.factory.NDValidation.isSameType;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.GRU;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlock;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlockCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRUCell;
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
   * The GRU operation. Gated Recurrent Unit - Cho et al. 2014.
   *
   *
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
    INDArray[] __tmp = Nd4j.exec(new GRU(x, hLast, Wx, Wh, biases));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * The GRU cell.  Does a single time step operation
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param hLast Output of the previous cell/time step, with shape [batchSize, numUnits] (NUMERIC type)
   * @param GRUWeights Configuration Object
   * @return r Reset gate output (NUMERIC type)
   * @return u Update gate output (NUMERIC type)
   * @return c Cell gate output (NUMERIC type)
   * @return h Cell output (NUMERIC type)
   */
  public INDArray[] gruCell(INDArray x, INDArray hLast, GRUWeights GRUWeights) {
    NDValidation.validateNumerical("gruCell", "x", x);
    NDValidation.validateNumerical("gruCell", "hLast", hLast);
    return Nd4j.exec(new GRUCell(x, hLast, GRUWeights));
  }

  /**
   * The LSTM cell.  Does a single time step operation.
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param cLast Previous cell state, with shape [batchSize, numUnits] (NUMERIC type)
   * @param yLast revious cell output, with shape [batchSize, numUnits] (NUMERIC type)
   * @param LSTMWeights Configuration Object
   * @param LSTMConfiguration Configuration Object
   * @return i Output - input modulation gate activations [batchSize, numUnits]. (NUMERIC type)
   * @return c Output - Activations, cell state (pre tanh) [batchSize, numUnits]. (NUMERIC type)
   * @return f Output - forget gate activations [batchSize, numUnits]. (NUMERIC type)
   * @return o Output - output gate activations [batchSize, numUnits]. (NUMERIC type)
   * @return z Output - input gate activations [batchSize, numUnits]. (NUMERIC type)
   * @return h Cell state, post tanh [batchSize, numUnits]. (NUMERIC type)
   * @return y Current cell output [batchSize, numUnits]. (NUMERIC type)
   */
  public INDArray[] lstmCell(INDArray x, INDArray cLast, INDArray yLast, LSTMWeights LSTMWeights,
      LSTMConfiguration LSTMConfiguration) {
    NDValidation.validateNumerical("lstmCell", "x", x);
    NDValidation.validateNumerical("lstmCell", "cLast", cLast);
    NDValidation.validateNumerical("lstmCell", "yLast", yLast);
    return Nd4j.exec(new LSTMBlockCell(x, cLast, yLast, LSTMWeights, LSTMConfiguration));
  }

  /**
   * Long Short-Term Memory layer - Hochreiter 1997.
   * SUPPORTS following data formats:
   * for unidirectional:
   * TNS: shapes [timeLength, numExamples, inOutSize]
   * NST: shapes [numExamples, inOutSize, timeLength]
   * NTS: shapes [numExamples, timeLength, inOutSize]
   * for bidirectional:
   * T2NS: shapes [timeLength, 2, numExamples, inOutSize] (for ONNX)
   * SUPPORTS following direction modes:
   * FWD: forward
   * BWD: backward
   * BIDIR_SUM: bidirectional sum
   * BIDIR_CONCAT: bidirectional concat
   * BIDIR_EXTRA_DIM: bidirectional extra output dim (in conjunction with format dataFormat - T2NS)
   * You may use different gate configurations:
   * specify gate/cell/out aplha/beta and numbers of activations for gate/cell/out described in activations enum
   * ("RELU","SIGMOID","AFFINE","LEAKY_RELU","THRESHHOLD_RELU","SCALED_TAHN","HARD_SIGMOID","ELU","SOFTSIGN","SOFTPLUS")
   * Also this layer supports MKLDNN (DNNL) and cuDNN acceleration
   *
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param cLast Previous/initial cell state, with shape [batchSize, numUnits] (NUMERIC type)
   * @param yLast Previous/initial cell output, with shape [batchSize, numUnits] (NUMERIC type)
   * @param maxTSLength maxTSLength with shape [batchSize] (NUMERIC type)
   * @param LSTMLayerWeights Configuration Object
   * @param LSTMLayerConfig Configuration Object
   * @return output The layer's outputs - full time series (NUMERIC type)
   * @return yLast The layer's outputs - last time step activations (yLast) (NUMERIC type)
   * @return cLast The layer's outputs - last time step cell state (cLast) (NUMERIC type)
   */
  public INDArray[] lstmLayer(INDArray x, INDArray cLast, INDArray yLast, INDArray maxTSLength,
      LSTMLayerWeights LSTMLayerWeights, LSTMLayerConfig LSTMLayerConfig) {
    NDValidation.validateNumerical("lstmLayer", "x", x);
    if (cLast != null) {
      NDValidation.validateNumerical("lstmLayer", "cLast", cLast);
    }
    if (yLast != null) {
      NDValidation.validateNumerical("lstmLayer", "yLast", yLast);
    }
    if (maxTSLength != null) {
      NDValidation.validateNumerical("lstmLayer", "maxTSLength", maxTSLength);
    }
    return Nd4j.exec(new LSTMLayer(x, cLast, yLast, maxTSLength, LSTMLayerWeights, LSTMLayerConfig));
  }

  /**
   * Long Short-Term Memory layer - Hochreiter 1997.
   * SUPPORTS following data formats:
   * for unidirectional:
   * TNS: shapes [timeLength, numExamples, inOutSize]
   * NST: shapes [numExamples, inOutSize, timeLength]
   * NTS: shapes [numExamples, timeLength, inOutSize]
   * for bidirectional:
   * T2NS: shapes [timeLength, 2, numExamples, inOutSize] (for ONNX)
   * SUPPORTS following direction modes:
   * FWD: forward
   * BWD: backward
   * BIDIR_SUM: bidirectional sum
   * BIDIR_CONCAT: bidirectional concat
   * BIDIR_EXTRA_DIM: bidirectional extra output dim (in conjunction with format dataFormat - T2NS)
   * You may use different gate configurations:
   * specify gate/cell/out aplha/beta and numbers of activations for gate/cell/out described in activations enum
   * ("RELU","SIGMOID","AFFINE","LEAKY_RELU","THRESHHOLD_RELU","SCALED_TAHN","HARD_SIGMOID","ELU","SOFTSIGN","SOFTPLUS")
   * Also this layer supports MKLDNN (DNNL) and cuDNN acceleration
   *
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param LSTMLayerWeights Configuration Object
   * @param LSTMLayerConfig Configuration Object
   * @return output The layer's outputs - full time series (NUMERIC type)
   * @return yLast The layer's outputs - last time step activations (yLast) (NUMERIC type)
   * @return cLast The layer's outputs - last time step cell state (cLast) (NUMERIC type)
   */
  public INDArray[] lstmLayer(INDArray x, LSTMLayerWeights LSTMLayerWeights,
      LSTMLayerConfig LSTMLayerConfig) {
    NDValidation.validateNumerical("lstmLayer", "x", x);
    return Nd4j.exec(new LSTMLayer(x, null, null, null, LSTMLayerWeights, LSTMLayerConfig));
  }

  /**
   * The LSTM block
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
    if (maxTSLength != null) {
      NDValidation.validateNumerical("lstmblock", "maxTSLength", maxTSLength);
    }
    NDValidation.validateNumerical("lstmblock", "x", x);
    if (cLast != null) {
      NDValidation.validateNumerical("lstmblock", "cLast", cLast);
    }
    if (yLast != null) {
      NDValidation.validateNumerical("lstmblock", "yLast", yLast);
    }
    INDArray[] __tmp = Nd4j.exec(new LSTMBlock(maxTSLength, x, cLast, yLast, LSTMWeights, LSTMConfiguration));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * The LSTM block
   *
   * @param x  Input, with shape dependent on the data format (in config). (NUMERIC type)
   * @param LSTMWeights Configuration Object
   * @param LSTMConfiguration Configuration Object
   * @return output The layer's outputs. (NUMERIC type)
   */
  public INDArray lstmblock(INDArray x, LSTMWeights LSTMWeights,
      LSTMConfiguration LSTMConfiguration) {
    NDValidation.validateNumerical("lstmblock", "x", x);
    INDArray[] __tmp = Nd4j.exec(new LSTMBlock(null, x, null, null, LSTMWeights, LSTMConfiguration));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * The SRU layer.  Does a single time step operation.
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
    if (mask != null) {
      NDValidation.validateNumerical("sru", "mask", mask);
    }
    INDArray[] __tmp = Nd4j.exec(new SRU(x, initialC, mask, SRUWeights));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * The SRU layer.  Does a single time step operation.
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param initialC Initial cell state, with shape [batchSize, inSize] (NUMERIC type)
   * @param SRUWeights Configuration Object
   * @return output The cell's outputs.. (NUMERIC type)
   */
  public INDArray sru(INDArray x, INDArray initialC, SRUWeights SRUWeights) {
    NDValidation.validateNumerical("sru", "x", x);
    NDValidation.validateNumerical("sru", "initialC", initialC);
    INDArray[] __tmp = Nd4j.exec(new SRU(x, initialC, null, SRUWeights));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * The SRU layer.  Does a single time step operation.
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param cLast Previous cell state, with shape [batchSize, inSize] (NUMERIC type)
   * @param SRUWeights Configuration Object
   * @return output The cell's outputs. (NUMERIC type)
   */
  public INDArray sruCell(INDArray x, INDArray cLast, SRUWeights SRUWeights) {
    NDValidation.validateNumerical("sruCell", "x", x);
    NDValidation.validateNumerical("sruCell", "cLast", cLast);
    INDArray[] __tmp = Nd4j.exec(new SRUCell(x, cLast, SRUWeights));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }
}
