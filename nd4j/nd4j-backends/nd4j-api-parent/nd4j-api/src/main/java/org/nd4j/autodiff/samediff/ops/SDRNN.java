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

package org.nd4j.autodiff.samediff.ops;

import java.lang.String;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlockCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs.GRUCellOutputs;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs.LSTMCellOutputs;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.GRUWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.SRUWeights;

public class SDRNN extends SDOps {
  public SDRNN(SameDiff sameDiff) {
    super(sameDiff);
  }

  /**
   * The GRU cell.  Does a single time step operation<br>
   *
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param hLast Output of the previous cell/time step, with shape [batchSize, numUnits] (NUMERIC type)
   * @param GRUWeights Configuration Object
   * @return output The cell's outputs. (NUMERIC type)
   */
  public SDVariable gru(SDVariable x, SDVariable hLast, GRUWeights GRUWeights) {
    SDValidation.validateNumerical("gru", "x", x);
    SDValidation.validateNumerical("gru", "hLast", hLast);
    return new org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell(sd,x, hLast, GRUWeights).outputVariable();
  }

  /**
   * The GRU cell.  Does a single time step operation<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param hLast Output of the previous cell/time step, with shape [batchSize, numUnits] (NUMERIC type)
   * @param GRUWeights Configuration Object
   * @return output The cell's outputs. (NUMERIC type)
   */
  public GRUCellOutputs gru(String name, SDVariable x, SDVariable hLast, GRUWeights GRUWeights) {
    SDValidation.validateNumerical("gru", "x", x);
    SDValidation.validateNumerical("gru", "hLast", hLast);
    GRUCell c =  new GRUCell(sd,x, hLast, GRUWeights);
    return new GRUCellOutputs(c.outputVariables(name));
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
  public LSTMCellOutputs lstmCell(SDVariable x, SDVariable cLast, SDVariable yLast,
      LSTMWeights LSTMWeights, LSTMConfiguration LSTMConfiguration) {
    SDValidation.validateNumerical("lstmCell", "x", x);
    SDValidation.validateNumerical("lstmCell", "cLast", cLast);
    SDValidation.validateNumerical("lstmCell", "yLast", yLast);
    LSTMBlockCell c = new LSTMBlockCell(sd,x, cLast, yLast, LSTMWeights, LSTMConfiguration);
    return new LSTMCellOutputs(c.outputVariables());
  }

  /**
   * The LSTM cell.  Does a single time step operation.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input, with shape [batchSize, inSize] (NUMERIC type)
   * @param cLast Previous cell state, with shape [batchSize, numUnits] (NUMERIC type)
   * @param yLast revious cell output, with shape [batchSize, numUnits] (NUMERIC type)
   * @param LSTMWeights Configuration Object
   * @param LSTMConfiguration Configuration Object
   * @return output The cell's outputs (NUMERIC type)
   */
  public LSTMCellOutputs lstmCell(String name, SDVariable x, SDVariable cLast, SDVariable yLast,
      LSTMWeights LSTMWeights, LSTMConfiguration LSTMConfiguration) {
    SDValidation.validateNumerical("lstmCell", "x", x);
    SDValidation.validateNumerical("lstmCell", "cLast", cLast);
    SDValidation.validateNumerical("lstmCell", "yLast", yLast);
    LSTMBlockCell c =  new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlockCell(sd,x, cLast, yLast, LSTMWeights, LSTMConfiguration);
    return new LSTMCellOutputs(c.outputVariables(name));
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
  public SDVariable lstmLayer(SDVariable maxTSLength, SDVariable x, SDVariable cLast,
      SDVariable yLast, LSTMWeights LSTMWeights, LSTMConfiguration LSTMConfiguration) {
    SDValidation.validateNumerical("lstmLayer", "maxTSLength", maxTSLength);
    SDValidation.validateNumerical("lstmLayer", "x", x);
    SDValidation.validateNumerical("lstmLayer", "cLast", cLast);
    SDValidation.validateNumerical("lstmLayer", "yLast", yLast);
    return new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer(sd,maxTSLength, x, cLast, yLast, LSTMWeights, LSTMConfiguration).outputVariable();
  }

  /**
   * The LSTM layer.  Does multiple time steps.<br>
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
  public SDVariable lstmLayer(String name, SDVariable maxTSLength, SDVariable x, SDVariable cLast,
      SDVariable yLast, LSTMWeights LSTMWeights, LSTMConfiguration LSTMConfiguration) {
    SDValidation.validateNumerical("lstmLayer", "maxTSLength", maxTSLength);
    SDValidation.validateNumerical("lstmLayer", "x", x);
    SDValidation.validateNumerical("lstmLayer", "cLast", cLast);
    SDValidation.validateNumerical("lstmLayer", "yLast", yLast);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer(sd,maxTSLength, x, cLast, yLast, LSTMWeights, LSTMConfiguration).outputVariable();
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
