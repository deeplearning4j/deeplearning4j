/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.nd4j.linalg.api.ops.impl.layers.recurrent.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlockCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer;

import java.util.LinkedHashMap;
import java.util.Map;


@Builder
@Data
public class LSTMLayerConfig {


    /**
     * notations <br>
     * bS - batch size
     * sL - sequence length, number of time steps
     * nIn - input size
     * nOut - output size (hidden size) <br<
     * <p>
     * for unidirectional:
     * SBN: 0 = [sL, bS, nIn],
     * BSN: 1 = [bS, sL ,nIn],
     * BNS: 2 = [bS, nIn, sL],
     * for bidirectional:
     * S2BN: 3 = [sL, 2, bS, nOut] (for ONNX)
     */
    @Builder.Default
    private LSTMDataFormat LSTMdataFormat = LSTMDataFormat.SBN;  //INT_ARG(0)


    /**
     * direction <br>
     * FWD: 0 = fwd
     * BWD: 1 = bwd
     * BS: 2 = bidirectional sum
     * BC: 3 = bidirectional concat
     * BE: 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3)
     */
    @Builder.Default
    private LSTMDirectionMode directionMode = LSTMDirectionMode.FWD;  //INT_ARG(1)

    /**
     * Activation for input (i), forget (f) and output (o) gates
     */
    @Builder.Default
    private LSTMActivations gateAct = LSTMActivations.RELU; // INT_ARG(2)

    @Builder.Default
    private LSTMActivations cellAct = LSTMActivations.RELU; // INT_ARG(3)

    @Builder.Default
    private LSTMActivations outAct = LSTMActivations.RELU; // INT_ARG(4)




    /**
     * indicates whether seqLen array is provided
     */
    private boolean hasBiases;            // B_ARG(0)

    /**
     * indicates whether seqLen array is provided
     */
    private boolean hasSeqLen;            // B_ARG(1)

    /**
     * indicates whether initial output is provided
     */
    private boolean hasInitH ;           // B_ARG(2)

    /**
     * indicates whether initial cell state is provided
     */
    private boolean hasInitC ;          //B_ARG(3)

    /**
     * indicates whether peephole connections are present
     */
    private boolean hasPH;            //B_ARG(4)

    /**
     * indicates whether to return whole time sequence h {h_0, h_1, ... , h_sL-1}
     */
    private boolean retFullSe;            //B_ARG(5)

    /**
     * indicates whether to return output at last time step only,
     * in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
     */

    private boolean retLastH;           //B_ARG(6)
    /**
     * indicates whether to return cells state at last time step only,
     * in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
     */
    private boolean retLastC;            // B_ARG(7)

    /**
     * Cell clipping value, if it = 0 then do not apply clipping
     */
    private double cellClip;   //T_ARG(0)


    public Map<String, Object> toProperties(boolean includeLSTMDataFormat, boolean includeLSTMDirectionMode) {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("gateAct", gateAct.ordinal());
        ret.put("outAct", outAct.ordinal());
        ret.put("cellAct", cellAct.ordinal());
        ret.put("hasBiases", hasBiases);
        ret.put("hasSeqLen", hasSeqLen);
        ret.put("hasInitH", hasInitH);
        ret.put("hasInitC", hasInitC);
        ret.put("hasPH", hasPH);
        ret.put("retFullSe", retFullSe);
        ret.put("retLastH", retLastH);
        ret.put("retLastC", retLastC);
        ret.put("cellClip", cellClip);

        if (includeLSTMDataFormat)
            ret.put("LSTMDataFormat", LSTMdataFormat.ordinal());
        if (includeLSTMDirectionMode)
            ret.put("LSTMDirectionMode", directionMode.ordinal());
        return ret;
    }


    public int[] iArgs(boolean includeLSTMDataFormat) {
        if (includeLSTMDataFormat) {
            return new int[]{ArrayUtil.fromBoolean(hasPH), LSTMdataFormat.ordinal()};
        } else return new int[]{ArrayUtil.fromBoolean(hasPH)};
    }

    public double[] tArgs() {
        return new double[]{cellClip};
    }
}


