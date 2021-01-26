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
package org.nd4j.linalg.api.ops.impl.layers.recurrent.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.LinkedHashMap;
import java.util.Map;


@Builder
@Data
@AllArgsConstructor
@NoArgsConstructor
public class LSTMLayerConfig {

    /**
     * notations <br>
     * for unidirectional:
     * TNS: shape [timeLength, numExamples, inOutSize] - sometimes referred to as "time major"<br>
     * NST: shape [numExamples, inOutSize, timeLength]<br>
     * NTS: shape [numExamples, timeLength, inOutSize] - TF "time_major=false" layout<br>
     * for bidirectional:
     * T2NS: 3 = [timeLength, 2, numExamples, inOutSize] (for ONNX)
     */
    @Builder.Default
    private LSTMDataFormat lstmdataformat = LSTMDataFormat.TNS;  //INT_ARG(0)


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
    private LSTMActivations gateAct = LSTMActivations.SIGMOID; // INT_ARG(2)

    @Builder.Default
    private LSTMActivations cellAct = LSTMActivations.TANH; // INT_ARG(3)

    @Builder.Default
    private LSTMActivations outAct = LSTMActivations.TANH; // INT_ARG(4)




    /**
     * indicates whether to return whole time sequence h {h_0, h_1, ... , h_sL-1}
     */
    @Builder.Default
    private boolean retFullSequence = true;            //B_ARG(5)

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
    @Builder.Default
    private double cellClip = 0;   //T_ARG(0)


    public Map<String, Object> toProperties(boolean includeLSTMDataFormat, boolean includeLSTMDirectionMode) {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("gateAct", gateAct.toString());
        ret.put("outAct", outAct.toString());
        ret.put("cellAct", cellAct.toString());
        ret.put("retFullSequence", retFullSequence);
        ret.put("retLastH", retLastH);
        ret.put("retLastC", retLastC);
        ret.put("cellClip", cellClip);

        if (includeLSTMDataFormat)
            ret.put("lstmdataformat", lstmdataformat.toString());
        if (includeLSTMDirectionMode)
            ret.put("directionMode", directionMode.toString());
        return ret;
    }

}






