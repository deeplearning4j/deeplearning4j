/*
 *  ******************************************************************************
 *  *
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

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlockCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer;
import org.nd4j.common.util.ArrayUtil;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * LSTM Configuration - for {@link LSTMLayer} and {@link LSTMBlockCell}
 *
 * @author Alex Black
 */
@Builder
@Data
public class LSTMConfiguration {

    /**
     * Whether to provide peephole connections.
     */
    private boolean peepHole;           //IArg(0)

    /**
     * The data format of the input.  Only used in {@link LSTMLayer}, ignored in {@link LSTMBlockCell}.
     */
    @Builder.Default private RnnDataFormat dataFormat = RnnDataFormat.TNS;  //IArg(1) (only for lstmBlock, not lstmBlockCell)

    /**
     * The bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training.
     */
    private double forgetBias;          //TArg(0)

    /**
     * Clipping value for cell state, if it is not equal to zero, then cell state is clipped.
     */
    private double clippingCellValue;   //TArg(1)

    public Map<String,Object> toProperties(boolean includeDataFormat)  {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("peepHole",peepHole);
        ret.put("clippingCellValue",clippingCellValue);
        ret.put("forgetBias",forgetBias);
        if(includeDataFormat)
            ret.put("dataFormat", dataFormat);
        return ret;
    }


    public int[] iArgs(boolean includeDataFormat) {
        if(includeDataFormat) {
            return new int[]{ArrayUtil.fromBoolean(peepHole), dataFormat.ordinal()};
        } else return new int[]{ArrayUtil.fromBoolean(peepHole)};
    }

    public double[] tArgs() {
        return new double[] {forgetBias,clippingCellValue};
    }
}
