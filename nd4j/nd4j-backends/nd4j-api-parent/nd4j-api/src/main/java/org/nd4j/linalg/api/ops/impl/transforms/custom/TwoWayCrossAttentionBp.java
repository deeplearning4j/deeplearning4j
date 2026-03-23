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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * Backward pass for SAM-style Two-Way Cross Attention.
 *
 * Inputs:
 *   0-5: same as forward (tokenQ, tokenK, tokenV, imageQ, imageK, imageV)
 *   6: gradTokenOutput (upstream gradient for token output)
 *   7: gradImageOutput (upstream gradient for image output)
 *
 * Float args:
 *   0: scale factor (default: 1/sqrt(embedDim))
 *
 * Outputs:
 *   0: dL/dTokenQuery
 *   1: dL/dTokenKey
 *   2: dL/dTokenValue
 *   3: dL/dImageQuery
 *   4: dL/dImageKey
 *   5: dL/dImageValue
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class TwoWayCrossAttentionBp extends DynamicCustomOp {

    private double scale = 0.0;

    public TwoWayCrossAttentionBp(SameDiff sameDiff,
                                   SDVariable tokenQuery, SDVariable tokenKey, SDVariable tokenValue,
                                   SDVariable imageQuery, SDVariable imageKey, SDVariable imageValue,
                                   SDVariable gradTokenOut, SDVariable gradImageOut) {
        super(null, sameDiff, new SDVariable[]{tokenQuery, tokenKey, tokenValue,
                imageQuery, imageKey, imageValue, gradTokenOut, gradImageOut}, false);
    }

    public TwoWayCrossAttentionBp(SameDiff sameDiff,
                                   SDVariable tokenQuery, SDVariable tokenKey, SDVariable tokenValue,
                                   SDVariable imageQuery, SDVariable imageKey, SDVariable imageValue,
                                   SDVariable gradTokenOut, SDVariable gradImageOut,
                                   double scale) {
        super(null, sameDiff, new SDVariable[]{tokenQuery, tokenKey, tokenValue,
                imageQuery, imageKey, imageValue, gradTokenOut, gradImageOut}, false);
        this.scale = scale;
        if (scale > 0) addTArgument(scale);
    }

    @Override
    public String opName() {
        return "two_way_cross_attention_bp";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException("Differentiation of " + getClass().getName() + " not supported");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        DataType dt = inputDataTypes.get(0);
        return Arrays.asList(dt, dt, dt, dt, dt, dt);
    }

    @Override
    public int getNumOutputs() {
        return 6;
    }
}
