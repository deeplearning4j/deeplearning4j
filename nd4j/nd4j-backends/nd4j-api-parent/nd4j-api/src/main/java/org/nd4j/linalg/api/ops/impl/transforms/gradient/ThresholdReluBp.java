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

package org.nd4j.linalg.api.ops.impl.transforms.gradient;

import java.util.Collections;
import java.util.List;
import lombok.Getter;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ThresholdRelu;

/**
 * Threshold ReLU Backprop op - dL/dIn from in and dL/dOut
 *
 * For {@link RectifiedLinear} as well as {@link ThresholdRelu}.
 */
public class ThresholdReluBp extends DynamicCustomOp {

    @Getter
    private double cutoff = 0;

    public ThresholdReluBp(){ }

    public ThresholdReluBp(SameDiff sd, SDVariable input, SDVariable gradient, double cutoff){
        super(sd, new SDVariable[]{input, gradient});
        this.cutoff = cutoff;
        addTArgument(cutoff);
    }

    public ThresholdReluBp(@NonNull INDArray input, @NonNull INDArray gradient, INDArray output, double cutoff){
        super(new INDArray[]{input, gradient}, wrapOrNull(output));
        this.cutoff = cutoff;
        addTArgument(cutoff);
    }

    @Override
    public String opName(){
        return "thresholdedrelu_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions
                .checkArgument(dataTypes != null && dataTypes.size() == 2, "Expected exactly 2 input datatypes, got %s", dataTypes);
        Preconditions.checkArgument(dataTypes.get(0).isFPType() && dataTypes.get(1).isFPType(), "Input datatypes must be floating point, got %s", dataTypes);

        return Collections.singletonList(dataTypes.get(0));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Not supported");
    }
}
