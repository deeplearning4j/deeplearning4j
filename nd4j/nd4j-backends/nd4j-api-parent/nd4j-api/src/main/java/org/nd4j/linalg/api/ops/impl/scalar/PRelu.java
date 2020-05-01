/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.linalg.api.ops.impl.scalar;

import java.util.Collections;
import java.util.List;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.PReluBp;

/**
 * Parameterized ReLU op
 */
@NoArgsConstructor
public class PRelu extends DynamicCustomOp {

    @Getter
    protected int[] sharedAxes;

    public PRelu(@NonNull SameDiff sameDiff, @NonNull SDVariable x, @NonNull SDVariable alpha, @NonNull int... sharedAxes) {
        super(sameDiff, new SDVariable[]{x, alpha});
        this.sharedAxes = sharedAxes;
        addIArgument(sharedAxes);
    }

    public PRelu(@NonNull INDArray x, @NonNull INDArray alpha, @NonNull int... sharedAxes) {
        this(x, null, alpha, sharedAxes);
    }

    public PRelu(@NonNull INDArray x, INDArray z, @NonNull INDArray alpha, @NonNull int... sharedAxes) {
        super(new INDArray[]{x, alpha}, new INDArray[]{z});
        this.sharedAxes = sharedAxes;
        addIArgument(sharedAxes);
    }

    @Override
    public String opName() {
        return "prelu";
    }

    @Override
    public String onnxName() { throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions
                .checkArgument(dataTypes != null && dataTypes.size() == 2, "Expected exactly 2 input datatypes, got %s", dataTypes);
        Preconditions.checkArgument(dataTypes.get(0).isFPType() && dataTypes.get(1).isFPType(), "Input datatypes must be floating point, got %s", dataTypes);

        return Collections.singletonList(dataTypes.get(0));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return new PReluBp(sameDiff, arg(0), arg(1), i_v.get(0), sharedAxes).outputs();
    }
}
