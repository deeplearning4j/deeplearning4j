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

package org.nd4j.linalg.api.ops.impl.transforms.strict;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.EluBp;

import java.util.List;

/**
 * ELU: Exponential Linear Unit (alpha=1.0)<br>
 * Introduced in paper:<br>
 * Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)<br>
 * Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter (2015)<br>
 * <a href="https://arxiv.org/abs/1511.07289">https://arxiv.org/abs/1511.07289</a>
 *
 * @author Alex Black
 */
public class ELU extends DynamicCustomOp {
    public static final double DEFAULT_ALPHA = 1.0;

    protected double alpha;

    public ELU(SameDiff sameDiff, SDVariable i_v) {
        super(sameDiff, new SDVariable[]{i_v});
        this.alpha = DEFAULT_ALPHA;
        addTArgument(alpha);
    }

    public ELU() {
    }

    public ELU(INDArray x, INDArray z) {
        this(x, z, DEFAULT_ALPHA);
    }

    public ELU(INDArray x, INDArray z, double alpha) {
        super(null, wrapOrNull(x), wrapOrNull(z));
        this.alpha = alpha;
        addTArgument(alpha);
    }

    public ELU(INDArray x) {
        this(x, null, DEFAULT_ALPHA);
    }

    @Override
    public String opName() {
        return "elu";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "Elu";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        //ELU: e^x-1 if x<0, x otherwise
        //dL/dIn = dL/Out * dOut/dIn
        return new EluBp(sameDiff, arg(), i_v.get(0), alpha).outputs();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 datatype for ELU, got %s", dataTypes);
        Preconditions.checkState(dataTypes.get(0).isFPType(), "Expected floating point input type for ELU, got %s", dataTypes);

        return dataTypes;
    }
}
