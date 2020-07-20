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

package org.nd4j.linalg.api.ops.impl.transforms.gradient;


import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 *  TanhDerivative: calculated dL/dIn from dL/dOut and In
 */
public class TanhDerivative extends DynamicCustomOp {
    public TanhDerivative(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, new SDVariable[]{i_v1, i_v2});
    }

    /**
     *
     * @param x Input
     * @param y Gradient at output (dL/dOut)
     * @param z Output array, gradient at input (dL/dIn - to be calculated)
     */
    public TanhDerivative(INDArray x, INDArray y, INDArray z) {
        super(null, new INDArray[]{x, y}, new INDArray[]{z});
    }

    public TanhDerivative() {
    }

    /**
     * @param x Input
     * @param y Gradient at output (dL/dOut)
     */
    public TanhDerivative(INDArray x, INDArray y) {
        this(x, y, null);
    }

    @Override
    public int opNum() {
        return 0;
    }

    /**
     * The opName of this operation
     *
     * @return the opName of this operation
     */
    @Override
    public String opName() {
        return "tanh_bp";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " + opName());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret = sameDiff.math.div(sameDiff.onesLike(outputVariables()[0]), sameDiff.math.pow(sameDiff.math.cosh(arg()), 2));
        return Collections.singletonList(ret);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
