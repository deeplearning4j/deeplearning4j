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

package org.nd4j.linalg.api.ops.impl.reduce.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.SumBp;

import java.util.Collections;
import java.util.List;

/**
 * LogSumExp - this op returns https://en.wikipedia.org/wiki/LogSumExp
 *
 * @author raver119@gmail.com
 */
public class LogSumExp extends DynamicCustomOp {

    protected boolean keepDims;

    public LogSumExp(SameDiff sameDiff, SDVariable i_v, boolean keepDims, int[] dimensions) {
        super(sameDiff, i_v);
        if(dimensions != null) {
            addIArgument(dimensions);
            this.dimensions = dimensions;
        }
        addTArgument(keepDims ? 1.0 : 0.0);
        this.keepDims = keepDims;
    }

    public LogSumExp(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        this(sameDiff, i_v, false, dimensions);
    }

    public LogSumExp() {}

    public LogSumExp(INDArray x, int... dimensions) {
        this(x, false, dimensions);
    }

    public LogSumExp(INDArray x, boolean keepDim, int... dimensions) {
        this(x, null, keepDim, dimensions);
    }

    public LogSumExp(INDArray x, INDArray z, boolean keepDim, int... dimensions) {
        super(null, x,z, Collections.singletonList(keepDim ? 1.0 : 0.0), dimensions);
    }

    @Override
    public String opName() {
        return "reduce_logsumexp";
    }


    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 1 || dataTypes.size() == 2),
                "Expected 1 or 2 input datatypes for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //z = log(sum_i exp(x_i)) = log(s)
        //dL/dx = dL/dz * dz/ds * ds/dx
        //dz/ds = 1/s
        SDVariable exp = sameDiff.math.exp(arg());
        SDVariable sumExp = exp.sum(dimensions);
        SDVariable gradProd = f1.get(0).div(sumExp);
        SDVariable dSumExpdx = new SumBp(sameDiff, arg(), gradProd, keepDims, dimensions).outputVariable().mul(exp);
        return Collections.singletonList(dSumExpdx);
    }

    @Override
    public String onnxName() {
        return "ReduceLogSumExp";
    }
}
