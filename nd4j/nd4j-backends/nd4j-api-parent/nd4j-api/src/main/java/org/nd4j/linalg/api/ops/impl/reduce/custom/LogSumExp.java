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

package org.nd4j.linalg.api.ops.impl.reduce.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

public class LogSumExp extends DynamicCustomOp {

    protected boolean keepDims;

    public LogSumExp(SameDiff sameDiff, SDVariable i_v, boolean keepDims, long[] dimensions) {
        super(sameDiff, i_v);
        if(dimensions != null && dimensions.length > 0) {
            // -1 or Integer.MAX_VALUE means "full array" - don't pass dimension args for full array reduction
            if(dimensions.length != 1 || (dimensions[0] != -1 && dimensions[0] != Integer.MAX_VALUE)) {
                addIArgument(dimensions);
            }
            this.dimensions = dimensions;
        }
        addTArgument(keepDims ? 1.0 : 0.0);
        this.keepDims = keepDims;
    }

    public LogSumExp(SameDiff sameDiff, SDVariable i_v, long[] dimensions) {
        this(sameDiff, i_v, false, dimensions);
    }

    public LogSumExp() {}

    public LogSumExp(INDArray x, long... dimensions) {
        this(x, false, dimensions);
    }

    public LogSumExp(INDArray x, boolean keepDim, long... dimensions) {
        this(x, null, keepDim, dimensions);
    }

    public LogSumExp(INDArray x, INDArray z, boolean keepDim, long... dimensions) {
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
        // d/dx logsumexp(x) = softmax(x)
        // dL/dx = dL/dz * softmax(x)  where z = logsumexp(x)
        SDVariable softmax;
        if(dimensions == null || dimensions.length == 0) {
            if(args().length >= 2) {
                // Dynamic dimensions from second input
                softmax = sameDiff.nn().softmax(arg());
            } else {
                // Full reduction: softmax over flattened array
                // Reshape to 1D, apply softmax, reshape back
                SDVariable flatInput = sameDiff.reshape(arg(), -1);
                SDVariable flatSoftmax = sameDiff.nn().softmax(flatInput);
                softmax = sameDiff.reshape(flatSoftmax, sameDiff.shape(arg()));
            }
        } else {
            // Reduce along specific dimensions - apply softmax along those dimensions
            // For single dimension, use softmax directly
            if(dimensions.length == 1) {
                softmax = sameDiff.nn().softmax(arg(), (int) dimensions[0]);
            } else {
                // Multi-dimension: flatten those dims, softmax, unflatten
                SDVariable flatInput = sameDiff.reshape(arg(), -1);
                SDVariable flatSoftmax = sameDiff.nn().softmax(flatInput);
                softmax = sameDiff.reshape(flatSoftmax, sameDiff.shape(arg()));
            }
        }

        return Collections.singletonList(f1.get(0).mul(softmax));
    }

    @Override
    public String onnxName() {
        return "ReduceLogSumExp";
    }
}
