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

package org.nd4j.linalg.api.ops.impl.accum.floating;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceFloatOp;
import org.nd4j.linalg.api.ops.impl.accum.same.Sum;

import java.util.Collections;
import java.util.List;

/**
 * Calculate the mean of the vector
 *
 * @author Adam Gibson
 */
public class Mean extends BaseReduceFloatOp {
    public Mean(SameDiff sameDiff, SDVariable i_v, boolean keepDims, int[] dimensions) {
        super(sameDiff, i_v, keepDims, dimensions);
    }

    public Mean() {
    }

    public Mean(INDArray x, INDArray z, long n) {
        super(x, null, z, n);
    }

    public Mean(INDArray x) {
        super(x);
    }

    public Mean(INDArray x, INDArray z) {
        super(x, null, z, x.lengthLong());
    }

    public Mean(INDArray x, INDArray z, boolean newFormat, boolean keepDims, int[] dimensions) {
        super(x, z, newFormat, keepDims, dimensions);
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "reduce_mean";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        if(!newFormat)
            throw new IllegalStateException("Cannot doDiff with newFormat == false");
        //If out = mean(in), then dL/dIn = 1/N * dL/dOut  (broadcast to appropriate shape)
        //Note that N differs for "along dimension" vs. "whole array" reduce cases
        return Collections.singletonList(f().meanBp(arg(), i_v1.get(0), keepDims, dimensions));
    }

    @Override
    public String onnxName() {
        return "ReduceMean";
    }

    @Override
    public String tensorflowName() {
        return "Mean";
    }
}
