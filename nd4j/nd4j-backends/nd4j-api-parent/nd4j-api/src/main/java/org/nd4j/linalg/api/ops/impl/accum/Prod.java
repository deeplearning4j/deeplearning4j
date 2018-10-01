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

package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceOp;

import java.util.Collections;
import java.util.List;

/**
 * Prod the components
 *
 * @author Adam Gibson
 */
public class Prod extends BaseReduceOp {
    public Prod(SameDiff sameDiff, SDVariable i_v, boolean keepDims, int[] dimensions) {
        super(sameDiff, i_v, dimensions, keepDims);
    }

    public Prod(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public Prod() {
    }

    public Prod(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Prod(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public Prod(INDArray x) {
        super(x);
    }

    public Prod(INDArray x, INDArray y) {
        super(x, y);
    }

    public Prod(INDArray x, INDArray y, INDArray z, boolean newFormat, boolean keepDims, int[] dimensions) {
        super(x, y, z, newFormat, keepDims, dimensions);
    }


    @Override
    public int opNum() {
        return 8;
    }

    @Override
    public String opName() {
        return "reduce_prod";
    }


    @Override
    public double zeroDouble() {
        return 1.0;
    }

    @Override
    public float zeroFloat() {
        return 1.0f;
    }

    @Override
    public float zeroHalf() {
        return zeroFloat();
    }

    @Override
    public String onnxName() {
        return "ReduceProd";
    }

    @Override
    public String tensorflowName() {
        return "Prod";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        return Collections.singletonList(f().prodBp(arg(), grad.get(0), keepDims, dimensions));
    }

    @Override
    public Type getOpType() {
        return Type.REDUCE;
    }
}
