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
import org.nd4j.linalg.api.ops.BaseReduceOp;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Collections;
import java.util.List;

/**
 * Sum of absolute values
 *
 * @author Adam Gibson
 */
public class Norm1 extends BaseReduceOp {
    public Norm1(SameDiff sameDiff, SDVariable i_v, boolean keepDims, int[] dimensions) {
        super(sameDiff, i_v, dimensions, keepDims);
    }

    public Norm1() {
    }

    public Norm1(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Norm1(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public Norm1(INDArray x) {
        super(x);
    }

    public Norm1(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public INDArray noOp() {
        return Transforms.abs(x());
    }


    @Override
    public int opNum() {
        return 5;
    }

    @Override
    public String opName() {
        return "reduce_norm1";
    }

    @Override
    public String onnxName() {
        return "Norm";
    }

    @Override
    public String tensorflowName() {
        return "norm";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        return Collections.singletonList(f().norm1Bp(arg(), grad.get(0), keepDims, dimensions));
    }

    @Override
    public Type getOpType() {
        return Type.REDUCE;
    }
}
