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

package org.nd4j.linalg.api.ops.impl.reduce.same;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceSameOp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.MaxBp;

import java.util.Collections;
import java.util.List;

public class AMax extends BaseReduceSameOp {
    public AMax(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public AMax(INDArray x, INDArray z, boolean keepDims, int[] dimensions) {
        super(x, z, keepDims, dimensions);
    }

    public AMax(INDArray x, INDArray y, INDArray z, int... dimensions) {
        super(x, y, z, dimensions);
    }

    public AMax(SameDiff sameDiff) {
        super(sameDiff);
    }

    public AMax(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public AMax(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public AMax(SameDiff sameDiff, SDVariable i_v, boolean keepDims) {
        super(sameDiff, i_v, keepDims);
    }

    public AMax(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, boolean keepDims) {
        super(sameDiff, i_v, dimensions, keepDims);
    }

    public AMax(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2) {
        super(sameDiff, i_v, i_v2);
    }

    public AMax(SameDiff sameDiff, SDVariable input, int[] dimensions, boolean keepDims) {
        super(sameDiff, input, dimensions, keepDims);
    }

    public AMax(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, boolean keepDims) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims);
    }

    public AMax(SameDiff sameDiff, SDVariable i_v) {
        super(sameDiff, i_v);
    }

    public AMax(INDArray x, INDArray z, int... dimensions) {
        super(x, null, z, dimensions);
    }

    public AMax() {}

    public AMax(INDArray x, INDArray y, INDArray z, boolean keepDims, int[] dimensions) {
        super(x, y, z, keepDims, dimensions);
    }


    public AMax(INDArray x, int... dimensions) {
        super(x, null, null, dimensions);
    }

    public AMax(INDArray in, INDArray dimensions, boolean keepDims) {
        super(in,keepDims,in.toIntVector());
    }

    public AMax(INDArray in, boolean keepDims, int[] dimensions) {
        super(in,keepDims,dimensions);
    }

    public AMax(INDArray in, int[] dimensions, boolean keepDims) {
        super(in,keepDims,dimensions);
    }


    @Override
    public int opNum() {
        return 5;
    }

    @Override
    public String opName() {
        return "amax";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        SDVariable sgn = sameDiff.math().sign(arg());
        SDVariable maxBp = new MaxBp(sameDiff, sameDiff.math().abs(arg()), f1.get(0), false, dimensions).outputVariable();
        return Collections.singletonList(sgn.mul(maxBp));
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }
}
