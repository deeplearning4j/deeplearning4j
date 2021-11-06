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

package org.nd4j.linalg.api.ops.impl.reduce.longer;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceLongOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.util.Collections;
import java.util.List;

public class MatchCondition extends BaseReduceLongOp {

    private double compare;
    private double eps;
    private int mode;

    public MatchCondition(SameDiff sameDiff, SDVariable in, Condition condition) {
        this(sameDiff, in, condition, false, null);
    }

    public MatchCondition(SameDiff sameDiff, SDVariable in, Condition condition, boolean keepDims, int... dimensions) {
        super(sameDiff, in, dimensions, keepDims);
        this.compare = condition.getValue();
        this.mode = condition.conditionNum();
        this.eps = Nd4j.EPS_THRESHOLD;
        this.extraArgs = new Object[] {compare, eps, (double) mode};
    }

    public MatchCondition() {}

    public MatchCondition(INDArray x, Condition condition, int... dimensions) {
        this(x, Nd4j.EPS_THRESHOLD, condition, dimensions);
    }

    public MatchCondition(INDArray x, Condition condition, boolean keepDims, int... dimensions) {
        this(x, Nd4j.EPS_THRESHOLD, condition, dimensions);
        this.keepDims = keepDims;
    }

    public MatchCondition(INDArray x, double eps, Condition condition, int... dimensions) {
        super(x);
        this.compare = condition.getValue();
        this.mode = condition.conditionNum();
        this.eps = eps;

        this.extraArgs = new Object[] {compare, eps, (double) mode};

        defineDimensions(dimensions);
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, double compare) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.compare = compare;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, boolean keepDims, double compare) {
        super(sameDiff, i_v, keepDims);
        this.compare = compare;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, boolean keepDims, double compare) {
        super(sameDiff, i_v, dimensions, keepDims);
        this.compare = compare;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, double compare) {
        super(sameDiff, i_v, i_v2);
        this.compare = compare;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable input, int[] dimensions, boolean keepDims, double compare) {
        super(sameDiff, input, dimensions, keepDims);
        this.compare = compare;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, boolean keepDims, double compare) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims);
        this.compare = compare;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, double compare) {
        super(sameDiff, i_v);
        this.compare = compare;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable input, double compare, int... dimensions) {
        super(sameDiff, input, dimensions);
        this.compare = compare;
    }

    public MatchCondition(INDArray x, double compare, int... dimensions) {
        super(x, dimensions);
        this.compare = compare;
    }

    public MatchCondition(INDArray x, boolean keepDims, double compare, int... dimensions) {
        super(x, keepDims, dimensions);
        this.compare = compare;
    }

    public MatchCondition(INDArray x, INDArray z, double compare, int... dimensions) {
        super(x, z, dimensions);
        this.compare = compare;
    }

    public MatchCondition(INDArray x, INDArray y, INDArray z, double compare, int... dimensions) {
        super(x, y, z, dimensions);
        this.compare = compare;
    }

    public MatchCondition(SameDiff sameDiff, double compare) {
        super(sameDiff);
        this.compare = compare;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions, double compare) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.compare = compare;
    }

    public MatchCondition(double compare) {
        this.compare = compare;
    }

    public MatchCondition(INDArray x, INDArray y, INDArray z, boolean keepDims, int[] dimensions, double compare) {
        super(x, y, z, keepDims, dimensions);
        this.compare = compare;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, double compare, double eps) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, boolean keepDims, double compare, double eps) {
        super(sameDiff, i_v, keepDims);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, boolean keepDims, double compare, double eps) {
        super(sameDiff, i_v, dimensions, keepDims);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, double compare, double eps) {
        super(sameDiff, i_v, i_v2);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable input, int[] dimensions, boolean keepDims, double compare, double eps) {
        super(sameDiff, input, dimensions, keepDims);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, boolean keepDims, double compare, double eps) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, double compare, double eps) {
        super(sameDiff, i_v);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable input, double compare, double eps, int... dimensions) {
        super(sameDiff, input, dimensions);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(INDArray x, double compare, double eps, int... dimensions) {
        super(x, dimensions);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(INDArray x, boolean keepDims, double compare, double eps, int... dimensions) {
        super(x, keepDims, dimensions);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(INDArray x, INDArray z, double compare, double eps, int... dimensions) {
        super(x, z, dimensions);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(INDArray x, INDArray y, INDArray z, double compare, double eps, int... dimensions) {
        super(x, y, z, dimensions);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(SameDiff sameDiff, double compare, double eps) {
        super(sameDiff);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions, double compare, double eps) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(double compare, double eps) {
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(INDArray x, INDArray y, INDArray z, boolean keepDims, int[] dimensions, double compare, double eps) {
        super(x, y, z, keepDims, dimensions);
        this.compare = compare;
        this.eps = eps;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, double compare, double eps, int mode) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, boolean keepDims, double compare, double eps, int mode) {
        super(sameDiff, i_v, keepDims);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, boolean keepDims, double compare, double eps, int mode) {
        super(sameDiff, i_v, dimensions, keepDims);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, double compare, double eps, int mode) {
        super(sameDiff, i_v, i_v2);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable input, int[] dimensions, boolean keepDims, double compare, double eps, int mode) {
        super(sameDiff, input, dimensions, keepDims);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, boolean keepDims, double compare, double eps, int mode) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, double compare, double eps, int mode) {
        super(sameDiff, i_v);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable input, double compare, double eps, int mode, int... dimensions) {
        super(sameDiff, input, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(INDArray x, double compare, double eps, int mode, int... dimensions) {
        super(x, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(INDArray x, boolean keepDims, double compare, double eps, int mode, int... dimensions) {
        super(x, keepDims, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(INDArray x, INDArray z, double compare, double eps, int mode, int... dimensions) {
        super(x, z, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(INDArray x, INDArray y, INDArray z, double compare, double eps, int mode, int... dimensions) {
        super(x, y, z, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, double compare, double eps, int mode) {
        super(sameDiff);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions, double compare, double eps, int mode) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(double compare, double eps, int mode) {
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(INDArray x, INDArray y, INDArray z, boolean keepDims, int[] dimensions, double compare, double eps, int mode) {
        super(x, y, z, keepDims, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    @Override
    public int opNum() {
        return 2;
    }

    @Override
    public String opName() {
        return "match_condition";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }
}
