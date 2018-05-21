/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.NonNull;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.util.List;

/**
 * Absolute sum the components
 *
 * @author raver119@gmail.com
 */
public class MatchConditionTransform extends BaseTransformOp {

    private double compare;
    private double eps;
    private int mode;

    public MatchConditionTransform(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double compare, double eps, int mode) {
        super(sameDiff, i_v, inPlace);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchConditionTransform(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs, double compare, double eps, int mode) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchConditionTransform(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs, double compare, double eps, int mode) {
        super(sameDiff, i_v, extraArgs);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchConditionTransform() {}

    public MatchConditionTransform(@NonNull INDArray x, @NonNull INDArray z, @NonNull Condition condition) {
        this(x, z, Nd4j.EPS_THRESHOLD, condition);
    }


    public MatchConditionTransform(INDArray x, @NonNull Condition condition) {
        this(x, x, Nd4j.EPS_THRESHOLD, condition);
    }


    public MatchConditionTransform(INDArray x, INDArray z, double eps, @NonNull Condition condition) {
        super(x, null, z, z.lengthLong());

        this.compare = condition.getValue();
        this.mode = condition.condtionNum();
        this.eps = eps;

        this.extraArgs = new Object[] {compare, eps, (double) mode};
    }

    public MatchConditionTransform(INDArray x, double eps, @NonNull Condition condition) {
        this(x, x, eps, condition);
    }

    @Override
    public int opNum() {
        return 72;
    }

    @Override
    public String opName() {
        return "match_condition_transform";
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
        return null;
    }
}
