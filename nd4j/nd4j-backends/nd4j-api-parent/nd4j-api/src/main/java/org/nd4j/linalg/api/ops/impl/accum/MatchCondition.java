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

package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Absolute sum the components
 *
 * @author raver119@gmail.com
 */
public class MatchCondition extends BaseAccumulation {

    private double compare;
    private double eps;
    private int mode;

    public MatchCondition(SameDiff sameDiff, SDVariable in, Condition condition) {
        super(sameDiff, in, null);
        this.compare = condition.getValue();
        this.mode = condition.condtionNum();
        this.eps = Nd4j.EPS_THRESHOLD;

        this.extraArgs = new Object[] {compare, eps, (double) mode};
    }

    public MatchCondition() {}


    public MatchCondition(INDArray x, Condition condition) {
        this(x, Nd4j.EPS_THRESHOLD, condition);
    }

    public MatchCondition(INDArray x, double eps, Condition condition) {
        super(x);
        this.compare = condition.getValue();
        this.mode = condition.condtionNum();
        this.eps = eps;

        this.extraArgs = new Object[] {compare, eps, (double) mode};
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("compare",compare);
        ret.put("eps",eps);
        ret.put("mode",mode);
        return ret;
    }


    @Override
    public int opNum() {
        return 12;
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
        return null;
    }

    @Override
    public Type getOpType() {
        return Type.REDUCE;
    }
}
