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

package org.nd4j.linalg.api.ops.impl.indexaccum;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseIndexAccumulation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Calculate the index
 * of max value over a vector
 *
 * @author raver119@gmail.com
 */
public class LastIndex extends BaseIndexAccumulation {
    protected Condition condition;
    protected double compare;
    protected double eps;
    protected int mode;

    public LastIndex(SameDiff sameDiff, SDVariable i_v, int[] dimensions, Condition condition, double compare, double eps, int mode) {
        super(sameDiff, i_v, dimensions);
        this.condition = condition;
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public LastIndex(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, Condition condition, double compare, double eps, int mode) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.condition = condition;
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public LastIndex() {}


    public LastIndex(INDArray x, @NonNull Condition condition) {
        this(x, condition, Nd4j.EPS_THRESHOLD);
    }

    public LastIndex(INDArray x, @NonNull Condition condition, double eps) {
        super(x,null,null,x.length());

        this.condition = condition;
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
        return 5;
    }

    @Override
    public String opName() {
        return "last_index";
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
    public float zeroHalf() {
        return 0;
    }
}
