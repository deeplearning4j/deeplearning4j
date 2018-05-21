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

package org.nd4j.linalg.api.ops.impl.shape;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ShapeOp;

import java.util.*;

/**
 * Transpose function
 *
 * @author Adam Gibson
 */
public class RollAxis extends ShapeOp {
    private int axis;

    public RollAxis(SameDiff sameDiff, int axis) {
        super(sameDiff);
        this.axis = axis;
    }

    public RollAxis(SameDiff sameDiff, SDVariable i_v, int axis) {
        super(sameDiff, i_v, false);
        this.axis = axis;
    }

    public RollAxis(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs, int axis) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
        this.axis = axis;
    }

    public RollAxis() {
    }

    public RollAxis(INDArray x, INDArray z) {
        super(x, z);
    }

    public RollAxis(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public RollAxis(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public RollAxis(INDArray x) {
        super(x);
    }


    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("axis", axis);
        return ret;
    }


    @Override
    public void exec(int... dimensions) {
        exec();
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public void exec() {
        if (x != z) {
            z.assign(x.transpose());
        } else {
            this.z = x.transpose();
        }

    }

    @Override
    public List<long[]> calculateOutputShape() {
        List<long[]> ret = new ArrayList<>();
        long[] inputShape = arg().getShape();
        long[] outputShape = new long[inputShape.length];
        outputShape[0] = inputShape[axis];
        for(int i = 1; i <=axis; ++i) {
            outputShape[i] = inputShape[i - 1];
        }
        for(int i = axis + 1; i < inputShape.length; ++i) {
            outputShape[i] = inputShape[i];
        }
        ret.add(outputShape);
        return ret;
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "rollaxis";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow opName found for " + opName());
    }


    @Override
    public INDArray z() {
        return x().transpose();
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret = outputVariables()[0];
        return Arrays.asList(ret);
    }

}
