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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.List;

/**
 * Log(softmax(X))
 *
 * @author Alex Black
 */

public class LogSoftMax extends BaseTransformOp {
    public LogSoftMax(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public LogSoftMax(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public LogSoftMax(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public LogSoftMax() {
    }

    public LogSoftMax(INDArray x, INDArray z) {
        this(x, null, z);
    }

    public LogSoftMax(INDArray x, INDArray z, long n) {
        this(x, null, z, n);
    }

    public LogSoftMax(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        //ensure the result is the same
        //do a reference check here because it's cheaper
        if (x != z)
            z.assign(x);
    }

    public LogSoftMax(INDArray x, INDArray y, INDArray z) {
        this(x, y, z, x.lengthLong());
    }

    public LogSoftMax(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 40;
    }


    @Override
    public String opName() {
        return "logsoftmax";
    }

    @Override
    public String onnxName() {
        return "LogSoftmax";
    }

    @Override
    public String tensorflowName() {
        return "LogSoftmax";
    }


    @Override
    public void exec() {
        exec(1);
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public void exec(int... dimensions) {
        if (dimensions[0] != 1)
            throw new IllegalArgumentException("Only supports row wise calculations");

        Nd4j.getExecutioner().exec(new OldSoftMax(x, z));
        Transforms.log(z, false);
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret = f().logSoftmaxDerivative(arg(), i_v.get(0));
        return Arrays.asList(ret);
    }
}
