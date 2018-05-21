/*-
 *
 * * Copyright 2015 Skymind,Inc. * * Licensed under the Apache License, Version 2.0 (the "License"); * you may not use
 * this file except in compliance with the License. * You may obtain a copy of the License at * *
 * http://www.apache.org/licenses/LICENSE-2.0 * * Unless required by applicable law or agreed to in writing, software *
 * distributed under the License is distributed on an "AS IS" BASIS, * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied. * See the License for the specific language governing permissions and * limitations under
 * the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.transforms.arithmetic;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.List;

/**
 * Floor mod
 *
 * @author raver119@gmail.com
 */
public class FloorModOp extends BaseTransformOp {
    public FloorModOp() {}

    public FloorModOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public FloorModOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public FloorModOp(SameDiff sameDiff) {
        super(sameDiff);
    }

    public FloorModOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, extraArgs);
    }

    public FloorModOp(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public FloorModOp(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public FloorModOp(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public FloorModOp(INDArray x, INDArray z) {
        super(x, z);
    }

    public FloorModOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public FloorModOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public FloorModOp(INDArray x) {
        super(x);
    }


    @Override
    public int opNum() {
        return 21;
    }

    @Override
    public String opName() {
        return "floormod";
    }

    @Override
    public String onnxName() {
        return "FloorMod";
    }

    @Override
    public String tensorflowName() {
        return "FloorMod";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return f().floorModBp(larg(), rarg(), f1.get(0));
    }
}
