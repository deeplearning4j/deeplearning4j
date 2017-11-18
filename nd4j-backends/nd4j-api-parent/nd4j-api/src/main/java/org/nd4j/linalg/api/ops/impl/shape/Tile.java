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

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ShapeOp;

import java.util.List;

/**
 * Tile function
 *
 * @author Adam Gibson
 */
public class Tile extends ShapeOp {
   private int[] axis;

    public Tile(SameDiff sameDiff, DifferentialFunction i_v, int[] axis) {
        super(sameDiff, i_v, false);
        this.axis = axis;
    }

    public Tile() {}

    public Tile(INDArray x, INDArray z) {
        super(x, z);
    }

    public Tile(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Tile(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Tile(INDArray x) {
        super(x);
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
        int[] permuteDims = extraArgs == null ? z().shape() : (int[]) extraArgs[0];
        if(x != z) {
            if(x.isScalar() && !z.isScalar()) {
                z.assign(x.getDouble(0));
            }
            else
                z.assign(x.broadcast(permuteDims));
        }
        else {
            if(x.isScalar() && !z.isScalar()) {
                z.assign(x.getDouble(0));
            }
            else
                this.z = x.broadcast(permuteDims);
        }

    }


    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "tile";
    }

    @Override
    public String onnxName() {
        return "Tile";
    }

    @Override
    public String tensorflowName() {
        return "tile";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        f().validateDifferentialFunctionsameDiff(i_v);
        throw new UnsupportedOperationException();
    }

}
