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

package org.nd4j.linalg.api.ops.impl.transforms.arithmetic;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 *  Level 1 blas op Axpy as libnd4j native op
 *
 * @author raver119@gmail.com
 */
public class Axpy extends BaseTransformOp {

    private double p;

    public Axpy(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, double p) {
        super(sameDiff, i_v1, i_v2);
        this.p = p;
    }

    public Axpy(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace, double p) {
        super(sameDiff, i_v1, i_v2, inPlace);
        this.p = p;
    }

    public Axpy(SameDiff sameDiff, double p) {
        super(sameDiff);
        this.p = p;
    }

    public Axpy(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, Object[] extraArgs, double p) {
        super(sameDiff, i_v1, i_v2, extraArgs);
        this.p = p;
    }

    public Axpy(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double p) {
        super(sameDiff, i_v, inPlace);
        this.p = p;
    }

    public Axpy(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs, double p) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
        this.p = p;
    }

    public Axpy(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs, double p) {
        super(sameDiff, i_v, extraArgs);
        this.p = p;
    }

    public Axpy() {

    }

    public Axpy(INDArray x, INDArray z, double p) {
        //      super(x, z, z, z.lengthLong());
        this.p = p;
        init(x, z, z, x.length());
    }

    public Axpy(INDArray x, INDArray z, double p, long n) {
        //        super(x, z, n);
        this.p = p;
        init(x, z, z, n);
    }

    public Axpy(INDArray x, INDArray y, INDArray z, double p, long n) {
        //        super(x,y,z,n);
        this.p = p;
        init(x, y, z, x.length());
    }

    @Override
    public int opNum() {
        return 17;
    }

    @Override
    public String opName() {
        return "axpy";
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
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("p",p);
        return ret;
    }



    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);

        if (x.lengthLong() < n || y.lengthLong() < n || z.lengthLong() < n)
            throw new IllegalStateException("Mis matched lengths: X: [" + x.lengthLong() + "], Y: [" + y.lengthLong()
                            + "], Z: [" + z.lengthLong() + "], N: [" + n + "]");

        this.extraArgs = new Object[] {p, (double) n};
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }
}
