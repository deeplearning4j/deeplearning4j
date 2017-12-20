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

package org.nd4j.linalg.api.ops;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;

/**
 * Base scalar operation
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseScalarOp extends BaseOp implements ScalarOp {

    public BaseScalarOp() {}

    public BaseScalarOp(INDArray x, INDArray y, INDArray z, long n, Number num) {
        super(x, y, z, n);
        this.scalarValue = num;

        init(x, y, z, n);
    }

    public BaseScalarOp(INDArray x, Number num) {
        super(x);
        this.scalarValue = num;
        init(x, y, z, n);

    }





    public BaseScalarOp(SameDiff sameDiff,SDVariable i_v,Number scalar) {
        this(sameDiff,i_v,scalar,false,null);
    }

    public BaseScalarOp(SameDiff sameDiff,SDVariable i_v,Number scalar,boolean inPlace) {
        this(sameDiff,i_v,scalar,inPlace,null);
    }

    public BaseScalarOp(SameDiff sameDiff,
                        SDVariable i_v,
                        Number scalar,
                        boolean inPlace,
                        Object[] extraArgs) {
        super(sameDiff,inPlace,extraArgs);
        this.scalarValue = scalar;
        if (i_v != null) {
            this.xVertexId = i_v.getVarName();
            sameDiff.addArgsFor(new String[]{xVertexId},this);
            this.n = ArrayUtil.prod(i_v.getShape());
            f().validateDifferentialFunctionsameDiff(i_v);
        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }

    }


    public BaseScalarOp(SameDiff sameDiff,
                        SDVariable i_v,
                        Number scalar,
                        Object[] extraArgs) {
        this(sameDiff,i_v,scalar,false,extraArgs);
    }



    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>(1);
        ret.add(arg().getShape());
        return ret;
    }

    @Override
    public Type opType() {
        return Type.SCALAR;
    }

    @Override
    public void setScalar(Number scalar) {
        this.scalarValue = scalar;
    }

    @Override
    public Number scalar() {
        return scalarValue;
    }


    @Override
    public int[] getDimension() {
        return dimensions;
    }

    @Override
    public void setDimension(int... dimension) {
        this.dimensions = dimension;
    }



}
