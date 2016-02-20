/*
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

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Base scalar operation
 *
 * @author Adam Gibson
 */
public abstract class BaseScalarOp extends BaseOp implements ScalarOp {
    protected Number num;
    protected IComplexNumber complexNumber;


    public BaseScalarOp() {
    }

    public BaseScalarOp(INDArray x, INDArray y, INDArray z, int n, Number num) {
        super(x, y, z, n);
        this.num = num;
        if(x instanceof IComplexNDArray)
            complexNumber = Nd4j.createComplexNumber(num,0);

        init(x, y, z, n);
    }

    public BaseScalarOp(INDArray x, Number num) {
        super(x);
        this.num = num;
        if(x instanceof IComplexNDArray)
            complexNumber = Nd4j.createComplexNumber(num,0);

        init(x, y, z, n);

    }

    public BaseScalarOp(INDArray x, INDArray y, INDArray z, int n, IComplexNumber num) {
        super(x, y, z, n);
        this.complexNumber = num;
        init(x, y, z, n);

    }

    public BaseScalarOp(INDArray x, IComplexNumber num) {
        super(x);
        this.complexNumber = num;
        init(x, y, z, n);

    }

    @Override
    public int broadcastLength() {
        return 1;
    }

    @Override
    public int[] broadcastShape() {
        return new int[]{1,1};
    }

    @Override
    public Number scalar() {
        return num;
    }

    @Override
    public IComplexNumber complexScalar() {
        return complexNumber;
    }

    @Override
    public int[] getDimension() {
        return new int[0];
    }

    @Override
    public void setDimension(int... dimension) {

    }
}
