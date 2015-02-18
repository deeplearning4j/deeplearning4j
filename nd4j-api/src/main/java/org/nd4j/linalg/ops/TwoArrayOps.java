/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.ops;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Builder for two array (possibly plus scalar operations)
 *
 * @author Adam Gibson
 */
public class TwoArrayOps {


    private INDArray from, to, other;
    private Object scalar;
    private Class<? extends BaseTwoArrayElementWiseOp> clazz;


    public TwoArrayOps op(Class<? extends BaseTwoArrayElementWiseOp> clazz) {
        this.clazz = clazz;
        return this;
    }


    public TwoArrayOps other(INDArray other) {

        this.other = other;
        return this;
    }

    public TwoArrayOps from(INDArray from) {

        this.from = from;
        return this;
    }

    public TwoArrayOps to(INDArray to) {

        this.to = to;
        return this;
    }

    public TwoArrayOps scalar(Object scalar) {
        if (scalar instanceof Number) {
            Number n = (Number) scalar;
            this.scalar = n.floatValue();
        } else if (scalar instanceof INDArray) {
            INDArray a = (INDArray) scalar;
            if (!a.isScalar())
                throw new IllegalArgumentException("Only scalar nd arrays allowed");
            Number n = a.getFloat(0);
            this.scalar = n.floatValue();
        } else {
            throw new IllegalArgumentException("Illegal type passed in: Only ndarrays and scalars allowed");
        }


        return this;
    }


    public BaseTwoArrayElementWiseOp build() {
        try {
            BaseTwoArrayElementWiseOp op = clazz.newInstance();
            op.from = from;
            op.to = to;
            op.other = other;
            op.scalarValue = scalar;
            return op;
        } catch (Exception e) {
            throw new RuntimeException(e);

        }
    }


}
