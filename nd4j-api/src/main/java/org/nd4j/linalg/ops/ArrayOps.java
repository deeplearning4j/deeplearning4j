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
import org.nd4j.linalg.ops.factory.ElementWiseOpFactory;

/**
 * Builder for element wise operations
 *
 * @author Adam Gibson
 */
public class ArrayOps {

    private INDArray from, scalar;
    private ElementWiseOpFactory clazz;
    private Object[] extraArgs;


    /**
     * Extra arguments for a constructor
     *
     * @param extraArgs the extra arguments for a constructor
     * @return
     */
    public ArrayOps extraArgs(Object[] extraArgs) {
        this.extraArgs = extraArgs;
        return this;
    }

    /**
     * The operation to perform
     *
     * @param clazz the class of the operation to perform
     * @return builder pattern
     */
    public ArrayOps op(ElementWiseOpFactory clazz) {
        this.clazz = clazz;
        return this;
    }


    public ArrayOps from(INDArray from) {
        this.from = from;
        return this;
    }


    public ArrayOps scalar(INDArray scalar) {
        this.scalar = scalar;
        return this;
    }


    public ElementWiseOp build() {
        try {
            ElementWiseOp op;
            if (extraArgs == null)
                op = clazz.create();
            else {
                op = clazz.create(extraArgs);
            }
            BaseElementWiseOp op2 = (BaseElementWiseOp) op;
            op2.from = from;
            op2.scalarValue = scalar;
            return op;
        } catch (Exception e) {
            throw new RuntimeException(e);

        }
    }

}
