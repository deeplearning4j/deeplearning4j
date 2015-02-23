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

package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A base op for basic getters and setters
 * @author Adam Gibson
 */
public abstract class BaseTransformOp extends BaseOp implements TransformOp {
    protected INDArray z;

    /**
     * Specify an alternative result array
     * @param x the input
     * @param z the output paray
     */
    public BaseTransformOp(INDArray x, INDArray z) {
        this(x,z,x.length());
    }

    /**
     * Specify an alternative output array
     * @param x the input
     * @param z the output
     * @param n the number of elements to iterate on
     */
    public BaseTransformOp(INDArray x, INDArray z, int n) {
        super(x,null,n);
        this.z = z;
    }


    public BaseTransformOp(INDArray x, INDArray y, INDArray z, int n) {
        super(x,y,n);
        this.z = z;
    }

    /**
     * An op for one ndarray
     * @param x the ndarray
     */
    public BaseTransformOp(INDArray x) {
        this(x,null,x,x.length());
    }


    @Override
    public INDArray z() {
        return z;
    }


}
